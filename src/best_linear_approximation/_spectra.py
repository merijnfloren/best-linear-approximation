"""Case-agnostic spectrum covariance computations."""

from dataclasses import dataclass

import numpy as np

from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._bla import (
    FrequencyDomainUncertainty,
    InputSpectrum,
    OutputSpectrum,
    Spectra,
)
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    project_onto_positive_semidefinite,
    propagate_covariance,
)
from best_linear_approximation._frequency_response import compute_frequency_response
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation._typing import (
    ComplexArray,
    ExcitedBins,
    FrequencyDomainSignal,
    TimeDomainSignal,
)


@dataclass(frozen=True)
class SpectrumCovarianceInputs:
    """Inputs for separating a spectrum's covariance components.

    Attributes
    ----------
    noise_covariance : ComplexArray or None
        Measurement-noise covariance of the spectrum.
    total_noise_covariance : ComplexArray or None
        Measurement-noise covariance at the same frequency bins as the total
        equation-error covariance. This is the noise contribution to
        ``total_covariance`` when the components can be separated.
    total_equation_error_covariance : ComplexArray or None
        Covariance of the residual between the spectrum and its linear
        prediction.
    equation_error_noise_covariance : ComplexArray or None
        Measurement-noise contribution to the residual covariance.
    predictor_is_noise_free : bool
        Whether the predictor in the residual is noise-free.

    """

    noise_covariance: ComplexArray | None
    total_noise_covariance: ComplexArray | None
    total_equation_error_covariance: ComplexArray | None
    equation_error_noise_covariance: ComplexArray | None
    predictor_is_noise_free: bool


def compute_spectra(
    U: FrequencyDomainSignal,  # noqa: N803
    Y: FrequencyDomainSignal,  # noqa: N803
    input_covariance_inputs: SpectrumCovarianceInputs,
    output_covariance_inputs: SpectrumCovarianceInputs,
    R: FrequencyDomainSignal | None = None,  # noqa: N803
) -> Spectra:
    """Create spectra with all covariance components available from the data.

    Parameters
    ----------
    U : FrequencyDomainSignal
        Input spectrum at all ``rfft`` bins.
    Y : FrequencyDomainSignal
        Output spectrum at all ``rfft`` bins.
    input_covariance_inputs : SpectrumCovarianceInputs
        Covariances used to construct the input spectrum uncertainties.
    output_covariance_inputs : SpectrumCovarianceInputs
        Covariances used to construct the output spectrum uncertainties. Its
        total equation-error covariance is
        ``Cov(Y_nonlinear) + Cov(Y_noise - G_bla @ U_noise)``.
    R : FrequencyDomainSignal or None, default=None
        Optional reference spectrum at all ``rfft`` bins. It is returned
        without uncertainty estimates.

    Returns
    -------
    Spectra
        Input, output, and optional reference spectra. Total covariance is the
        sum of nonlinear and measurement-noise covariance when both can be
        computed. Otherwise, for a noise-free predictor it is computed
        directly from the total equation error.

    """
    input_spectrum = _create_input_spectrum(U, input_covariance_inputs)
    output_spectrum = _create_output_spectrum(Y, output_covariance_inputs)
    reference_spectrum = _create_reference_spectrum(R) if R is not None else None
    return Spectra(input_spectrum, output_spectrum, reference_spectrum)


def compute_noise_covariance(
    spectrum: FrequencyDomainSignal,
) -> ComplexArray | None:
    """Compute the period-averaged measurement-noise covariance.

    Parameters
    ----------
    spectrum : FrequencyDomainSignal
        Spectrum with canonical shape
        ``(n_freqs, n_channels, nu, n_experiments, n_periods)``.

    Returns
    -------
    ComplexArray or None
        Measurement-noise covariance with shape
        ``(n_freqs, n_channels, n_channels)``, or ``None`` when there is only
        one period.

    """
    n_periods = spectrum.shape[-1]
    if n_periods == 1:
        return None

    samples = np.moveaxis(spectrum, 1, -1)
    return np.mean(compute_sample_covariance(samples), axis=(1, 2))


def compute_reference_to_input_frequency_response(
    R: FrequencyDomainSignal,  # noqa: N803
    U: FrequencyDomainSignal,  # noqa: N803
) -> ComplexArray:
    """Estimate the reference-to-input frequency response.

    Parameters
    ----------
    R : FrequencyDomainSignal
        Reference spectrum at the excited frequency bins.
    U : FrequencyDomainSignal
        Input spectrum at the excited frequency bins.

    Returns
    -------
    ComplexArray
        Reference-to-input frequency response at the excited frequency bins.

    """
    reference = as_batched_matrices(R)
    input_ = as_batched_matrices(U)
    reference = np.squeeze(reference, axis=2)
    input_ = np.mean(input_, axis=2)
    reference_spectrum = np.mean(reference @ reference.conj().mT, axis=1)
    input_reference_spectrum = np.mean(input_ @ reference.conj().mT, axis=1)
    return compute_frequency_response(reference_spectrum, input_reference_spectrum)


def compute_residual_noise_covariance(
    joint_noise_covariance: ComplexArray | None,
    frequency_response: ComplexArray,
    nu: int,
) -> ComplexArray | None:
    """Compute covariance of period-averaged residual measurement noise.

    Parameters
    ----------
    joint_noise_covariance : ComplexArray or None
        Joint input-output measurement-noise covariance, with shape
        ``(n_freqs, n_experiments, (ny + nu) * nu, (ny + nu) * nu)``.
    frequency_response : ComplexArray
        Response-to-predictor frequency response, with shape
        ``(n_freqs, ny, nu)``.
    nu : int
        Number of predictor channels and subexperiments.

    Returns
    -------
    ComplexArray or None
        Residual measurement-noise covariance with shape
        ``(n_freqs, ny, ny)``, or ``None`` when the joint noise covariance is
        unavailable.

    """
    if joint_noise_covariance is None:
        return None

    n_freqs, ny = frequency_response.shape[:2]
    identity_nu = np.broadcast_to(np.eye(nu), (n_freqs, nu, nu))
    identity_ny = np.broadcast_to(np.eye(ny), (n_freqs, ny, ny))
    residual_transform = np.concatenate((identity_ny, -frequency_response), axis=-1)
    jacobian = kronecker_product(identity_nu, residual_transform)

    mean_noise_covariance = np.mean(joint_noise_covariance, axis=1)
    residual_covariance = propagate_covariance(mean_noise_covariance, jacobian)
    residual_covariance = residual_covariance.reshape(n_freqs, nu, ny, nu, ny)
    residual_by_subexperiment = np.diagonal(residual_covariance, axis1=1, axis2=3)
    return np.mean(residual_by_subexperiment, axis=-1)


def compute_residual_total_covariance(
    predictor: FrequencyDomainSignal,
    response: FrequencyDomainSignal,
    frequency_response: ComplexArray,
    *,
    independent_subexperiments: bool,
) -> ComplexArray | None:
    """Estimate covariance of period-averaged response residuals.

    The residual is ``response - frequency_response @ predictor``. If
    ``independent_subexperiments`` is true, experiment-subexperiment pairs are
    pooled as independent realizations. Otherwise, covariance is estimated
    across experiments for each subexperiment and then averaged.

    Parameters
    ----------
    predictor : FrequencyDomainSignal
        Predictor spectrum at the excited bins.
    response : FrequencyDomainSignal
        Response spectrum at the excited bins.
    frequency_response : ComplexArray
        Linear response from ``predictor`` to ``response``.
    independent_subexperiments : bool
        Whether subexperiments are independent realizations.

    Returns
    -------
    ComplexArray or None
        Residual total covariance, or ``None`` when fewer than two independent
        realizations are available.

    """
    batched_predictor = as_batched_matrices(predictor)
    batched_response = as_batched_matrices(response)
    n_freqs, n_experiments, _, ny, nu = batched_response.shape
    n_realizations = n_experiments * nu if independent_subexperiments else n_experiments
    if n_realizations == 1:
        return None

    residual = np.mean(
        batched_response - frequency_response[:, None, None] @ batched_predictor,
        axis=2,
    )
    if independent_subexperiments:
        samples = residual.transpose(0, 1, 3, 2).reshape(n_freqs, n_realizations, ny)
        return compute_sample_covariance(samples)

    samples = residual.transpose(0, 3, 1, 2)
    return np.mean(compute_sample_covariance(samples), axis=1)


def _compute_spectra_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    frequency_response: ComplexArray,
    excited_bins: ExcitedBins,
    *,
    independent_subexperiments: bool,
) -> Spectra:
    """Compute spectra for noiseless input and noisy output data."""
    n_periods = y.shape[-1]
    U = FrequencyDomainSignal(np.fft.rfft(u, axis=0))
    Y = FrequencyDomainSignal(np.fft.rfft(y, axis=0))
    U_excited = FrequencyDomainSignal(U[excited_bins])
    Y_excited = FrequencyDomainSignal(Y[excited_bins])

    output_noise_covariance = compute_noise_covariance(Y)
    total_equation_error_covariance = compute_residual_total_covariance(
        U_excited,
        Y_excited,
        frequency_response,
        independent_subexperiments=independent_subexperiments,
    )
    equation_error_noise_covariance = (
        output_noise_covariance[excited_bins] / n_periods
        if output_noise_covariance is not None
        else None
    )

    input_covariance_inputs = SpectrumCovarianceInputs(
        noise_covariance=None,
        total_noise_covariance=None,
        total_equation_error_covariance=None,
        equation_error_noise_covariance=None,
        predictor_is_noise_free=True,
    )
    output_covariance_inputs = SpectrumCovarianceInputs(
        noise_covariance=output_noise_covariance,
        total_noise_covariance=(
            output_noise_covariance[excited_bins]
            if output_noise_covariance is not None
            else None
        ),
        total_equation_error_covariance=total_equation_error_covariance,
        equation_error_noise_covariance=equation_error_noise_covariance,
        predictor_is_noise_free=True,
    )
    return compute_spectra(
        U,
        Y,
        input_covariance_inputs,
        output_covariance_inputs,
    )


def _create_input_spectrum(
    U: FrequencyDomainSignal,  # noqa: N803
    covariance_inputs: SpectrumCovarianceInputs,
) -> InputSpectrum:
    """Create an input spectrum from covariance inputs."""
    nu = U.shape[1]
    covariances = _separate_covariances(covariance_inputs)
    return InputSpectrum(
        value=U,
        noise=_create_uncertainty(covariances.noise, nu),
        nonlinear=_create_uncertainty(covariances.nonlinear, nu),
        total=_create_uncertainty(covariances.total, nu),
    )


def _create_output_spectrum(
    Y: FrequencyDomainSignal,  # noqa: N803
    covariance_inputs: SpectrumCovarianceInputs,
) -> OutputSpectrum:
    """Create an output spectrum from covariance inputs."""
    ny = Y.shape[1]
    covariances = _separate_covariances(covariance_inputs)
    return OutputSpectrum(
        value=Y,
        noise=_create_uncertainty(covariances.noise, ny),
        nonlinear=_create_uncertainty(covariances.nonlinear, ny),
        total=_create_uncertainty(covariances.total, ny),
        total_equation_error=_create_uncertainty(covariances.total_equation_error, ny),
    )


def _create_reference_spectrum(R: FrequencyDomainSignal) -> InputSpectrum:  # noqa: N803
    """Create a reference spectrum with unavailable uncertainty estimates."""
    return InputSpectrum(
        value=R,
        noise=FrequencyDomainUncertainty.unavailable(),
        nonlinear=FrequencyDomainUncertainty.unavailable(),
        total=FrequencyDomainUncertainty.unavailable(),
    )


@dataclass(frozen=True)
class _SpectrumCovariances:
    """Separated covariance components of a spectrum."""

    noise: ComplexArray | None
    nonlinear: ComplexArray | None
    total: ComplexArray | None
    total_equation_error: ComplexArray | None


def _separate_covariances(
    covariance_inputs: SpectrumCovarianceInputs,
) -> _SpectrumCovariances:
    """Separate covariance inputs into the reported spectrum components."""
    nonlinear_covariance = None
    total_covariance = None
    total_equation_error_covariance = covariance_inputs.total_equation_error_covariance

    if total_equation_error_covariance is not None:
        if (
            covariance_inputs.equation_error_noise_covariance is not None
            and covariance_inputs.total_noise_covariance is not None
        ):
            nonlinear_covariance = project_onto_positive_semidefinite(
                total_equation_error_covariance
                - covariance_inputs.equation_error_noise_covariance,
            )
            total_covariance = (
                nonlinear_covariance + covariance_inputs.total_noise_covariance
            )
        elif covariance_inputs.predictor_is_noise_free:
            total_covariance = total_equation_error_covariance

    return _SpectrumCovariances(
        noise=covariance_inputs.noise_covariance,
        nonlinear=nonlinear_covariance,
        total=total_covariance,
        total_equation_error=total_equation_error_covariance,
    )


def _create_uncertainty(
    covariance: ComplexArray | None,
    n_channels: int,
) -> FrequencyDomainUncertainty:
    """Create an uncertainty object from an optional covariance."""
    if covariance is None:
        return FrequencyDomainUncertainty.unavailable()

    return FrequencyDomainUncertainty.from_cov(covariance, (n_channels,))
