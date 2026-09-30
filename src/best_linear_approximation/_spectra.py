
"""Spectrum result types and covariance computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    propagate_covariance,
)
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation._uncertainty import Uncertainty

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from best_linear_approximation._typing import (
        ComplexArray,
        ExcitedBins,
        FrequencyDomainSignal,
        TimeDomainSignal,
    )


@dataclass(frozen=True)
class InputSpectrum:
    """Input spectrum and its uncertainty estimates.

    Attributes
    ----------
    value : ComplexArray
        Input spectrum at all ``rfft`` bins.
    noise : Uncertainty
        Measurement-noise uncertainty, with covariance ``Cov(U_noise)``
        estimated at all ``rfft`` bins. Requires ``n_periods > 1``.
    nonlinear : Uncertainty
        Nonlinear-distortion uncertainty, with covariance ``Cov(U_nonlinear)``
        estimated at every excited frequency bin. Requires a clean reference
        signal, ``n_periods > 1``, and an estimable ``total`` uncertainty.
    total : Uncertainty
        Total uncertainty, with covariance ``Cov(U_noise) + Cov(U_nonlinear)``
        estimated at every excited frequency bin. Requires a clean reference
        signal, and generally ``n_experiments > 1``, though
        ``n_experiments * nu > 1`` is sufficient when subexperiments are
        independent. A single period per experiment is sufficient.

    """

    value: ComplexArray
    noise: Uncertainty
    nonlinear: Uncertainty
    total: Uncertainty


@dataclass(frozen=True)
class OutputSpectrum:
    """Output spectrum and its uncertainty estimates.

    Attributes
    ----------
    value : ComplexArray
        Output spectrum at all ``rfft`` bins.
    noise : Uncertainty
        Measurement-noise uncertainty, with covariance ``Cov(Y_noise)``
        estimated at all ``rfft`` bins. Requires ``n_periods > 1``.
    nonlinear : Uncertainty
        Nonlinear-distortion uncertainty, with covariance ``Cov(Y_nonlinear)``
        estimated at every excited frequency bin. Requires ``n_periods > 1``
        and an estimable ``total`` uncertainty.
    total : Uncertainty
        Total uncertainty, with covariance ``Cov(Y_noise) + Cov(Y_nonlinear)``
        estimated at every excited frequency bin. Generally requires
        ``n_experiments > 1``, though ``n_experiments * nu > 1`` is sufficient
        when subexperiments are independent. With a noiseless input, it can be
        estimated when ``n_periods == 1``. With a noisy input, ``n_periods > 1``
        is also required.
    total_equation_error : Uncertainty
        Total equation-error uncertainty, with covariance
        ``Cov(Y_nonlinear) + Cov(Y_noise - G_bla @ U_noise)``. Has mostly the same
        requirements as ``total``, but needs only ``n_periods >= 1`` for the
        noisy-input case. Reduces to ``total`` when the input is noiseless.

    """

    value: ComplexArray
    noise: Uncertainty
    nonlinear: Uncertainty
    total: Uncertainty
    total_equation_error: Uncertainty


@dataclass(frozen=True)
class Spectra:
    """Spectra used to estimate a BLA and their spectrum-level uncertainty."""

    U: InputSpectrum
    Y: OutputSpectrum
    R: InputSpectrum | None = None


def create_input_spectrum(  # noqa: PLR0913
    input_signal: FrequencyDomainSignal,
    U_noise_cov: ComplexArray | None,
    U_nonlinear_cov: ComplexArray | None,
    U_total_cov: ComplexArray | None,
    *,
    n_samples: int,
    excited_bins: ExcitedBins,
) -> InputSpectrum:
    """Create an InputSpectrum object with the given noise and uncertainty."""
    nu = input_signal.shape[1]
    return InputSpectrum(
        value=input_signal,
        noise=(
            _create_spectrum_uncertainty(
                U_noise_cov,
                (nu,),
                input_signal,
                n_samples,
                np.arange(input_signal.shape[0]),
            )
            if U_noise_cov is not None
            else Uncertainty.unavailable()
        ),
        nonlinear=(
            _create_spectrum_uncertainty(
                U_nonlinear_cov,
                (nu,),
                input_signal[excited_bins],
                n_samples,
                excited_bins,
            )
            if U_nonlinear_cov is not None
            else Uncertainty.unavailable()
        ),
        total=(
            _create_spectrum_uncertainty(
                U_total_cov,
                (nu,),
                input_signal[excited_bins],
                n_samples,
                excited_bins,
            )
            if U_total_cov is not None
            else Uncertainty.unavailable()
        ),
    )


def compute_frequency_domain_signal(
    signal: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[FrequencyDomainSignal, FrequencyDomainSignal]:
    """Compute a signal's full and excited-bin spectra as FrequencyDomainSignal objects."""
    frequency_domain_signal = cast("FrequencyDomainSignal", np.fft.rfft(signal, axis=0))
    return frequency_domain_signal, cast(
        "FrequencyDomainSignal",
        frequency_domain_signal[excited_bins],
    )


def create_noiseless_input_spectrum(
    input_signal: FrequencyDomainSignal,
) -> InputSpectrum:
    """Create an InputSpectrum object with no noise or uncertainty."""
    unavailable = Uncertainty.unavailable()
    return InputSpectrum(input_signal, unavailable, unavailable, unavailable)


def create_output_spectrum(  # noqa: PLR0913
    output_signal: FrequencyDomainSignal,
    Y_noise_cov: ComplexArray | None,
    Y_nonlinear_cov: ComplexArray | None,
    Y_total_cov: ComplexArray | None,
    Y_total_equation_error_cov: ComplexArray | None,
    *,
    n_samples: int,
    excited_bins: ExcitedBins,
) -> OutputSpectrum:
    """Create an OutputSpectrum object with the given noise and uncertainty."""
    ny = output_signal.shape[1]
    return OutputSpectrum(
        value=output_signal,
        noise=(
            _create_spectrum_uncertainty(
                Y_noise_cov,
                (ny,),
                output_signal,
                n_samples,
                np.arange(output_signal.shape[0]),
            )
            if Y_noise_cov is not None
            else Uncertainty.unavailable()
        ),
        nonlinear=(
            _create_spectrum_uncertainty(
                Y_nonlinear_cov,
                (ny,),
                output_signal[excited_bins],
                n_samples,
                excited_bins,
            )
            if Y_nonlinear_cov is not None
            else Uncertainty.unavailable()
        ),
        total=(
            _create_spectrum_uncertainty(
                Y_total_cov,
                (ny,),
                output_signal[excited_bins],
                n_samples,
                excited_bins,
            )
            if Y_total_cov is not None
            else Uncertainty.unavailable()
        ),
        total_equation_error=(
            _create_spectrum_uncertainty(
                Y_total_equation_error_cov,
                (ny,),
                output_signal[excited_bins],
                n_samples,
                excited_bins,
            )
            if Y_total_equation_error_cov is not None
            else Uncertainty.unavailable()
        ),
    )


def compute_noise_covariance(
    signal: FrequencyDomainSignal,
) -> ComplexArray | None:
    """Compute the spectrum noise covariance over periods."""
    n_periods = signal.shape[-1]
    if n_periods == 1:
        return None

    signal = np.moveaxis(signal, 1, -1)  # (n_bins, nu, n_experiments, n_periods, ny)
    return np.mean(compute_sample_covariance(signal), axis=(1, 2))


def compute_output_total_covariance(
    U_excited: FrequencyDomainSignal,
    Y_excited: FrequencyDomainSignal,
    G_bla: ComplexArray,
    *,
    independent_subexperiments: bool,
) -> ComplexArray | None:
    """Estimate covariance of the period-averaged output residuals.

    The residual is ``Y - G_bla @ U``. Its period-averaged total covariance is

    ``Cov(Y_nonlinear) + Cov(Y_noise - G_bla @ U_noise) / n_periods``.

    If ``independent_subexperiments`` is ``True``, experiment and subexperiment
    samples are treated as independent realizations and pooled before estimating
    the covariance. If ``False``, covariance is estimated across experiments for
    each subexperiment and then averaged across subexperiments.
    """
    n_excited_bins = U_excited.shape[0]
    ny, nu, n_experiments = Y_excited.shape[1:4]

    U_excited = as_batched_matrices(U_excited)
    Y_excited = as_batched_matrices(Y_excited)

    n_realizations = n_experiments * nu if independent_subexperiments else n_experiments
    if n_realizations == 1:
        return None

    # Period-averaged output residual: (n_excited_bins, n_experiments, ny, nu)
    Y_residual = np.mean(Y_excited - G_bla[:, None, None] @ U_excited, axis=2)

    if independent_subexperiments:
        # Estimate covariance jointly across experiments and subexperiments
        Y_residual = Y_residual.transpose(0, 1, 3, 2).reshape(
            n_excited_bins,
            n_realizations,
            ny,
        )
        Y_total_cov = compute_sample_covariance(Y_residual)
    else:
        # Estimate covariance across experiments for each subexperiment, then average
        Y_residual = Y_residual.transpose(0, 3, 1, 2)  # (n_excited_bins, nu, n_experiments, ny)
        Y_total_cov = np.mean(compute_sample_covariance(Y_residual), axis=1)

    return Y_total_cov


def compute_output_residual_noise_covariance_noisy_input(
    Z_noise_cov: ComplexArray | None,
    G_bla: ComplexArray,
    nu: int,
) -> ComplexArray | None:
    """Estimate covariance of the period-averaged output residual noise.

    The residual noise covariance is defined as ``Cov(Y_noise - G_bla @ U_noise)``.
    """
    if Z_noise_cov is None:
        return None

    n_excited_bins, ny = G_bla.shape[:2]

    # Batched Jacobian: (n_excited_bins, (ny + nu) * nu, ny * nu)
    I_nu = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu))
    I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
    residual_transform = np.concatenate((I_ny, -G_bla), axis=-1)
    jacobian = kronecker_product(I_nu, residual_transform)

    # Average over experiments: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    Z_noise_cov = np.mean(Z_noise_cov, axis=1)

    # Residual covariance by subexperiment: (n_excited_bins, ny, ny, nu)
    Y_noise_cov_residual = propagate_covariance(Z_noise_cov, jacobian)
    Y_noise_cov_residual = Y_noise_cov_residual.reshape(n_excited_bins, nu, ny, nu, ny)
    Y_noise_cov_residual = np.diagonal(Y_noise_cov_residual, axis1=1, axis2=3)

    return np.mean(Y_noise_cov_residual, axis=-1)


def _create_spectrum_uncertainty(
    covariance: ComplexArray,
    marginal_var_shape: tuple[int, ...],
    spectrum: ComplexArray,
    n_samples: int,
    frequency_bins: NDArray[np.int_],
) -> Uncertainty:
    """Create uncertainty with the parent spectrum's integration context."""
    return Uncertainty.from_cov(
        covariance,
        marginal_var_shape,
        spectrum=spectrum,
        n_samples=n_samples,
        frequency_bins=frequency_bins,
    )
