
import numpy as np

from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._bla import (
    FrequencyDomainUncertainty,
    InputSpectrum,
    OutputSpectrum,
)
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    propagate_covariance,
)
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation._typing import (
    ComplexArray,
    ExcitedBins,
    FrequencyDomainSignal,
    TimeDomainSignal,
)


def create_input_spectrum(
    input_signal: FrequencyDomainSignal,
    U_noise_cov: ComplexArray | None,  # noqa: N803
    U_nonlinear_cov: ComplexArray | None,  # noqa: N803
    U_total_cov: ComplexArray | None,  # noqa: N803
) -> InputSpectrum:
    """Create an InputSpectrum object with the given noise and uncertainty."""
    nu = input_signal.shape[1]
    return InputSpectrum(
        value=input_signal,
        noise=(
            FrequencyDomainUncertainty.from_cov(U_noise_cov, (nu,))
            if U_noise_cov is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        nonlinear=(
            FrequencyDomainUncertainty.from_cov(U_nonlinear_cov, (nu,))
            if U_nonlinear_cov is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        total=(
            FrequencyDomainUncertainty.from_cov(U_total_cov, (nu,))
            if U_total_cov is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
    )


def compute_frequency_domain_signal(
    signal: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[FrequencyDomainSignal, FrequencyDomainSignal]:
    """Compute a signal's full and excited-bin spectra as FrequencyDomainSignal objects."""
    frequency_domain_signal = FrequencyDomainSignal(np.fft.rfft(signal, axis=0))
    return frequency_domain_signal, FrequencyDomainSignal(
        frequency_domain_signal[excited_bins],
    )


def create_noiseless_input_spectrum(
    input_signal: FrequencyDomainSignal,
) -> InputSpectrum:
    """Create an InputSpectrum object with no noise or uncertainty."""
    return create_input_spectrum(input_signal, None, None, None)


def create_output_spectrum(
    output_signal: FrequencyDomainSignal,
    Y_noise_cov: ComplexArray | None,  # noqa: N803
    Y_nonlinear_cov: ComplexArray | None,  # noqa: N803
    Y_total_cov: ComplexArray | None,  # noqa: N803
    Y_total_equation_error_cov: ComplexArray | None,  # noqa: N803
) -> OutputSpectrum:
    """Create an OutputSpectrum object with the given noise and uncertainty."""
    ny = output_signal.shape[1]
    return OutputSpectrum(
        value=output_signal,
        noise=(
            FrequencyDomainUncertainty.from_cov(Y_noise_cov, (ny,))
            if Y_noise_cov is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        nonlinear=(
            FrequencyDomainUncertainty.from_cov(Y_nonlinear_cov, (ny,))
            if Y_nonlinear_cov is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        total=(
            FrequencyDomainUncertainty.from_cov(Y_total_cov, (ny,))
            if Y_total_cov is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        total_equation_error=(
            FrequencyDomainUncertainty.from_cov(Y_total_equation_error_cov, (ny,))
            if Y_total_equation_error_cov is not None
            else FrequencyDomainUncertainty.unavailable()
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
    U_excited: FrequencyDomainSignal,  # noqa: N803
    Y_excited: FrequencyDomainSignal,  # noqa: N803
    G_bla: ComplexArray,  # noqa: N803
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
    Z_noise_cov: ComplexArray | None,  # noqa: N803
    G_bla: ComplexArray,  # noqa: N803
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
