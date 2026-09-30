from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._bla import (
    EstimationMethod,
    ExperimentInfo,
    FrequencyResponse,
    NonparametricBLA,
    create_bla_frequency_response,
    create_frequency_info,
)
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    project_onto_positive_semidefinite,
    propagate_covariance,
)
from best_linear_approximation._frequency_response import compute_frequency_response
from best_linear_approximation._linear_algebra import (
    kronecker_product,
    vec,
)
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
)
from best_linear_approximation._spectra import (
    Spectra,
    compute_frequency_domain_signal,
    compute_noise_covariance,
    compute_output_residual_noise_covariance_noisy_input,
    compute_output_total_covariance,
    create_input_spectrum,
    create_noiseless_input_spectrum,
    create_output_spectrum,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from best_linear_approximation._typing import (
        ComplexArray,
        ExcitedBins,
        RealArray,
        SamplingFrequencyHz,
        TimeDomainSignal,
    )


KNOWN_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=None, u=3, y=4),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2)),),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=None, u=4, y=5),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),),
    ),
}


NOISY_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=None, u=4, y=4),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=None, u=5, y=5),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3, 4)),),
    ),
}


def known_input(
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
    *,
    independent_subexperiments: bool = False,
) -> NonparametricBLA:
    """Estimate the BLA from known-input and noisy-output data.

    Input, and output measurements must all use the same layout: either
    realization layout or experiment layout (see below). All dimensions shown
    below must be present, including singleton dimensions.

    Parameters
    ----------
    u : RealArray
        Periodic excitation signals in either realization layout with shape
        ``(n_samples, nu, n_realizations)``, or experiment layout with shape
        ``(n_samples, nu, nu, n_experiments)``.
    y : RealArray
        Periodic output measurements in either realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        If provided as an array, specifies strictly increasing indices of the
        excited non-DC, non-Nyquist ``rfft`` bins. Otherwise, if provided as a
        float in ``(0, 1)``, selects them automatically from ``u`` using the
        mean spectral magnitude across normalized input channels. Bins exceeding
        this fraction of the maximum magnitude are selected.
    independent_subexperiments : bool, default=False
        Only relevant for multi-input data. If True, subexperiments are treated
        as independent realizations when estimating the total covariance, allowing
        experiments and subexperiments to be pooled. This requires an excitation
        design that makes the stochastic nonlinear distortions independent across
        subexperiments, such as full random orthogonal multisines [1, Eq. (3-31)].

        If False, covariance is estimated across experiments separately for
        each subexperiment and then averaged over subexperiments. This makes
        no independence assumption across subexperiments, but uses fewer
        independent samples and is therefore less statistically efficient.

    Returns
    -------
    NonparametricBLA
        Best linear approximation with its frequency response, spectra and their
        uncertainties, frequency metadata, and experiment metadata.


    References
    ----------
    [1] Pintelon, R., and Schoukens, J. (2012).
        *System Identification: A Frequency Domain Approach*, 2nd ed.,
        Wiley-IEEE Press, ISBN 978-0-470-64037-1.

    """
    u, y, fs, excited_bins = _prepare_arguments_known_input(u, y, fs, excited_bins)

    G_bla = _compute_bla_known_input(u, y, excited_bins)
    spectra = _compute_spectra_known_input(
        u,
        y,
        G_bla.value,
        excited_bins,
        independent_subexperiments=independent_subexperiments,
    )
    freq = create_frequency_info(y.shape[0], fs, excited_bins)
    experiment = ExperimentInfo.from_signals(EstimationMethod.ROBUST_DIRECT_KNOWN_INPUT, u, y)

    return NonparametricBLA(G_bla, spectra, freq, experiment)


def noisy_input(
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
    *,
    independent_subexperiments: bool = False,
) -> NonparametricBLA:
    """Estimate the BLA from noisy input-output data.

    Input, and output measurements must all use the same layout: either
    realization layout or experiment layout (see below). All dimensions shown
    below must be present, including singleton dimensions.

    Parameters
    ----------
    u : RealArray
        Periodic input measurements in either realization layout with shape
        ``(n_samples, nu, n_realizations, n_periods)``, or experiment layout with
        shape ``(n_samples, nu, nu, n_experiments, n_periods)``.
    y : RealArray
        Periodic output measurements in either realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        If provided as an array, specifies strictly increasing indices of the
        excited non-DC, non-Nyquist ``rfft`` bins. Otherwise, if provided as a
        float in ``(0, 1)``, selects them automatically from ``u`` using the
        mean spectral magnitude across normalized input channels. Bins exceeding
        this fraction of the maximum magnitude are selected.
    independent_subexperiments : bool, default=False
        Only relevant for multi-input data. If True, subexperiments are treated
        as independent realizations when estimating the total covariance, allowing
        experiments and subexperiments to be pooled. This requires an excitation
        design that makes the stochastic nonlinear distortions independent across
        subexperiments, such as full random orthogonal multisines [1, Eq. (3-31)].

        If False, covariance is estimated across experiments separately for
        each subexperiment and then averaged over subexperiments. This makes
        no independence assumption across subexperiments, but uses fewer
        independent samples and is therefore less statistically efficient.

    Returns
    -------
    NonparametricBLA
        Best linear approximation with its frequency response, spectra and their
        uncertainties, frequency metadata, and experiment metadata.


    References
    ----------
    [1] Pintelon, R., and Schoukens, J. (2012).
        *System Identification: A Frequency Domain Approach*, 2nd ed.,
        Wiley-IEEE Press, ISBN 978-0-470-64037-1.

    """
    u, y, fs, excited_bins = _prepare_arguments_noisy_input(u, y, fs, excited_bins)
    G_bla, Z_noise_cov = _compute_bla_noisy_input(u, y, excited_bins)

    spectra = _compute_spectra_noisy_input(
        u,
        y,
        G_bla.value,
        Z_noise_cov,
        excited_bins,
        independent_subexperiments=independent_subexperiments,
    )
    freq = create_frequency_info(y.shape[0], fs, excited_bins)
    experiment = ExperimentInfo.from_signals(EstimationMethod.ROBUST_DIRECT_NOISY_INPUT, u, y)

    return NonparametricBLA(G_bla, spectra, freq, experiment)


def _prepare_arguments_known_input(
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve all arguments according to :func:`prepare_arguments`."""
    return prepare_arguments(None, u, y, fs, excited_bins, KNOWN_INPUT_CONTRACTS)[1:]


def _prepare_arguments_noisy_input(
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve all arguments according to :func:`prepare_arguments`."""
    return prepare_arguments(None, u, y, fs, excited_bins, NOISY_INPUT_CONTRACTS)[1:]


def _compute_bla_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> FrequencyResponse:
    """Compute the BLA and its uncertainty from known-input and noisy-output data."""
    n_samples, _, _, n_experiments, n_periods = y.shape

    # To excited frequencies
    U = np.fft.rfft(  # (n_excited_bins, n_experiments, 1, nu, nu)
        as_batched_matrices(u),
        axis=0,
    )[excited_bins]
    Y = np.fft.rfft(  # (n_excited_bins, n_experiments, n_periods, ny, nu)
        as_batched_matrices(y),
        axis=0,
    )[excited_bins]

    # Frequency response: (n_excited_bins, n_experiments, n_periods, ny, nu)
    G_per_experiment_and_period = compute_frequency_response(U, Y)

    # Average over periods: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = np.mean(G_per_experiment_and_period, axis=2)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)

    # BLA total covariance: (n_excited_bins, ny * nu, ny * nu)
    G_total_cov = None
    if n_experiments > 1:
        G_total_cov = compute_sample_covariance(vec(G_per_experiment)) / n_experiments

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    G_noise_cov = None
    if n_periods > 1:
        G_noise_cov = np.mean( # (n_excited_bins, n_experiments, 1, nu, nu)
            compute_sample_covariance(vec(G_per_experiment_and_period)),
            axis=1,
        ) / (n_experiments * n_periods)

    return create_bla_frequency_response(
        G,
        G_total_cov,
        G_noise_cov,
        n_samples,
        excited_bins,
    )


def _compute_bla_noisy_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[FrequencyResponse, ComplexArray | None]:
    """Compute the BLA and its uncertainty from noisy input-output data."""
    ny, nu, n_experiments, n_periods = y.shape[-4:]

    # To excited frequencies
    U = np.fft.rfft(  # (n_excited_bins, n_experiments, n_periods, nu, nu)
        as_batched_matrices(u),
        axis=0,
    )[excited_bins]
    Y = np.fft.rfft(  # (n_excited_bins, n_experiments, n_periods, ny, nu)
        as_batched_matrices(y),
        axis=0,
    )[excited_bins]

    # Data noise covariance: (n_excited_bins, n_experiments, (ny + nu) * nu, (ny + nu) * nu)
    Z_noise_cov = None
    if n_periods > 1:
        Z = np.concatenate(  # (n_samples, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )
        Z_noise_cov = compute_sample_covariance(vec(Z))

    # Proceed with the period sample means
    U = np.mean(U, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y = np.mean(Y, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Frequency response: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = compute_frequency_response(U, Y)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)

    # BLA total covariance: (n_excited_bins, ny * nu, ny * nu)
    G_total_cov = None
    if n_experiments > 1:
        G_total_cov = compute_sample_covariance(vec(G_per_experiment)) / n_experiments

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    G_noise_cov = None
    if Z_noise_cov is not None:
        n_excited_bins = G.shape[0]

        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        U_inv_transpose = np.linalg.solve(U, np.eye(nu)).mT if nu > 1 else 1 / U
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        residual_transform = np.concatenate((I_ny, -G_per_experiment), axis=-1)
        jacobian = kronecker_product(U_inv_transpose, residual_transform)

        G_noise_cov = np.mean(
            propagate_covariance(Z_noise_cov, jacobian),
            axis=1,
        ) / (n_experiments * n_periods)

    G_bla = create_bla_frequency_response(
        G,
        G_total_cov,
        G_noise_cov,
        u.shape[0],
        excited_bins,
    )
    return G_bla, Z_noise_cov


def _compute_spectra_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    G_bla: ComplexArray,
    excited_bins: ExcitedBins,
    *,
    independent_subexperiments: bool,
) -> Spectra:
    """Compute input and output spectra from known-input and noisy-output data."""
    n_samples = y.shape[0]
    n_periods = y.shape[-1]

    U, U_excited = compute_frequency_domain_signal(u, excited_bins)
    Y, Y_excited = compute_frequency_domain_signal(y, excited_bins)

    # Input spectrum
    input_spectrum = create_noiseless_input_spectrum(U)

    # Output spectrum
    Y_noise_cov = compute_noise_covariance(Y)
    Y_nonlinear_cov = None
    Y_total_cov = compute_output_total_covariance(
        U_excited,
        Y_excited,
        G_bla,
        independent_subexperiments=independent_subexperiments,
    )
    Y_total_equation_error_cov = Y_total_cov
    if (
        Y_total_cov is not None
        and Y_noise_cov is not None  # n_periods > 1
    ):
        Y_nonlinear_cov = project_onto_positive_semidefinite(
            Y_total_cov - Y_noise_cov[excited_bins] / n_periods,
        )
        Y_total_cov = Y_nonlinear_cov + Y_noise_cov[excited_bins]
        Y_total_equation_error_cov = Y_total_cov

    output_spectrum = create_output_spectrum(
        Y,
        Y_noise_cov,
        Y_nonlinear_cov,
        Y_total_cov,
        Y_total_equation_error_cov,
        n_samples=n_samples,
        excited_bins=excited_bins,
    )

    return Spectra(U=input_spectrum, Y=output_spectrum)


def _compute_spectra_noisy_input(  # noqa: PLR0913
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    G_bla: ComplexArray,
    Z_noise_cov: ComplexArray | None,
    excited_bins: ExcitedBins,
    *,
    independent_subexperiments: bool,
) -> Spectra:
    """Compute input and output spectra from noisy input-output data."""
    nu, _, n_periods = y.shape[2:]

    U, U_excited = compute_frequency_domain_signal(u, excited_bins)
    Y, Y_excited = compute_frequency_domain_signal(y, excited_bins)

    Y_residual_noise_cov = compute_output_residual_noise_covariance_noisy_input(
        Z_noise_cov,
        G_bla,
        nu,
    )

    # Input spectrum
    U_noise_cov = compute_noise_covariance(U)
    U_nonlinear_cov = None
    U_total_cov = None
    input_spectrum = create_input_spectrum(
        U,
        U_noise_cov,
        U_nonlinear_cov,
        U_total_cov,
        n_samples=y.shape[0],
        excited_bins=excited_bins,
    )

    # Output spectrum
    Y_noise_cov = compute_noise_covariance(Y)
    Y_nonlinear_cov = None
    Y_total_cov = None
    Y_total_equation_error_cov = compute_output_total_covariance(
        U_excited,
        Y_excited,
        G_bla,
        independent_subexperiments=independent_subexperiments,
    )
    if (
        Y_total_equation_error_cov is not None
        # n_periods > 1
        and Y_residual_noise_cov is not None
        and Y_noise_cov is not None
    ):
        Y_nonlinear_cov = project_onto_positive_semidefinite(
            Y_total_equation_error_cov - Y_residual_noise_cov / n_periods,
        )
        Y_total_cov = Y_nonlinear_cov + Y_noise_cov[excited_bins]
        Y_total_equation_error_cov = Y_nonlinear_cov + Y_residual_noise_cov

    output_spectrum = create_output_spectrum(
        Y,
        Y_noise_cov,
        Y_nonlinear_cov,
        Y_total_cov,
        Y_total_equation_error_cov,
        n_samples=y.shape[0],
        excited_bins=excited_bins,
    )

    return Spectra(U=input_spectrum, Y=output_spectrum)
