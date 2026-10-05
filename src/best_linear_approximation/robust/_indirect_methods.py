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
        FrequencyDomainSignal,
        RealArray,
        SamplingFrequencyHz,
        TimeDomainSignal,
    )

INDIRECT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=3, u=4, y=4),
        matching_axes=(
            MatchingAxes(signals=("r", "u"), axes=(0, 1, 2)),
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=4, u=5, y=5),
        matching_axes=(
            MatchingAxes(signals=("r", "u"), axes=(0, 1, 2, 3)),
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3, 4)),
        ),
    ),
}


def known_reference(  # noqa: PLR0913
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
    *,
    independent_subexperiments: bool = False,
    show_warnings: bool = True,
) -> NonparametricBLA:
    """Estimate a BLA using a known reference as an instrumental variable.

    Reference, input, and output measurements must all use the same layout:
    either realization layout or experiment layout (see below). All dimensions
    shown below must be present, including singleton dimensions.

    Parameters
    ----------
    r : RealArray
        Periodic reference measurements in realization layout with shape
        ``(n_samples, nu, n_realizations)``, or experiment layout with shape
        ``(n_samples, nu, nu, n_experiments)``.
    u : RealArray
        Periodic input measurements in realization layout with shape
        ``(n_samples, nu, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, nu, nu, n_experiments, n_periods)``.
    y : RealArray
        Periodic output measurements in realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        If provided as an array, specifies strictly increasing indices of the
        excited non-DC, non-Nyquist ``rfft`` bins. Otherwise, if provided as a
        float in ``(0, 1)``, selects them automatically from ``r`` using the
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
    show_warnings : bool, default=True
        If False, suppress warnings emitted by this package about data quality,
        unavailable covariance estimates, and discarded realizations.

    Returns
    -------
    NonparametricBLA
        Best linear approximation with its frequency response, spectra and their
        uncertainties, frequency metadata, and experiment metadata.

    """
    return _estimate_indirect(
        r,
        u,
        y,
        fs,
        excited_bins,
        EstimationMethod.ROBUST_INDIRECT_KNOWN_REFERENCE,
        independent_subexperiments=independent_subexperiments,
        show_warnings=show_warnings,
    )


def closed_loop(  # noqa: PLR0913
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
    *,
    independent_subexperiments: bool = False,
    show_warnings: bool = True,
) -> NonparametricBLA:
    """Estimate a closed-loop BLA using a known reference.

    Reference, input, and output measurements must all use the same layout:
    either realization layout or experiment layout (see below). All dimensions
    shown below must be present, including singleton dimensions.

    Parameters
    ----------
    r : RealArray
        Periodic reference measurements in realization layout with shape
        ``(n_samples, nu, n_realizations)``, or experiment layout with shape
        ``(n_samples, nu, nu, n_experiments)``.
    u : RealArray
        Periodic input measurements in realization layout with shape
        ``(n_samples, nu, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, nu, nu, n_experiments, n_periods)``.
    y : RealArray
        Periodic output measurements in realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        If provided as an array, specifies strictly increasing indices of the
        excited non-DC, non-Nyquist ``rfft`` bins. Otherwise, if provided as a
        float in ``(0, 1)``, selects them automatically from ``r`` using the
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
    show_warnings : bool, default=True
        If False, suppress warnings emitted by this package about data quality,
        unavailable covariance estimates, and discarded realizations.

    Returns
    -------
    NonparametricBLA
        Best linear approximation with its frequency response, spectra and their
        uncertainties, frequency metadata, and experiment metadata.

    """
    return _estimate_indirect(
        r,
        u,
        y,
        fs,
        excited_bins,
        EstimationMethod.ROBUST_INDIRECT_CLOSED_LOOP,
        independent_subexperiments=independent_subexperiments,
        show_warnings=show_warnings,
    )


def _estimate_indirect(  # noqa: PLR0913, PLR0917
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
    estimation_method: EstimationMethod,
    *,
    independent_subexperiments: bool,
    show_warnings: bool,
) -> NonparametricBLA:
    """Estimate an indirect BLA using the specified estimation-method label."""
    r, u, y, fs, excited_bins = _prepare_arguments_indirect(
        r,
        u,
        y,
        fs,
        excited_bins,
        show_warnings=show_warnings,
    )
    G_bla, Z_noise_cov = _compute_bla_indirect(r, u, y, excited_bins)
    spectra = _compute_spectra_indirect(
        r,
        u,
        y,
        G_bla.value,
        Z_noise_cov,
        excited_bins,
        independent_subexperiments=independent_subexperiments,
    )
    freq = create_frequency_info(u.shape[0], fs, excited_bins)
    experiment = ExperimentInfo.from_signals(estimation_method, u, y, r)
    return NonparametricBLA(G_bla, spectra, freq, experiment)


def _prepare_arguments_indirect(  # noqa: PLR0913
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
    *,
    show_warnings: bool,
) -> tuple[TimeDomainSignal, TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve indirect-estimation arguments."""
    return prepare_arguments(
        r,
        u,
        y,
        fs,
        excited_bins,
        INDIRECT_CONTRACTS,
        show_warnings=show_warnings,
    )


def _compute_bla_indirect(
    r: TimeDomainSignal,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[FrequencyResponse, ComplexArray | None]:
    """Compute the best linear approximation and its covariances from noisy input data."""
    ny, nu, n_experiments, n_periods = y.shape[-4:]

    # To excited frequencies
    R = np.fft.rfft(  # (n_excited_bins, n_experiments, 1, nu, nu)
        as_batched_matrices(r),
        axis=0,
    )[excited_bins]
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
        Z = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )
        Z_noise_cov = compute_sample_covariance(vec(Z))

    # Proceed with the period sample means
    U = np.mean(U, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y = np.mean(Y, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Remove singleton reference dimension
    R = np.squeeze(R, axis=2)  # (n_excited_bins, n_experiments, nu, nu)

    # Project the input-output spectra onto the known reference
    reference_projection = R.conj().mT
    U_R = U @ reference_projection  # (n_excited_bins, n_experiments, nu, nu)
    Y_R = Y @ reference_projection  # (n_excited_bins, n_experiments, ny, nu)

    # Project the per-experiment noise covariance onto the known reference
    n_excited_bins = excited_bins.size
    Z_R_noise_cov = None
    if Z_noise_cov is not None:
        n_channels = ny + nu
        I_channels = np.broadcast_to(
            np.eye(n_channels), (n_excited_bins, n_experiments, n_channels, n_channels),
        )
        reference_transform = kronecker_product(R.conj(), I_channels)
        Z_R_noise_cov = np.mean(
            propagate_covariance(Z_noise_cov, reference_transform),
            axis=1,
        ) / (n_experiments * n_periods)

    # Data total covariance: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    Z_R_total_cov = None
    if n_experiments > 1:
        Z_R = np.concatenate(  # (n_excited_bins, n_experiments, ny + nu, nu)
            (Y_R, U_R),
            axis=-2,
        )
        Z_R_total_cov = compute_sample_covariance(vec(Z_R)) / n_experiments

    # Proceed with the experiment sample means
    U_R = np.mean(U_R, axis=1)  # (n_excited_bins, nu, nu)
    Y_R = np.mean(Y_R, axis=1)  # (n_excited_bins, ny, nu)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = compute_frequency_response(U_R, Y_R)

    # BLA covariances
    G_total_cov = None
    G_noise_cov = None
    if Z_R_total_cov is not None or Z_R_noise_cov is not None:
        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        U_R_inv_transpose = np.linalg.solve(U_R, np.eye(nu)).mT if nu > 1 else 1 / U_R
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
        residual_transform = np.concatenate((I_ny, -G), axis=-1)
        jacobian = kronecker_product(U_R_inv_transpose, residual_transform)

        # Total covariance: (n_excited_bins, ny * nu, ny * nu)
        if Z_R_total_cov is not None:
            G_total_cov = propagate_covariance(Z_R_total_cov, jacobian)

        # Noise covariance: (n_excited_bins, ny * nu, ny * nu)
        if Z_R_noise_cov is not None:
            G_noise_cov = propagate_covariance(Z_R_noise_cov, jacobian)

    G_bla = create_bla_frequency_response(
        G,
        G_total_cov,
        G_noise_cov,
        u.shape[0],
        excited_bins,
    )
    return G_bla, Z_noise_cov


def _compute_spectra_indirect(  # noqa: PLR0913, PLR0917
    r: TimeDomainSignal,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    G_bla: ComplexArray,
    Z_noise_cov: ComplexArray | None,
    excited_bins: ExcitedBins,
    *,
    independent_subexperiments: bool,
) -> Spectra:
    """Compute unprojected reference, input, and output spectra."""
    nu, _, n_periods = y.shape[2:]

    R, R_excited = compute_frequency_domain_signal(r, excited_bins)
    U, U_excited = compute_frequency_domain_signal(u, excited_bins)
    Y, Y_excited = compute_frequency_domain_signal(y, excited_bins)

    Y_residual_noise_cov = compute_output_residual_noise_covariance_noisy_input(
        Z_noise_cov,
        G_bla,
        nu,
    )
    H_bla = _compute_reference_to_input_frequency_response(R_excited, U_excited)

    # Input spectrum
    U_noise_cov = compute_noise_covariance(U)
    U_nonlinear_cov = None
    U_total_cov = compute_output_total_covariance(
        U_excited=R_excited,
        Y_excited=U_excited,
        G_bla=H_bla,
        independent_subexperiments=independent_subexperiments,
    )
    if (
        U_total_cov is not None
        and U_noise_cov is not None  # n_periods > 1:
    ):
        U_nonlinear_cov = project_onto_positive_semidefinite(
            U_total_cov - U_noise_cov[excited_bins] / n_periods,
        )
        U_total_cov = U_nonlinear_cov + U_noise_cov[excited_bins]

    input_spectrum = create_input_spectrum(
        U,
        U_noise_cov,
        U_nonlinear_cov,
        U_total_cov,
        n_samples=u.shape[0],
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
        n_samples=u.shape[0],
        excited_bins=excited_bins,
    )

    # Reference spectrum
    reference_spectrum = create_noiseless_input_spectrum(R)

    return Spectra(U=input_spectrum, Y=output_spectrum, R=reference_spectrum)


def _compute_reference_to_input_frequency_response(
    R: FrequencyDomainSignal,
    U: FrequencyDomainSignal,
) -> ComplexArray:
    R = as_batched_matrices(R)
    U = as_batched_matrices(U)

    U_mean = np.mean(U, axis=2)  # average over periods
    R_mean = np.squeeze(R, axis=2)  # remove singleton dimension

    S_RR = np.mean(R_mean @ R_mean.conj().mT, axis=1)
    S_UR = np.mean(U_mean @ R_mean.conj().mT, axis=1)
    return compute_frequency_response(S_RR, S_UR)
