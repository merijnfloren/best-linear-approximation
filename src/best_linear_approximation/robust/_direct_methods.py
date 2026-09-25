from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._bla import (
    EstimationMethod,
    ExperimentInfo,
    FrequencyDomainUncertainty,
    FrequencyResponse,
    InputSpectrum,
    NonparametricBLA,
    OutputSpectrum,
    Spectra,
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
from best_linear_approximation._typing import (
    ComplexArray,
    ExcitedBins,
    FrequencyDomainSignal,
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

    Parameters
    ----------
    u : RealArray
        Periodic excitation signals in either realization layout with shape
        ``(n_samples, nu, n_realizations)``, or experiment layout with shape
        ``(n_samples, nu, nu, n_experiments)``.
    y : RealArray
        Periodic output measurements in either realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``. Note
        that ``y`` always has one more dimension than ``u``, even when
        ``n_periods == 1``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        Strictly increasing indices of the excited non-DC, non-Nyquist ``rfft``
        bins. A float in ``(0, 1)`` instead selects them automatically from
        ``u``: bins whose channel-averaged spectral magnitude exceeds this
        fraction of the maximum magnitude are selected.
    independent_subexperiments : bool, default=False
        Only relevant for multi-input experiment-layout data. If True,
        subexperiments are treated as independent realizations when estimating
        the total covariance, allowing experiments and subexperiments to be
        pooled. This requires an excitation design that makes the stochastic
        nonlinear distortions independent across subexperiments, such as full
        random orthogonal multisines [1, Eq. (3-31)].

        If False, covariance is estimated across experiments separately for
        each subexperiment and then averaged over subexperiments. This makes
        no independence assumption across subexperiments, but uses fewer
        independent samples and is therefore less statistically efficient.

    Returns
    -------
    NonparametricBLA
        Frequency response and available total, noise, and nonlinear covariances.
        
    
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
        Strictly increasing indices of the excited non-DC, non-Nyquist ``rfft``
        bins. A float in ``(0, 1)`` instead selects them automatically from
        ``u``: bins whose channel-averaged spectral magnitude exceeds this
        fraction of the maximum magnitude are selected.
    independent_subexperiments : bool, default=False
        Only relevant for multi-input experiment-layout data. If True,
        subexperiments are treated as independent realizations when estimating
        the total covariance, allowing experiments and subexperiments to be
        pooled. This requires an excitation design that makes the stochastic
        nonlinear distortions independent across subexperiments, such as full
        random orthogonal multisines [1, Eq. (3-31)].

        If False, covariance is estimated across experiments separately for
        each subexperiment and then averaged over subexperiments. This makes
        no independence assumption across subexperiments, but uses fewer
        independent samples and is therefore less statistically efficient.

    Returns
    -------
    NonparametricBLA
        Frequency response and available total, noise, and nonlinear covariances.
        
    
    References
    ----------
    [1] Pintelon, R., and Schoukens, J. (2012).
        *System Identification: A Frequency Domain Approach*, 2nd ed.,
        Wiley-IEEE Press, ISBN 978-0-470-64037-1.

    """
    u, y, fs, excited_bins = _prepare_arguments_noisy_input(u, y, fs, excited_bins)
    G_bla, cov_Z_noise = _compute_bla_noisy_input(u, y, excited_bins)

    spectra = _compute_spectra_noisy_input(
        u,
        y,
        G_bla.value,
        cov_Z_noise,
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
    n_experiments, n_periods = y.shape[-2:]

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
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(vec(G_per_experiment)) / n_experiments
    else:
        G_cov_total = None

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_periods > 1:
        G_cov_noise = np.mean(
            compute_sample_covariance(vec(G_per_experiment_and_period)),
            axis=1,
        ) / (n_experiments * n_periods)
    else:
        G_cov_noise = None

    return create_bla_frequency_response(G, G_cov_total, G_cov_noise)


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
    if n_periods > 1:
        Z = np.concatenate(  # (n_samples, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )
        cov_Z_noise = compute_sample_covariance(vec(Z)) / n_periods
    else:
        cov_Z_noise = None

    # Proceed with the period sample means
    U = np.mean(U, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y = np.mean(Y, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Frequency response: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = compute_frequency_response(U, Y)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)

    # BLA total covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(vec(G_per_experiment)) / n_experiments
    else:
        G_cov_total = None

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if cov_Z_noise is not None:
        n_excited_bins = G.shape[0]

        # Batched U^(-T)
        U_inv_transpose = np.linalg.solve(U, np.eye(nu)).mT if nu > 1 else 1 / U

        # Batched V = [I_ny, -G]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        V = np.concatenate((I_ny, -G_per_experiment), axis=-1)

        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        jacobian = kronecker_product(U_inv_transpose, V)

        G_cov_noise = np.mean(propagate_covariance(cov_Z_noise, jacobian), axis=1) / n_experiments
    else:
        G_cov_noise = None

    G_bla = create_bla_frequency_response(G, G_cov_total, G_cov_noise)
    return G_bla, cov_Z_noise 


def _compute_spectra_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    G_bla: ComplexArray,  # noqa: N803
    excited_bins: ExcitedBins,
    *,
    independent_subexperiments: bool,
) -> Spectra:
    """Compute input and output spectra from known-input and noisy-output data."""
    ny = y.shape[1]
    n_periods = y.shape[-1]

    U = FrequencyDomainSignal(np.fft.rfft(u, axis=0))
    Y = FrequencyDomainSignal(np.fft.rfft(y, axis=0))
    U_excited = FrequencyDomainSignal(U[excited_bins])
    Y_excited = FrequencyDomainSignal(Y[excited_bins])

    input_spectrum = InputSpectrum(
        value=U,
        total=FrequencyDomainUncertainty.unavailable(),
        nonlinear=FrequencyDomainUncertainty.unavailable(),
        noise=FrequencyDomainUncertainty.unavailable(),
    )

    Y_cov_noise = _compute_noise_covariance(Y)
    Y_cov_total_per_experiment = _compute_output_total_covariance(
        U_excited,
        Y_excited,
        G_bla,
        independent_subexperiments=independent_subexperiments,
    )
    
    Y_cov_nonlinear: ComplexArray | None = None
    Y_cov_total: ComplexArray | None = None
    if Y_cov_total_per_experiment is not None:
        if Y_cov_noise is None:
            # Noise covariance is unavailable only when there is one period. In that
            # case, period averaging does not change the total covariance, so:
            Y_cov_total = Y_cov_total_per_experiment
        else:
            Y_cov_nonlinear = project_onto_positive_semidefinite(
                Y_cov_total_per_experiment - Y_cov_noise[excited_bins] / n_periods,
            )
            Y_cov_total = Y_cov_nonlinear + Y_cov_noise[excited_bins]
            
    output_total_uncertainty = (
        FrequencyDomainUncertainty.from_cov(Y_cov_total, (ny,))
        if Y_cov_total is not None
        else FrequencyDomainUncertainty.unavailable()
    )
    
    output_spectrum = OutputSpectrum(
        value=Y,
        total=output_total_uncertainty,
        nonlinear=(
            FrequencyDomainUncertainty.from_cov(Y_cov_nonlinear, (ny,))
            if Y_cov_nonlinear is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        noise=(
            FrequencyDomainUncertainty.from_cov(Y_cov_noise, (ny,))
            if Y_cov_noise is not None
            else FrequencyDomainUncertainty.unavailable()
        ),
        total_equation_error=output_total_uncertainty,
    )
    return Spectra(input_spectrum, output_spectrum)


# def _compute_spectra_noisy_input(
#     u: TimeDomainSignal,
#     y: TimeDomainSignal,
#     G_bla: ComplexArray,  # noqa: N803
#     cov_Z_noise: ComplexArray | None,
#     excited_bins: ExcitedBins,
#     *,
#     independent_subexperiments: bool,
# ) -> Spectra:
#     """Compute input and output spectra from noisy input-output data."""
#     ny, nu, _, _ = y.shape[1:]

#     U = FrequencyDomainSignal(np.fft.rfft(u, axis=0))
#     Y = FrequencyDomainSignal(np.fft.rfft(y, axis=0))
#     U_excited = FrequencyDomainSignal(U[excited_bins])
#     Y_excited = FrequencyDomainSignal(Y[excited_bins])
    
#     U_cov_noise = _compute_noise_covariance(U)
#     Y_cov_noise = _compute_noise_covariance(Y)
    
#     Y_cov_residual_noise_per_experiment = _compute_residual_noise_covariance_noisy_input(
#         cov_Z_noise,
#         G_bla,
#         nu,
#     )

#     input_spectrum = InputSpectrum(
#         value=U,
#         total=FrequencyDomainUncertainty.unavailable(),
#         nonlinear=FrequencyDomainUncertainty.unavailable(),
#         noise=(
#             FrequencyDomainUncertainty.from_cov(U_cov_noise, (nu,))
#             if U_cov_noise is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#     )
    
#     Y_cov_total_per_experiment = _compute_output_total_covariance(
#         U_excited,
#         Y_excited,
#         G_bla,
#         independent_subexperiments=independent_subexperiments,
#     )

#     Y_cov_nonlinear: ComplexArray | None = None
#     Y_cov_total: ComplexArray | None = None
#     if Y_cov_total_per_experiment is not None:
#         if Y_cov_noise is None:
#             # Noise covariance is unavailable only when there is one period. In that
#             # case, period averaging does not change the total covariance.
#             Y_cov_total = Y_cov_total_per_experiment
#         else:
#             Y_cov_nonlinear = project_onto_positive_semidefinite(
#                 Y_cov_total_per_experiment - Y_cov_residual_noise_per_experiment,
#             )
#             Y_cov_total = Y_cov_nonlinear + Y_cov_noise[excited_bins]

#     output_spectrum = OutputSpectrum(
#         value=Y,
#         total=(
#             FrequencyDomainUncertainty.from_cov(Y_cov_total, (ny,))
#             if Y_cov_total is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#         nonlinear=(
#             FrequencyDomainUncertainty.from_cov(Y_cov_nonlinear, (ny,))
#             if Y_cov_nonlinear is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#         noise=(
#             FrequencyDomainUncertainty.from_cov(Y_cov_noise, (ny,))
#             if Y_cov_noise is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#     )
#     return Spectra(input_spectrum, output_spectrum)


def _compute_noise_covariance(
    signal: FrequencyDomainSignal,
) -> ComplexArray | None:
    """Compute the averaged noise covariance over periods."""
    n_periods = signal.shape[-1]
    if n_periods == 1:
        return None

    signal = np.moveaxis(signal, 1, -1)  # (n_bins, nu, n_experiments, n_periods, ny)
    return np.mean(compute_sample_covariance(signal), axis=(1, 2))


def _compute_residual_noise_covariance_noisy_input(
    cov_Z_noise: ComplexArray | None,
    G_bla: ComplexArray,  # noqa: N803
    nu: int,
) -> ComplexArray | None:
    """Compute covariance of the period-averaged residual measurement noise."""
    if cov_Z_noise is None:
        return None

    n_excited_bins, n_experiments = cov_Z_noise.shape[:2]
    ny = G_bla.shape[1]
    I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
    residual_transform = np.concatenate((I_ny, -G_bla), axis=-1)
    I_nu = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu))
    jacobian = kronecker_product(I_nu, residual_transform)[:, None]
    cov_residual = propagate_covariance(cov_Z_noise, jacobian)
    cov_residual = cov_residual.reshape(n_excited_bins, n_experiments, nu, ny, nu, ny)
    cov_by_input_direction = np.diagonal(cov_residual, axis1=2, axis2=4)
    return np.mean(cov_by_input_direction, axis=(1, -1))


def _compute_output_total_covariance(
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
        Y_residual = Y_residual.transpose(0, 1, 3, 2).reshape(n_excited_bins, n_realizations, ny)
        Y_cov_total = compute_sample_covariance(Y_residual)
    else:
        # Estimate covariance across experiments for each subexperiment, then average
        Y_residual = Y_residual.transpose(0, 3, 1, 2)  # (n_excited_bins, nu, n_experiments, ny)
        Y_cov_total = np.mean(compute_sample_covariance(Y_residual), axis=1)

    return Y_cov_total


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from best_linear_approximation._dataloader import (
        load_fine_steering_mirror,
        load_parallel_wiener_hammerstein,
    )

    data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]
    u = np.mean(data.u, axis=-1)  # average over periods
    bla = known_input(u, data.y, data.fs, data.excited_bins)
    freqs = bla.freq.freqs
    excited_freqs = freqs[bla.freq.excited_bins]

    U = bla.spectra.U.value
    Y = bla.spectra.Y.value
    reduction_axes = (2, 3, 4)
    U_magnitude = np.mean(np.abs(U), axis=reduction_axes)
    Y_magnitude = np.mean(np.abs(Y), axis=reduction_axes)

    def plot_uncertainties(
        axis: Any,
        estimate: FrequencyResponse | InputSpectrum | OutputSpectrum,
        label: str,
        indices: tuple[int, ...],
    ) -> None:
        """Plot available marginal standard deviations for one spectral component."""
        for uncertainty_name in ("total", "nonlinear", "noise"):
            uncertainty = getattr(estimate, uncertainty_name)
            standard_deviation = uncertainty.std
            if standard_deviation is None:
                continue

            uncertainty_freqs = (
                freqs if standard_deviation.shape[0] == freqs.size else excited_freqs
            )
            component = standard_deviation[(slice(None), *indices)]
            axis.plot(
                uncertainty_freqs,
                20 * np.log10(component),
                linestyle="--",
                label=f"{label} {uncertainty_name} std",
            )

    fig, axes = plt.subplots(3, 1, sharex=True, layout="constrained")
    for input_channel in range(bla.experiment.nu):
        axes[0].plot(freqs, 20 * np.log10(U_magnitude[:, input_channel]), label=f"U[{input_channel}]")
        plot_uncertainties(axes[0], bla.spectra.U, f"U[{input_channel}]", (input_channel,))
    for output_channel in range(bla.experiment.ny):
        axes[1].plot(freqs, 20 * np.log10(Y_magnitude[:, output_channel]), label=f"Y[{output_channel}]")
        plot_uncertainties(axes[1], bla.spectra.Y, f"Y[{output_channel}]", (output_channel,))
        for input_channel in range(bla.experiment.nu):
            label = f"G[{output_channel}, {input_channel}]"
            axes[2].plot(
                excited_freqs,
                20 * np.log10(np.abs(bla.G.value[:, output_channel, input_channel])),
                label=label,
            )
            plot_uncertainties(axes[2], bla.G, label, (output_channel, input_channel))

    axes[0].set_ylabel("input magnitude [dB]")
    axes[1].set_ylabel("output magnitude [dB]")
    axes[2].set_ylabel("FRF magnitude [dB]")
    axes[2].set_xlabel("frequency [Hz]")
    for axis in axes:
        axis.legend()

    data = load_fine_steering_mirror()["train 300mV"]
    u = np.mean(data.u, axis=-1)
    bla = known_input(u, data.y, data.fs, data.excited_bins)
    freqs = bla.freq.freqs
    excited_freqs = freqs[bla.freq.excited_bins]
    U = bla.spectra.U.value
    Y = bla.spectra.Y.value
    U_magnitude = np.mean(np.abs(U), axis=reduction_axes)
    Y_magnitude = np.mean(np.abs(Y), axis=reduction_axes)

    fig_spectra, spectrum_axes = plt.subplots(
        bla.experiment.ny,
        2,
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    for channel in range(bla.experiment.ny):
        input_axis, output_axis = spectrum_axes[channel]
        input_axis.plot(freqs, 20 * np.log10(U_magnitude[:, channel]), label=f"U[{channel}]")
        plot_uncertainties(input_axis, bla.spectra.U, f"U[{channel}]", (channel,))
        output_axis.plot(freqs, 20 * np.log10(Y_magnitude[:, channel]), label=f"Y[{channel}]")
        plot_uncertainties(output_axis, bla.spectra.Y, f"Y[{channel}]", (channel,))
        input_axis.set_ylabel(f"channel {channel} [dB]")
        input_axis.legend()
        output_axis.legend()

    spectrum_axes[0, 0].set_title("input spectra")
    spectrum_axes[0, 1].set_title("output spectra")
    spectrum_axes[-1, 0].set_xlabel("frequency [Hz]")
    spectrum_axes[-1, 1].set_xlabel("frequency [Hz]")

    fig_frf, frf_axes = plt.subplots(
        bla.experiment.ny,
        bla.experiment.nu,
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    for output_channel in range(bla.experiment.ny):
        for input_channel in range(bla.experiment.nu):
            axis = frf_axes[output_channel, input_channel]
            label = f"G[{output_channel}, {input_channel}]"
            axis.plot(
                excited_freqs,
                20 * np.log10(np.abs(bla.G.value[:, output_channel, input_channel])),
                label=label,
            )
            plot_uncertainties(axis, bla.G, label, (output_channel, input_channel))
            axis.legend()

    for input_channel in range(bla.experiment.nu):
        frf_axes[-1, input_channel].set_xlabel("frequency [Hz]")
    for output_channel in range(bla.experiment.ny):
        frf_axes[output_channel, 0].set_ylabel(f"output {output_channel} [dB]")
    plt.show()


if __name__ == "__main__" and False:
    from best_linear_approximation._dataloader import (
        load_f16,
        load_fine_steering_mirror,
        load_parallel_wiener_hammerstein,
    )

    data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]

    bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    G = bla.G.value
    std_total, std_noise, std_nonlinear = bla.G.total.std, bla.G.noise.std, bla.G.nonlinear.std

    import matplotlib.pyplot as plt


    def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return 20 * np.log10(np.abs(magnitude))

    plt.figure()
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(std_total[:, 0, 0]), label="std_total[0, 0]")
    plt.plot(to_db(std_noise[:, 0, 0]), label="std_noise[0, 0]")
    plt.plot(to_db(std_nonlinear[:, 0, 0]), label="std_nonlinear[0, 0]", linestyle="--")
    plt.legend()
    plt.show()

    # data = load_silverbox()["train SB multisine"]
    # bla = noisy_input(data.u, data.y, data.fs)
    # G = bla.G.value
    # std_total, std_noise, std_nonlinear = bla.G.total.std, bla.G.noise.std, bla.G.nonlinear.std


    # plt.figure()
    # plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    # plt.plot(to_db(std_total[:, 0, 0]), label="std_total[0, 0]")
    # plt.plot(to_db(std_noise[:, 0, 0]), label="std_noise[0, 0]")
    # plt.plot(to_db(std_nonlinear[:, 0, 0]), label="std_nonlinear[0, 0]")
    # plt.legend()
    # plt.show()

    data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    G = bla.G.value
    std_total, std_noise, std_nonlinear = bla.G.total.std, bla.G.noise.std, bla.G.nonlinear.std


    # create 3x1 subplots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(std_total[:, 0, 0]), label="std_total[0, 0]")
    plt.plot(to_db(std_noise[:, 0, 0]), label="std_noise[0, 0]")
    plt.plot(to_db(std_nonlinear[:, 0, 0]), label="std_nonlinear[0, 0]")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    plt.plot(to_db(std_total[:, 1, 0]), label="std_total[1, 0]")
    plt.plot(to_db(std_noise[:, 1, 0]), label="std_noise[1, 0]")
    plt.plot(to_db(std_nonlinear[:, 1, 0]), label="std_nonlinear[1, 0]")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    plt.plot(to_db(std_total[:, 2, 0]), label="std_total[2, 0]")
    plt.plot(to_db(std_noise[:, 2, 0]), label="std_noise[2, 0]")
    plt.plot(to_db(std_nonlinear[:, 2, 0]), label="std_nonlinear[2, 0]")
    plt.legend()
    plt.show()

    # data = load_f16()["F16Data_FullMSine_Level3.mat"]
    # bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    # G = bla.G.value
    # std_total, std_noise, std_nonlinear = bla.G.total.std, bla.G.noise.std, bla.G.nonlinear.std

    # # create 3x1 subplots
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")

    # plt.plot(to_db(std_total[:, 0, 0]), label="std_total[0, 0]")
    # plt.plot(to_db(std_noise[:, 0, 0]), label="std_noise[0, 0]")
    # plt.plot(to_db(std_nonlinear[:, 0, 0]), label="std_nonlinear[0, 0]")
    # plt.legend()
    # plt.subplot(3, 1, 2)
    # plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    # plt.plot(to_db(std_total[:, 1, 0]), label="std_total[1, 0]")
    # plt.plot(to_db(std_noise[:, 1, 0]), label="std_noise[1, 0]")
    # plt.plot(to_db(std_nonlinear[:, 1, 0]), label="std_nonlinear[1, 0]")
    # plt.legend()
    # plt.subplot(3, 1, 3)
    # plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    # plt.plot(to_db(std_total[:, 2, 0]), label="std_total[2, 0]")
    # plt.plot(to_db(std_noise[:, 2, 0]), label="std_noise[2, 0]")
    # plt.plot(to_db(std_nonlinear[:, 2, 0]), label="std_nonlinear[2, 0]")
    # plt.legend()
    # plt.show()


    data = load_fine_steering_mirror()["train 300mV"]
    bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    G = bla.G.value
    std_total, std_noise, std_nonlinear = bla.G.total.std, bla.G.noise.std, bla.G.nonlinear.std

    # create 3x3 subplots
    plt.figure()
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1)
        plt.plot(to_db(G[:, i, 0]), label=f"G[{i}, 0]")
        plt.plot(to_db(std_total[:, i, 0]), label=f"std_total[{i}, 0]")
        plt.plot(to_db(std_noise[:, i, 0]), label=f"std_noise[{i}, 0]")
        plt.plot(to_db(std_nonlinear[:, i, 0]), label=f"std_nonlinear[{i}, 0]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 2)
        plt.plot(to_db(G[:, i, 1]), label=f"G[{i}, 1]")
        plt.plot(to_db(std_total[:, i, 1]), label=f"std_total[{i}, 1]")
        plt.plot(to_db(std_noise[:, i, 1]), label=f"std_noise[{i}, 1]")
        plt.plot(to_db(std_nonlinear[:, i, 1]), label=f"std_nonlinear[{i}, 1]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 3)
        plt.plot(to_db(G[:, i, 2]), label=f"G[{i}, 2]")
        plt.plot(to_db(std_total[:, i, 2]), label=f"std_total[{i}, 2]")
        plt.plot(to_db(std_noise[:, i, 2]), label=f"std_noise[{i}, 2]")
        plt.plot(to_db(std_nonlinear[:, i, 2]), label=f"std_nonlinear[{i}, 2]")
        plt.legend()
    plt.show()

    # u, y = np.load("src/best_linear_approximation/robust/input_data.npy"), np.load("src/best_linear_approximation/robust/output_data.npy")
    # # # Process data
    # # u_mean = np.mean(u, axis=(0, 3), keepdims=True)  # every realisation has a different mean
    # # y_mean = np.mean(y, axis=(0, 3), keepdims=True)  # every realisation has a different mean
    # # u = u - u_mean
    # # y = y - y_mean

    # # R = 12  # should be an integer multiple of nu
    # # u_train = u[:, :, :R, :]
    # # y_train = y[:, :, :R, :]

    # bla = noisy_input(u, y, 6400)
    # G = bla.G.value
    # std_total, std_noise, std_nonlinear = bla.G.total.std, bla.G.noise.std, bla.G.nonlinear.std

    # def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    #     return 20 * np.log10(np.abs(magnitude))


    # import matplotlib.pyplot as plt
    # # create 3x3 subplots
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(3, 3, i * 3 + 1)
    #     plt.plot(to_db(G[:, i, 0]), label=f"G[{i}, 0]")
    #     plt.plot(to_db(std_total[:, i, 0]), label=f"std_total[{i}, 0]")
    #     plt.plot(to_db(std_noise[:, i, 0]), label=f"std_noise[{i}, 0]")
    #     plt.plot(to_db(std_nonlinear[:, i, 0]), label=f"std_nonlinear[{i}, 0]")
    #     plt.legend()
    #     plt.subplot(3, 3, i * 3 + 2)
    #     plt.plot(to_db(G[:, i, 1]), label=f"G[{i}, 1]")
    #     plt.plot(to_db(std_total[:, i, 1]), label=f"std_total[{i}, 1]")
    #     plt.plot(to_db(std_noise[:, i, 1]), label=f"std_noise[{i}, 1]")
    #     plt.plot(to_db(std_nonlinear[:, i, 1]), label=f"std_nonlinear[{i}, 1]")
    #     plt.legend()
    #     plt.subplot(3, 3, i * 3 + 3)
    #     plt.plot(to_db(G[:, i, 2]), label=f"G[{i}, 2]")
    #     plt.plot(to_db(std_total[:, i, 2]), label=f"std_total[{i}, 2]")
    #     plt.plot(to_db(std_noise[:, i, 2]), label=f"std_noise[{i}, 2]")
    #     plt.plot(to_db(std_nonlinear[:, i, 2]), label=f"std_nonlinear[{i}, 2]")
    #     plt.legend()
    # plt.show()


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     from best_linear_approximation._dataloader import load_f16


#     def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
#         return 20 * np.log10(np.abs(magnitude))


#     data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
#     r = data.r.mean(axis=-1)
#     bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
#     G, cov_total, cov_noise = bla.G.value, bla.G.total.cov, bla.G.noise.cov

#     print(data.u.shape)


#     # create 3x1 subplots
#     plt.figure()
#     plt.subplot(3, 1, 1)
#     plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
#     plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
#     plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
#     plt.legend()
#     plt.subplot(3, 1, 2)
#     plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
#     plt.plot(to_db(np.sqrt(cov_total[:, 1, 1])), label="cov_total[1, 0]")
#     plt.plot(to_db(np.sqrt(cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
#     plt.legend()
#     plt.subplot(3, 1, 3)
#     plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
#     plt.plot(to_db(np.sqrt(cov_total[:, 2, 2])), label="cov_total[2, 0]")
#     plt.plot(to_db(np.sqrt(cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
#     plt.legend()
#     plt.show()

#     data = load_f16()["F16Data_FullMSine_Level3.mat"]
#     r = data.r.mean(axis=-1)
#     bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
#     G, cov_total, cov_noise = bla.G.value, bla.G.total.cov, bla.G.noise.cov

#     # create 3x1 subplots
#     plt.figure()
#     plt.subplot(3, 1, 1)
#     plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")

#     plt.plot(to_db(np.sqrt(8*cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
#     plt.legend()
#     plt.subplot(3, 1, 2)
#     plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
#     plt.plot(to_db(np.sqrt(8*cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
#     plt.legend()
#     plt.subplot(3, 1, 3)
#     plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
#     plt.plot(to_db(np.sqrt(8*cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
#     plt.legend()
#     plt.show()

#         # create 3x1 subplots
#     print(data.y.shape)
#     plt.figure()
#     plt.plot(np.squeeze(data.y[:, 0, 0, :]), label="G[0, 0]")
#     plt.show()
