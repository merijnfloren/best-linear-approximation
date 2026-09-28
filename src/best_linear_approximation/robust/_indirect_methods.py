from collections.abc import Mapping

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
from best_linear_approximation._spectra2 import (
    compute_frequency_domain_signal,
    compute_noise_covariance,
    compute_output_residual_noise_covariance_noisy_input,
    compute_output_total_covariance,
    create_input_spectrum,
    create_noiseless_input_spectrum,
    create_output_spectrum,
)
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
) -> NonparametricBLA:
    """Estimate a BLA using a known reference as an instrumental variable.

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
        Strictly increasing indices of the excited non-DC, non-Nyquist ``rfft``
        bins. A float in ``(0, 1)`` instead selects them automatically from
        the clean reference ``r``: bins whose channel-averaged spectral
        magnitude exceeds this fraction of the maximum magnitude are selected.
    independent_subexperiments : bool, default=False
        Whether subexperiments are treated as independent realizations when
        estimating total covariances.

    Returns
    -------
    NonparametricBLA
        Frequency response and available noise, nonlinear, and total covariances.

    """
    r, u, y, fs, excited_bins = _prepare_arguments_indirect(r, u, y, fs, excited_bins)
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
    experiment = ExperimentInfo.from_signals(
        EstimationMethod.ROBUST_INDIRECT_KNOWN_REFERENCE,
        u,
        y,
        r,
    )
    return NonparametricBLA(G_bla, spectra, freq, experiment)


def closed_loop(  # noqa: PLR0913
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
    *,
    independent_subexperiments: bool = False,
) -> NonparametricBLA:
    """Estimate a closed-loop BLA using a known reference.

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
        Strictly increasing indices of the excited non-DC, non-Nyquist ``rfft``
        bins. A float in ``(0, 1)`` instead selects them automatically from
        the clean reference ``r``: bins whose channel-averaged spectral
        magnitude exceeds this fraction of the maximum magnitude are selected.

    independent_subexperiments : bool, default=False
        Whether subexperiments are treated as independent realizations when
        estimating total covariances.

    Returns
    -------
    NonparametricBLA
        Frequency response and available noise, nonlinear, and total covariances.

    """
    return known_reference(
        r,
        u,
        y,
        fs,
        excited_bins,
        independent_subexperiments=independent_subexperiments,
    )


def _prepare_arguments_indirect(
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve indirect-estimation arguments."""
    return prepare_arguments(r, u, y, fs, excited_bins, INDIRECT_CONTRACTS)


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

    G_bla = create_bla_frequency_response(G, G_total_cov, G_noise_cov)
    return G_bla, Z_noise_cov


def _compute_spectra_indirect(  # noqa: PLR0913, PLR0917
    r: TimeDomainSignal,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    G_bla: ComplexArray,  # noqa: N803
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

    U_noise_cov, Y_noise_cov = compute_noise_covariance(U), compute_noise_covariance(Y)
    H_bla = _compute_reference_to_input_frequency_response(R_excited, U_excited)

    U_total_cov_per_experiment = _compute_residual_total_covariance(
        R_excited,
        U_excited,
        H_bla,
        independent_subexperiments=independent_subexperiments,
    )
    U_nonlinear_cov: ComplexArray | None = None
    U_total_cov: ComplexArray | None = None
    if U_total_cov_per_experiment is not None and U_noise_cov is not None:
        U_nonlinear_cov = project_onto_positive_semidefinite(
            U_total_cov_per_experiment - U_noise_cov[excited_bins] / n_periods,
        )
        U_total_cov = U_nonlinear_cov + U_noise_cov[excited_bins]

    Y_total_cov_equation_error = _compute_residual_total_covariance(
        U_excited,
        Y_excited,
        G_bla,
        independent_subexperiments=independent_subexperiments,
    )
    Y_noise_cov_residual = compute_output_residual_noise_covariance_noisy_input(
        Z_noise_cov,
        G_bla,
        nu,
    )

    Y_nonlinear_cov = None
    Y_total_cov = None
    if (
        Y_total_cov_equation_error is not None
        and Y_noise_cov_residual is not None
        and Y_noise_cov is not None
    ):
        Y_nonlinear_cov = project_onto_positive_semidefinite(
            Y_total_cov_equation_error - Y_noise_cov_residual / n_periods,
        )
        Y_total_cov = Y_nonlinear_cov + Y_noise_cov[excited_bins]

    reference_spectrum = create_noiseless_input_spectrum(R)
    input_spectrum = create_input_spectrum(U, U_noise_cov, U_nonlinear_cov, U_total_cov)
    output_spectrum = create_output_spectrum(
        Y,
        Y_noise_cov,
        Y_nonlinear_cov,
        Y_total_cov,
        Y_total_cov_equation_error,
    )
    return Spectra(input_spectrum, output_spectrum, reference_spectrum)


def _compute_noise_covariance(signal: FrequencyDomainSignal) -> ComplexArray | None:
    """Compute the period-averaged measurement-noise covariance."""
    n_periods = signal.shape[-1]
    if n_periods == 1:
        return None

    signal = np.moveaxis(signal, 1, -1)
    return np.mean(compute_sample_covariance(signal), axis=(1, 2))


def _compute_reference_to_input_frequency_response(
    R: FrequencyDomainSignal,  # noqa: N803
    U: FrequencyDomainSignal,  # noqa: N803
) -> ComplexArray:
    """Estimate the reference-to-input frequency response."""
    R = as_batched_matrices(R)
    U = as_batched_matrices(U)
    R = np.squeeze(R, axis=2)
    U = np.mean(U, axis=2)
    reference_spectrum = np.mean(R @ R.conj().mT, axis=1)
    input_reference_spectrum = np.mean(U @ R.conj().mT, axis=1)
    return compute_frequency_response(reference_spectrum, input_reference_spectrum)


def _compute_residual_total_covariance(
    predictor: FrequencyDomainSignal,
    response: FrequencyDomainSignal,
    frequency_response: ComplexArray,
    *,
    independent_subexperiments: bool,
) -> ComplexArray | None:
    """Estimate covariance of period-averaged residuals."""
    predictor = as_batched_matrices(predictor)
    response = as_batched_matrices(response)
    n_excited_bins, n_experiments, _, ny, nu = response.shape
    n_realizations = n_experiments * nu if independent_subexperiments else n_experiments
    if n_realizations == 1:
        return None

    residual = np.mean(response - frequency_response[:, None, None] @ predictor, axis=2)
    if independent_subexperiments:
        residual = residual.transpose(0, 1, 3, 2).reshape(n_excited_bins, n_realizations, ny)
        return compute_sample_covariance(residual)

    residual = residual.transpose(0, 3, 1, 2)
    return np.mean(compute_sample_covariance(residual), axis=1)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from best_linear_approximation._dataloader import load_f16
    from best_linear_approximation.robust._direct_methods import noisy_input

    def plot_uncertainties(
        axis: Axes,
        estimate: FrequencyResponse | InputSpectrum | OutputSpectrum,
        label: str,
        indices: tuple[int, ...],
        linestyle: str,
    ) -> None:
        """Plot available marginal standard deviations for one spectral component."""
        for uncertainty_name in ("noise", "nonlinear", "total"):
            standard_deviation = getattr(estimate, uncertainty_name).std
            if standard_deviation is None:
                continue

            uncertainty_freqs = (
                freqs if standard_deviation.shape[0] == freqs.size else excited_freqs
            )
            component = standard_deviation[(slice(None), *indices)]
            if (
                show_excited_input_bins_only
                and isinstance(estimate, InputSpectrum)
                and standard_deviation.shape[0] == freqs.size
            ):
                component = component[excited_bins]
                uncertainty_freqs = excited_freqs
            axis.plot(
                uncertainty_freqs,
                20 * np.log10(component),
                linestyle=linestyle,
                label=f"{label} {uncertainty_name} std",
            )

    f16_data = load_f16()
    dataset_names = (
        "F16Data_SpecialOddMSine_Level3.mat",
        "F16Data_FullMSine_Level3.mat",
    )
    for dataset_name in dataset_names:
        data = f16_data[dataset_name]
        r = np.mean(data.r, axis=-1)
        bla_reference = known_reference(r, data.u, data.y, data.fs, data.excited_bins)
        bla_noisy = noisy_input(data.u, data.y, data.fs, data.excited_bins)
        blas = {"known reference": bla_reference, "noisy input": bla_noisy}
        freqs = bla_reference.freq.freqs
        excited_bins = bla_reference.freq.excited_bins
        excited_freqs = freqs[bla_reference.freq.excited_bins]
        show_excited_input_bins_only = "SpecialOddMSine" in dataset_name

        n_spectrum_rows = max(bla_reference.experiment.nu, bla_reference.experiment.ny)
        fig_spectra, spectrum_axes = plt.subplots(
            n_spectrum_rows,
            2,
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        fig_spectra.suptitle(dataset_name)
        for method, bla in blas.items():
            linestyle = "-" if method == "known reference" else "--"
            U = bla.spectra.U.value
            Y = bla.spectra.Y.value
            U_magnitude = np.mean(np.abs(U), axis=tuple(range(2, U.ndim)))
            Y_magnitude = np.mean(np.abs(Y), axis=tuple(range(2, Y.ndim)))
            for input_channel in range(bla.experiment.nu):
                axis = spectrum_axes[input_channel, 0]
                label = f"{method} U[{input_channel}]"
                input_freqs = excited_freqs if show_excited_input_bins_only else freqs
                input_magnitude = U_magnitude[:, input_channel]
                if show_excited_input_bins_only:
                    input_magnitude = input_magnitude[excited_bins]
                axis.plot(
                    input_freqs,
                    20 * np.log10(input_magnitude),
                    linestyle=linestyle,
                    label=label,
                )
                plot_uncertainties(
                    axis,
                    bla.spectra.U,
                    label,
                    (input_channel,),
                    linestyle,
                )
            for output_channel in range(bla.experiment.ny):
                axis = spectrum_axes[output_channel, 1]
                label = f"{method} Y[{output_channel}]"
                axis.plot(
                    freqs,
                    20 * np.log10(Y_magnitude[:, output_channel]),
                    linestyle=linestyle,
                    label=label,
                )
                plot_uncertainties(
                    axis,
                    bla.spectra.Y,
                    label,
                    (output_channel,),
                    linestyle,
                )
        spectrum_axes[0, 0].set_title("input spectra")
        spectrum_axes[0, 1].set_title("output spectra")
        for row in range(n_spectrum_rows):
            input_axis, output_axis = spectrum_axes[row]
            input_axis.set_ylabel(f"channel {row} [dB]")
            if input_axis.lines:
                input_axis.legend()
            if output_axis.lines:
                output_axis.legend()
        spectrum_axes[-1, 0].set_xlabel("frequency [Hz]")
        spectrum_axes[-1, 1].set_xlabel("frequency [Hz]")

        fig_frf, frf_axes = plt.subplots(
            bla_reference.experiment.ny,
            bla_reference.experiment.nu,
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        fig_frf.suptitle(dataset_name)
        for method, bla in blas.items():
            linestyle = "-" if method == "known reference" else "--"
            for output_channel in range(bla.experiment.ny):
                for input_channel in range(bla.experiment.nu):
                    axis = frf_axes[output_channel, input_channel]
                    label = f"{method} G[{output_channel}, {input_channel}]"
                    axis.plot(
                        excited_freqs,
                        20 * np.log10(np.abs(bla.G.value[:, output_channel, input_channel])),
                        linestyle=linestyle,
                        label=label,
                    )
                    plot_uncertainties(
                        axis,
                        bla.G,
                        label,
                        (output_channel, input_channel),
                        linestyle,
                    )
        for output_channel in range(bla_reference.experiment.ny):
            for input_channel in range(bla_reference.experiment.nu):
                axis = frf_axes[output_channel, input_channel]
                axis.legend()
        for input_channel in range(bla_reference.experiment.nu):
            frf_axes[-1, input_channel].set_xlabel("frequency [Hz]")
        for output_channel in range(bla_reference.experiment.ny):
            frf_axes[output_channel, 0].set_ylabel(f"output {output_channel} [dB]")

    plt.show()
