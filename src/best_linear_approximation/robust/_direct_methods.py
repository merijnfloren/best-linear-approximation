from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._bla import BLA, FrequencyInfo, create_bla, create_frequency_info
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._covariance import (
    compute_sample_covariance,
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
    ExcitedBins,
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
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> BLA:
    """Estimate a BLA when the input is known without measurement noise.

    Parameters
    ----------
    u, y : NDArray[np.floating[Any]]
        Periodic input and output measurements.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        Excited DFT bins, or a relative detection threshold.

    Returns
    -------
    BLA
        Frequency response and available total, noise, and nonlinear covariances.

    """
    u, y, fs, excited_bins = _prepare_arguments_known_input(u, y, fs, excited_bins)
    freq = create_frequency_info(u.shape[0], fs, excited_bins)
    return _compute_robust_known_input(u, y, freq)


def noisy_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> BLA:
    """Estimate a BLA from input and output measurements with input noise.

    Parameters
    ----------
    u, y : NDArray[np.floating[Any]]
        Periodic input and output measurements.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        Excited DFT bins, or a relative detection threshold.

    Returns
    -------
    BLA
        Frequency response and available total, noise, and nonlinear covariances.

    """
    u, y, fs, excited_bins = _prepare_arguments_noisy_input(u, y, fs, excited_bins)
    freq = create_frequency_info(u.shape[0], fs, excited_bins)
    return _compute_robust_noisy_input(u, y, freq)


def _prepare_arguments_known_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve all arguments according to :func:`prepare_arguments`."""
    return prepare_arguments(None, u, y, fs, excited_bins, KNOWN_INPUT_CONTRACTS)[1:]


def _prepare_arguments_noisy_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve all arguments according to :func:`prepare_arguments`."""
    return prepare_arguments(None, u, y, fs, excited_bins, NOISY_INPUT_CONTRACTS)[1:]


def _compute_robust_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    freq: FrequencyInfo,
) -> BLA:
    """Compute the best linear approximation and its covariances from known input data."""
    n_experiments, n_periods = y.shape[-2:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_batched_matrices = as_batched_matrices(u)  # (n_samples, n_experiments, 1, nu, nu)
    y_batched_matrices = as_batched_matrices(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U = np.fft.rfft(u_batched_matrices, axis=0)[freq.excited_bins]
    Y = np.fft.rfft(y_batched_matrices, axis=0)[freq.excited_bins]

    # Frequency response: (n_excited_bins, n_experiments, n_periods, ny, nu)
    G_per_experiment_period = compute_frequency_response(U, Y)

    # Average over periods: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = np.mean(G_per_experiment_period, axis=2)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)

    # BLA total covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(
            vec(G_per_experiment),  # (n_excited_bins, n_experiments, ny * nu)
        ) / n_experiments
    else:
        G_cov_total = None

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_periods > 1:
        G_cov_noise = np.mean(
            compute_sample_covariance(
                vec(G_per_experiment_period),  # (n_excited_bins, n_experiments, n_periods, ny * nu)
            ) / n_periods,
            axis=1,
        ) / n_experiments
    else:
        G_cov_noise = None

    return create_bla(freq, G, G_cov_total, G_cov_noise)


def _compute_robust_noisy_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    freq: FrequencyInfo,
) -> BLA:
    """Compute the best linear approximation and its covariances from noisy input data."""
    ny, nu, n_experiments, n_periods = y.shape[-4:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_batched_matrices = as_batched_matrices(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_batched_matrices = as_batched_matrices(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U = np.fft.rfft(u_batched_matrices, axis=0)[freq.excited_bins]
    Y = np.fft.rfft(y_batched_matrices, axis=0)[freq.excited_bins]

    # Data noise covariance: (n_excited_bins, n_experiments, (ny + nu) * nu, (ny + nu) * nu)
    if n_periods > 1:
        Z = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )

        cov_Z_noise = compute_sample_covariance(
            vec(Z),  # (n_excited_bins, n_experiments, n_periods, (ny + nu) * nu)
        ) / n_periods
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
        G_cov_total = compute_sample_covariance(
            vec(G_per_experiment),  # (n_excited_bins, n_experiments, ny * nu)
        ) / n_experiments
    else:
        G_cov_total = None

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if cov_Z_noise is not None:
        n_excited_bins = G.shape[0]

        # Batched U^{-T}
        U_inv_transpose = np.linalg.solve(U, np.eye(nu)).mT if nu > 1 else 1 / U

        # Batched V = [I_ny, -G]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        V = np.concatenate((I_ny, -G_per_experiment), axis=-1)

        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        jacobian = kronecker_product(U_inv_transpose, V)

        G_cov_noise = np.mean(
            propagate_covariance(cov_Z_noise, jacobian),
            axis=1,
        ) / n_experiments
    else:
        G_cov_noise = None

    return create_bla(freq, G, G_cov_total, G_cov_noise)


if __name__ == "__main__":
    from best_linear_approximation._dataloader import (
        load_f16,
        load_fine_steering_mirror,
        load_parallel_wiener_hammerstein,
        load_silverbox,
    )

    data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]

    bla = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    G = bla.G.value
    std_total, std_noise, std_nonlinear = bla.G.std_total, bla.G.std_noise, bla.G.std_nonlinear

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
    # std_total, std_noise, std_nonlinear = bla.G.std_total, bla.G.std_noise, bla.G.std_nonlinear


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
    std_total, std_noise, std_nonlinear = bla.G.std_total, bla.G.std_noise, bla.G.std_nonlinear


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
    # std_total, std_noise, std_nonlinear = bla.G.std_total, bla.G.std_noise, bla.G.std_nonlinear

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
    std_total, std_noise, std_nonlinear = bla.G.std_total, bla.G.std_noise, bla.G.std_nonlinear

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
    # std_total, std_noise, std_nonlinear = bla.G.std_total, bla.G.std_noise, bla.G.std_nonlinear

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
#     G, cov_total, cov_noise = bla.G.value, bla.G.cov_total, bla.G.cov_noise
    
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
#     G, cov_total, cov_noise = bla.G.value, bla.G.cov_total, bla.G.cov_noise

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
