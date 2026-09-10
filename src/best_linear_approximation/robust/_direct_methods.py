from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._best_linear_approximation import (
    compute_best_linear_approximation_known_input,
    compute_best_linear_approximation_noisy_input,
)
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
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
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    u, y, fs, excited_bins = _prepare_arguments_known_input(u, y, fs, excited_bins)

    return compute_best_linear_approximation_known_input(u, y, excited_bins)


def noisy_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    u, y, fs, excited_bins = _prepare_arguments_noisy_input(u, y, fs, excited_bins)

    return compute_best_linear_approximation_noisy_input(u, y, excited_bins)


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


if __name__ == "__main__":
    from best_linear_approximation._dataloader import (
        load_f16,
        load_fine_steering_mirror,
        load_parallel_wiener_hammerstein,
        load_silverbox,
    )

    data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]

    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)

    import matplotlib.pyplot as plt


    def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return 20 * np.log10(np.abs(magnitude))

    plt.figure()
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.show()

    data = load_silverbox()["train SB multisine"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs)


    plt.figure()
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.legend()
    plt.show()

    data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)


    # create 3x1 subplots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 1, 1])), label="cov_total[1, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 2, 2])), label="cov_total[2, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
    plt.legend()
    plt.show()

    data = load_f16()["F16Data_FullMSine_Level3.mat"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)

    # create 3x1 subplots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")

    plt.plot(to_db(np.sqrt(8*cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    plt.plot(to_db(np.sqrt(8*cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    plt.plot(to_db(np.sqrt(8*cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
    plt.legend()
    plt.show()


    data = load_fine_steering_mirror()["train 300mV"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)

    # create 3x3 subplots
    plt.figure()
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1)
        plt.plot(to_db(G[:, i, 0]), label=f"G[{i}, 0]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 0]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 0]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 2)
        plt.plot(to_db(G[:, i, 1]), label=f"G[{i}, 1]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 1]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 1]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 3)
        plt.plot(to_db(G[:, i, 2]), label=f"G[{i}, 2]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 2]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 2]")
        plt.legend()
    plt.show()

    u, y = np.load("src/best_linear_approximation/robust/input_data.npy"), np.load("src/best_linear_approximation/robust/output_data.npy")
    # # Process data
    # u_mean = np.mean(u, axis=(0, 3), keepdims=True)  # every realisation has a different mean
    # y_mean = np.mean(y, axis=(0, 3), keepdims=True)  # every realisation has a different mean
    # u = u - u_mean
    # y = y - y_mean

    # R = 12  # should be an integer multiple of nu
    # u_train = u[:, :, :R, :]
    # y_train = y[:, :, :R, :]

    G, cov_total, cov_noise = noisy_input(u, y, 6400)

    def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return 20 * np.log10(np.abs(magnitude))


    import matplotlib.pyplot as plt
    # create 3x3 subplots
    plt.figure()
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1)
        plt.plot(to_db(G[:, i, 0]), label=f"G[{i}, 0]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 0]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 0]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 2)
        plt.plot(to_db(G[:, i, 1]), label=f"G[{i}, 1]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 1]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 1]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 3)
        plt.plot(to_db(G[:, i, 2]), label=f"G[{i}, 2]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 2]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 2]")
        plt.legend()
    plt.show()
