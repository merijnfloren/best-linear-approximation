from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._best_linear_approximation import (
    compute_best_linear_approximation_indirect,
)
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
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


def known_reference(
    r: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:

    r, u, y, fs, excited_bins = prepare_arguments(r, u, y, fs, excited_bins, INDIRECT_CONTRACTS)
    return compute_best_linear_approximation_indirect(r, u, y, excited_bins)


def closed_loop(
    r: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    return known_reference(r, u, y, fs, excited_bins)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from best_linear_approximation._dataloader import load_f16


    def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return 20 * np.log10(np.abs(magnitude))


    data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    r = data.r.mean(axis=-1)
    G, cov_total, cov_noise = closed_loop(r, data.u, data.y, data.fs, data.excited_bins)


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
    r = data.r.mean(axis=-1)
    G, cov_total, cov_noise = closed_loop(r, data.u, data.y, data.fs, data.excited_bins)

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
