from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import (
    prepare_arguments,
    warn_for_possible_transients,
)
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._dataloader import load_f16
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
)

INDIRECT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    "realization": SignalContract(
        ranks=SignalRanks(r=3, u=4, y=4),
        matching_axes=(
            MatchingAxes(signals=("r", "u"), axes=(0, 1, 2)),
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
    "experiment": SignalContract(
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
) -> None:
    
    r, u, y, fs, excited_bins = prepare_arguments(  # pyright: ignore[reportAssignmentType]
        r, u, y, fs, excited_bins, INDIRECT_CONTRACTS
    )
    
    n_samples, ny, nu, n_experiments, n_periods = y.shape
    
    if n_periods > 1:
        warn_for_possible_transients(y, max_bin=excited_bins[-1])
        


def closed_loop(
    r: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> None:
    return known_reference(r, u, y, fs, excited_bins)




if __name__ == "__main__":

    data = load_f16()["F16Data_FullMSine_Level7.mat"]
    known_reference(np.mean(data.r, axis=-1), data.u, data.y, data.fs)

    data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    closed_loop(np.mean(data.r, axis=-1), data.u, data.y, data.fs)
