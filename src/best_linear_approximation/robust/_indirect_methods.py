from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalName,
    SignalRanks,
)

INDIRECT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=3, u=4, y=4),
        matching_axes=(
            MatchingAxes(signals=(SignalName.REFERENCE, SignalName.INPUT), axes=(0, 1, 2)),
            MatchingAxes(signals=(SignalName.INPUT, SignalName.OUTPUT), axes=(0, 2, 3)),
        ),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=4, u=5, y=5),
        matching_axes=(
            MatchingAxes(
                signals=(SignalName.REFERENCE, SignalName.INPUT),
                axes=(0, 1, 2, 3),
            ),
            MatchingAxes(signals=(SignalName.INPUT, SignalName.OUTPUT), axes=(0, 2, 3, 4)),
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

    r, u, y, fs, excited_bins = prepare_arguments(r, u, y, fs, excited_bins, INDIRECT_CONTRACTS)


def closed_loop(
    r: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> None:
    return known_reference(r, u, y, fs, excited_bins)
