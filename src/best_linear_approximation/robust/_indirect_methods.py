from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._array_shapes import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
    check_zero_sized_axes,
    validate_estimation_requirements,
    validate_signal_contract,
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


def _validate_arguments(
    r: NDArray[Any],
    u: NDArray[Any],
    y: NDArray[Any],
) -> ContractType:
    check_zero_sized_axes([r, u, y])

    contract_type = validate_signal_contract(r, u, y, INDIRECT_CONTRACTS)

    validate_estimation_requirements(u, n_periods=y.shape[-1], contract_type=contract_type)

    return contract_type


def known_reference(*, r: NDArray[Any], u: NDArray[Any], y: NDArray[Any]) -> ContractType:
    return _validate_arguments(r, u, y)


def closed_loop(*, r: NDArray[Any], u: NDArray[Any], y: NDArray[Any]) -> ContractType:
    return known_reference(r=r, u=u, y=y)


if __name__ == "__main__":
    # Example usage
    r = np.empty((10, 3, 9))  # Example reference signal
    u = np.empty((10, 3, 9, 2))
    y = np.empty((10, 9, 9, 2))

    contract_type = known_reference(r=r, u=u, y=y)
    print(f"Contract type for known reference: {contract_type}")

    r = np.empty((10, 3, 3, 1))  # Example reference signal
    u = np.empty((10, 3, 3, 1, 1))
    y = np.empty((10, 9, 3, 1, 1))
    contract_type = closed_loop(r=r, u=u, y=y)
    print(f"Contract type for closed loop: {contract_type}")
