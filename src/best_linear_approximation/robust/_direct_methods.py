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

KNOWN_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    "realization": SignalContract(
        ranks=SignalRanks(r=None, u=3, y=4),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2)),
        ),
    ),
    "experiment": SignalContract(
        ranks=SignalRanks(r=None, u=4, y=5),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
}


NOISY_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    "realization": SignalContract(
        ranks=SignalRanks(r=None, u=4, y=4),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
    "experiment": SignalContract(
        ranks=SignalRanks(r=None, u=5, y=5),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3, 4)),
        ),
    ),
}


def _validate_arguments(
    u: NDArray[Any],
    y: NDArray[Any],
    direct_contracts: Mapping[ContractType, SignalContract],
) -> ContractType:
    check_zero_sized_axes([u, y])

    contract_type = validate_signal_contract(None, u, y, direct_contracts)

    validate_estimation_requirements(u, n_periods=y.shape[-1], contract_type=contract_type)

    return contract_type


def _validate_arguments_known_input(u: NDArray[Any], y: NDArray[Any]) -> ContractType:
    return _validate_arguments(u, y, KNOWN_INPUT_CONTRACTS)


def _validate_arguments_noisy_input(u: NDArray[Any], y: NDArray[Any]) -> ContractType:
    return _validate_arguments(u, y, NOISY_INPUT_CONTRACTS)


def noisy_input(*, u: NDArray[Any], y: NDArray[Any]) -> ContractType:
    return _validate_arguments_noisy_input(u, y)


def known_input(*, u: NDArray[Any], y: NDArray[Any]) -> ContractType:
    return _validate_arguments_known_input(u, y)


if __name__ == "__main__":
    # Example usage
    u = np.empty((10, 3, 9, 2))
    y = np.empty((10, 9, 9, 2))

    contract_type = noisy_input(u=u, y=y)
    print(f"Contract type for noisy input: {contract_type}")

    u = np.empty((10, 3, 9))
    y = np.empty((10, 19, 9, 2))
    contract_type = known_input(u=u, y=y)
    print(f"Contract type for known input: {contract_type}")
