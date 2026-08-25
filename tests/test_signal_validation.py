from collections.abc import Mapping

import numpy as np
import pytest

from best_linear_approximation._exceptions import InvalidSignalAxesError, InvalidSignalRanksError
from best_linear_approximation._signal_validation import (
    ContractType,
    SignalContract,
    validate_signal_contract,
)
from best_linear_approximation.robust._direct_methods import (
    KNOWN_INPUT_CONTRACTS,
    NOISY_INPUT_CONTRACTS,
)
from best_linear_approximation.robust._indirect_methods import INDIRECT_CONTRACTS


@pytest.mark.parametrize(
    ("direct_contracts", "u_shape", "y_shape", "expected"),
    [
        (KNOWN_INPUT_CONTRACTS, (5, 2, 7), (5, 3, 7, 4), ContractType.REALIZATION),
        (KNOWN_INPUT_CONTRACTS, (5, 2, 2, 3), (5, 3, 2, 3, 4), ContractType.EXPERIMENT),
        (NOISY_INPUT_CONTRACTS, (5, 2, 7, 4), (5, 3, 7, 4), ContractType.REALIZATION),
        (NOISY_INPUT_CONTRACTS, (5, 2, 2, 3, 4), (5, 3, 2, 3, 4), ContractType.EXPERIMENT),
    ],
)
def test_validate_signal_contract_accepts_direct_contracts(
    direct_contracts: Mapping[ContractType, SignalContract],
    u_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    expected: ContractType,
) -> None:
    assert (
        validate_signal_contract(
            None,
            np.empty(u_shape),
            np.empty(y_shape),
            direct_contracts,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("r_shape", "u_shape", "y_shape", "expected"),
    [
        ((5, 2, 7), (5, 2, 7, 4), (5, 3, 7, 4), ContractType.REALIZATION),
        ((5, 2, 2, 3), (5, 2, 2, 3, 4), (5, 3, 2, 3, 4), ContractType.EXPERIMENT),
    ],
)
def test_validate_signal_contract_accepts_indirect_contracts(
    r_shape: tuple[int, ...],
    u_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    expected: ContractType,
) -> None:
    assert (
        validate_signal_contract(
            np.empty(r_shape),
            np.empty(u_shape),
            np.empty(y_shape),
            INDIRECT_CONTRACTS,
        )
        == expected
    )


def test_validate_signal_contract_rejects_unsupported_ranks() -> None:
    with pytest.raises(InvalidSignalRanksError):
        validate_signal_contract(
            None,
            np.empty((5, 2)),
            np.empty((5, 3, 7, 4)),
            KNOWN_INPUT_CONTRACTS,
        )


def test_validate_signal_contract_rejects_mismatched_direct_axes() -> None:
    with pytest.raises(InvalidSignalAxesError):
        validate_signal_contract(
            None,
            np.empty((5, 2, 7, 4)),
            np.empty((6, 3, 7, 4)),
            NOISY_INPUT_CONTRACTS,
        )


def test_validate_signal_contract_rejects_mismatched_indirect_reference_axes() -> None:
    with pytest.raises(InvalidSignalAxesError):
        validate_signal_contract(
            np.empty((5, 3, 7)),
            np.empty((5, 2, 7, 4)),
            np.empty((5, 3, 7, 4)),
            INDIRECT_CONTRACTS,
        )
