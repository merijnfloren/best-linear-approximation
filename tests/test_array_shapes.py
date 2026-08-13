import warnings
from collections.abc import Mapping

import numpy as np
import pytest

from best_linear_approximation._array_shapes import (
    ContractType,
    SignalContract,
    _axes_match,
    check_zero_sized_axes,
    validate_estimation_requirements,
    validate_signal_contract,
)
from best_linear_approximation._exceptions import (
    InsufficientExperimentsError,
    InvalidSignalAxesError,
    InvalidSignalRanksError,
    NoiseCovarianceUnavailableWarning,
    NonSquareExperimentError,
    RealizationsTruncatedWarning,
    TotalCovarianceUnavailableWarning,
    ZeroSizedAxisError,
)
from best_linear_approximation.robust._direct_methods import (
    KNOWN_INPUT_CONTRACTS,
    NOISY_INPUT_CONTRACTS,
)
from best_linear_approximation.robust._indirect_methods import INDIRECT_CONTRACTS


def test_check_zero_sized_axes_accepts_nonempty_arrays() -> None:
    check_zero_sized_axes([np.empty((2, 3)), np.empty((4, 5, 6))])


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_check_zero_sized_axes_rejects_zero_sized_axis(shape: tuple[int, ...]) -> None:
    with pytest.raises(ZeroSizedAxisError):
        check_zero_sized_axes([np.empty(shape)])


@pytest.mark.parametrize(
    ("u_shape", "n_periods", "contract_type", "error_type"),
    [
        ((5, 3, 2, 2), 2, "realization", InsufficientExperimentsError),
        ((5, 3, 2, 2, 2), 2, "experiment", NonSquareExperimentError),
    ],
)
def test_validate_estimation_requirements_rejects_invalid_structure(
    u_shape: tuple[int, ...],
    n_periods: int,
    contract_type: ContractType,
    error_type: type[ValueError],
) -> None:
    with pytest.raises(error_type):
        validate_estimation_requirements(np.empty(u_shape), n_periods, contract_type)


@pytest.mark.parametrize(
    ("u_shape", "n_periods", "contract_type", "warning_type"),
    [
        ((5, 3, 7, 2), 2, "realization", RealizationsTruncatedWarning),
        ((5, 3, 3, 2), 2, "realization", TotalCovarianceUnavailableWarning),
        ((5, 3, 3, 1, 2), 2, "experiment", TotalCovarianceUnavailableWarning),
        ((5, 3, 6, 1), 1, "realization", NoiseCovarianceUnavailableWarning),
    ],
)
def test_validate_estimation_requirements_warns_about_unavailable_estimates(
    u_shape: tuple[int, ...],
    n_periods: int,
    contract_type: ContractType,
    warning_type: type[Warning],
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_estimation_requirements(np.empty(u_shape), n_periods, contract_type)

    assert [warning.category for warning in caught] == [warning_type]


def test_validate_estimation_requirements_accepts_sufficient_data_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_estimation_requirements(
            np.empty((5, 3, 6, 2)),
            n_periods=2,
            contract_type="realization",
        )


@pytest.mark.parametrize(
    ("direct_contracts", "u_shape", "y_shape", "expected"),
    [
        (KNOWN_INPUT_CONTRACTS, (5, 2, 7), (5, 3, 7, 4), "realization"),
        (KNOWN_INPUT_CONTRACTS, (5, 2, 2, 3), (5, 3, 2, 3, 4), "experiment"),
        (NOISY_INPUT_CONTRACTS, (5, 2, 7, 4), (5, 3, 7, 4), "realization"),
        (NOISY_INPUT_CONTRACTS, (5, 2, 2, 3, 4), (5, 3, 2, 3, 4), "experiment"),
    ],
)
def test_validate_signal_contract_accepts_direct_contracts(
    direct_contracts: Mapping[ContractType, SignalContract],
    u_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    expected: str,
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
        ((5, 2, 7), (5, 2, 7, 4), (5, 3, 7, 4), "realization"),
        ((5, 2, 2, 3), (5, 2, 2, 3, 4), (5, 3, 2, 3, 4), "experiment"),
    ],
)
def test_validate_signal_contract_accepts_indirect_contracts(
    r_shape: tuple[int, ...],
    u_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    expected: str,
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


@pytest.mark.parametrize(
    ("shapes", "axes", "expected"),
    [
        ([(2, 7, 3), (2, 11, 3, 5)], (0, 2), True),
        ([(2, 7, 3), (2, 11, 4, 5)], (0, 2), False),
        ([(2, 7, 3), (2, 11, 3)], (0, -1), True),
        ([(2, 7, 3), (2, 11, 4)], (0, -1), False),
        ([(2, 7, 3), (9, 11, 5, 3)], -1, True),
        ([(2, 7, 3), (9, 11, 5, 4)], -1, False),
        ([(2, 7, 3), (9, 11, 3)], -1, True),
        ([(2, 7, 3), (9, 11, 4)], -1, False),
        ([(2, 7, 3), (9, 11, 4)], (), True),
        ([], 0, True),
        ([(2, 7, 3), (2, 4, 3), (2, 9, 3)], (-3, -1), True),
    ],
)
def test_axes_match(
    shapes: list[tuple[int, ...]],
    axes: int | tuple[int, ...],
    expected: bool,
) -> None:
    arrays = [np.empty(shape) for shape in shapes]

    assert _axes_match(arrays, axes) is expected
