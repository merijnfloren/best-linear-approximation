from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal, NamedTuple

from best_linear_approximation._exceptions import InvalidSignalAxesError, InvalidSignalRanksError
from best_linear_approximation._typing import ComplexArray, RealArray


class ContractType(StrEnum):
    REALIZATION = "realization"
    EXPERIMENT = "experiment"


SignalName = Literal["r", "u", "y"]


class SignalRanks(NamedTuple):
    r: int | None
    u: int
    y: int


class MatchingAxes(NamedTuple):
    signals: tuple[SignalName, ...]
    axes: tuple[int, ...]


class SignalContract(NamedTuple):
    ranks: SignalRanks
    matching_axes: tuple[MatchingAxes, ...]


def check_for_zero_sized_axes(
    arrays: Iterable[RealArray | ComplexArray],
) -> None:
    """Raise a ``ValueError`` if any array has a zero-sized axis."""
    for array in arrays:
        if any(axis == 0 for axis in array.shape):
            msg = f"Array with shape {array.shape} has a zero-sized axis."
            raise ValueError(msg)


def validate_signal_contract(
    r: RealArray | None,
    u: RealArray,
    y: RealArray,
    contract_types: Mapping[ContractType, SignalContract],
) -> ContractType:
    """Validate that the signals have no zero-sized axes and conform to a supported contract."""
    # Create a mapping of signal names to their corresponding arrays
    arrays_by_signal: Mapping[SignalName, RealArray] = {"u": u, "y": y}
    if r is not None:
        arrays_by_signal["r"] = r

    check_for_zero_sized_axes(arrays_by_signal.values())

    # Determine the contract type based on the ranks of the arrays
    ranks = SignalRanks(r=r.ndim if r is not None else None, u=u.ndim, y=y.ndim)
    if ranks == contract_types[ContractType.REALIZATION].ranks:
        contract_type = ContractType.REALIZATION
    elif ranks == contract_types[ContractType.EXPERIMENT].ranks:
        contract_type = ContractType.EXPERIMENT
    else:
        msg = (
            f"Invalid input dimensions. Expected {contract_types[ContractType.REALIZATION].ranks} "
            f"or {contract_types[ContractType.EXPERIMENT].ranks}, got {ranks}."
        )
        raise InvalidSignalRanksError(msg)

    _validate_matching_axes(arrays_by_signal, contract_types[contract_type])

    return contract_type


def _validate_matching_axes(
    arrays_by_signal: Mapping[
        SignalName,
        RealArray | ComplexArray,
    ],
    contract: SignalContract,
) -> None:
    for requirement in contract.matching_axes:
        arrays = [arrays_by_signal[name] for name in requirement.signals]
        if not _axes_match(arrays, requirement.axes):
            shapes_by_signal = {
                name: array.shape for name, array in zip(requirement.signals, arrays, strict=True)
            }
            msg = (
                f"Signals {requirement.signals} must have equal sizes "
                f"along axes {requirement.axes}, got shapes {shapes_by_signal}."
            )
            raise InvalidSignalAxesError(msg)


def _axes_match(
    arrays: Sequence[RealArray | ComplexArray],
    axis: int | tuple[int, ...],
) -> bool:
    """Check whether all arrays have equal sizes along the selected axis or axes.

    Arrays may have different numbers of dimensions. Each selected axis is
    applied to every array independently, following NumPy's axis-indexing
    rules; negative axes are therefore relative to each array's own rank.

    An empty ``arrays`` list or ``axes`` tuple produces ``True``.
    """
    selected_axes = axis if isinstance(axis, tuple) else (axis,)
    return all(len({array.shape[axis] for array in arrays}) <= 1 for axis in selected_axes)
