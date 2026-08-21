from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._exceptions import InvalidSignalAxesError, InvalidSignalRanksError
from best_linear_approximation._typing import TimeDomainSignal

ContractType = Literal["realization", "experiment"]
SignalName = Literal["r", "u", "y"]


CANONICAL_SIGNAL_NDIM = 5


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


def check_for_zero_sized_axes(arrays: Iterable[TimeDomainSignal]) -> None:
    """Raise a ``ValueError`` if any array has a zero-sized axis."""
    for array in arrays:
        if any(axis == 0 for axis in array.shape):
            msg = f"Array with shape {array.shape} has a zero-sized axis."
            raise ValueError(msg)


# def validate_estimation_requirements(
#     u: TimeDomainSignal, n_periods: int, contract_type: ContractType,
# ) -> None:
#     """Validate that the data supports the requested frequency-response estimates.

#     Raises a ValueError when a frequency response cannot be estimated. Warns when
#     there are too few experiments or periods to estimate uncertainty terms.
#     """
#     if contract_type == "realization":
#         nu, n_realizations = u.shape[1], u.shape[2]
#         n_experiments = n_realizations // nu

#         if n_experiments < 1:
#             msg = (
#                 f"There must be at least {nu} independent realizations to estimate the "
#                 f"frequency response, but only {n_realizations} realizations were provided."
#             )
#             raise InsufficientExperimentsError(msg)

#         if n_experiments == 1:
#             msg = (
#                 "Only a single experiment (n_experiment = n_realizations // nu == 1) is "
#                 "provided, so the total covariance (noise plus nonlinear distortions) "
#                 "cannot be estimated."
#             )
#             warnings.warn(msg, TotalCovarianceUnavailableWarning, stacklevel=2)

#     else:
#         if u.shape[1] != u.shape[2]:
#             msg = f"u must have the same size along axes 1 and 2, got u.shape={u.shape}."
#             raise NonSquareExperimentError(msg)

#         n_experiments = u.shape[3]
#         if n_experiments == 1:
#             msg = (
#                 "Only a single experiment is provided, so the total covariance "
#                 "(noise plus nonlinear distortions) cannot be estimated."
#             )
#             warnings.warn(msg, TotalCovarianceUnavailableWarning, stacklevel=2)

#     if n_periods == 1:
#         msg = "Only a single period is provided, so the noise covariance cannot be estimated."
#         warnings.warn(msg, NoiseCovarianceUnavailableWarning, stacklevel=2)


def validate_signal_contract(
    r: NDArray[np.floating[Any]] | None,
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    contract_types: Mapping[ContractType, SignalContract],
) -> ContractType:
    """Validate that the signals have no zero-sized axes and conform to a supported contract."""
    # Create a mapping of signal names to their corresponding arrays
    arrays_by_signal: Mapping[SignalName, NDArray[np.floating[Any]]] = {
        "u": TimeDomainSignal(u),
        "y": TimeDomainSignal(y),
    }
    if r is not None:
        arrays_by_signal["r"] = TimeDomainSignal(r)

    check_for_zero_sized_axes(arrays_by_signal.values())

    # Determine the contract type based on the ranks of the arrays
    ranks = SignalRanks(r=r.ndim if r is not None else None, u=u.ndim, y=y.ndim)
    if ranks == contract_types["realization"].ranks:
        contract_type = "realization"
    elif ranks == contract_types["experiment"].ranks:
        contract_type = "experiment"
    else:
        msg = (
            f"Invalid input dimensions. Expected {contract_types['realization'].ranks} "
            f"or {contract_types['experiment'].ranks}, got {ranks}."
        )
        raise InvalidSignalRanksError(msg)

    _validate_matching_axes(arrays_by_signal, contract_types[contract_type])

    return contract_type


def _validate_matching_axes(
    arrays_by_signal: Mapping[SignalName, TimeDomainSignal],
    contract: SignalContract,
) -> None:
    for requirement in contract.matching_axes:
        arrays = [arrays_by_signal[name] for name in requirement.signals]
        if not _axes_match(arrays, requirement.axes):
            shapes_by_signal = {
                name: array.shape
                for name, array in zip(requirement.signals, arrays, strict=True)
            }
            msg = (
                f"Signals {requirement.signals} must have equal sizes "
                f"along axes {requirement.axes}, got shapes {shapes_by_signal}."
            )
            raise InvalidSignalAxesError(msg)


def _axes_match(arrays: Sequence[TimeDomainSignal], axis: int | tuple[int, ...]) -> bool:
    """Check whether all arrays have equal sizes along the selected axis or axes.

    Arrays may have different numbers of dimensions. Each selected axis is
    applied to every array independently, following NumPy's axis-indexing
    rules; negative axes are therefore relative to each array's own rank.

    An empty ``arrays`` list or ``axes`` tuple produces ``True``.
    """
    selected_axes = axis if isinstance(axis, tuple) else (axis,)
    return all(len({array.shape[axis] for array in arrays}) <= 1 for axis in selected_axes)
