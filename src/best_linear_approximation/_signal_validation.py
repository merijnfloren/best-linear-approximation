import warnings
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

from numpy.typing import NDArray

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

ContractType = Literal["realization", "experiment"]
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


def check_zero_sized_axes(arrays: list[NDArray[Any]]) -> None:
    """Check for zero-sized axes in the provided arrays.

    Raises a ValueError if any array has a zero-sized axis.
    """
    for array in arrays:
        if any(axis == 0 for axis in array.shape):
            msg = f"Array with shape {array.shape} has a zero-sized axis."
            raise ZeroSizedAxisError(msg)


def validate_estimation_requirements(
    u: NDArray[Any], n_periods: int, contract_type: ContractType,
) -> None:
    """Validate that the data supports the requested frequency-response estimates.

    Raises a ValueError when a frequency response cannot be estimated. Warns when
    there are too few experiments or periods to estimate uncertainty terms.
    """
    if contract_type == "realization":
        nu, n_realizations = u.shape[1], u.shape[2]
        n_experiments = n_realizations // nu

        if n_experiments < 1:
            msg = (
                f"There must be at least {nu} independent realizations to estimate the "
                f"frequency response, but only {n_realizations} realizations were provided."
            )
            raise InsufficientExperimentsError(msg)

        n_effective_realizations = n_experiments * nu
        if n_effective_realizations != n_realizations:
            msg = (
                f"The number of realizations ({n_realizations}) is not a multiple of "
                f"the number of input channels ({nu}). Only the first "
                f"{n_effective_realizations} realizations will be used for estimation."
            )
            warnings.warn(msg, RealizationsTruncatedWarning, stacklevel=2)

        if n_experiments == 1:
            msg = (
                "Only a single experiment (n_experiment = n_realizations // nu == 1) is "
                "provided, so the total covariance (noise plus nonlinear distortions) "
                "cannot be estimated."
            )
            warnings.warn(msg, TotalCovarianceUnavailableWarning, stacklevel=2)

    else:
        if u.shape[1] != u.shape[2]:
            msg = f"u must have the same size along axes 1 and 2, got u.shape={u.shape}."
            raise NonSquareExperimentError(msg)

        n_experiments = u.shape[3]
        if n_experiments == 1:
            msg = (
                "Only a single experiment is provided, so the total covariance "
                "(noise plus nonlinear distortions) cannot be estimated."
            )
            warnings.warn(msg, TotalCovarianceUnavailableWarning, stacklevel=2)

    if n_periods == 1:
        msg = "Only a single period is provided, so the noise covariance cannot be estimated."
        warnings.warn(msg, NoiseCovarianceUnavailableWarning, stacklevel=2)


def validate_signal_contract(
    r: NDArray[Any] | None,
    u: NDArray[Any],
    y: NDArray[Any],
    contract_types: Mapping[ContractType, SignalContract],
) -> ContractType:

    # Create a mapping of signal names to their corresponding arrays
    arrays_by_signal: Mapping[SignalName, NDArray[Any]] = {"u": u, "y": y}
    if r is not None:
        arrays_by_signal["r"] = r

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
    arrays_by_signal: Mapping[SignalName, NDArray[Any]],
    contract: SignalContract,
) -> None:
    for requirement in contract.matching_axes:
        arrays = [arrays_by_signal[name] for name in requirement.signals]
        if not _axes_match(arrays, requirement.axes):
            shapes_by_signal = {
                name: array.shape for name, array in zip(requirement.signals, arrays)
            }
            msg = (
                f"Signals {requirement.signals} must have equal sizes "
                f"along axes {requirement.axes}, got shapes {shapes_by_signal}."
            )
            raise InvalidSignalAxesError(msg)


def _axes_match(arrays: list[NDArray[Any]], axes: int | tuple[int, ...]) -> bool:
    """Check whether all arrays have equal sizes along the selected axes.

    Arrays may have different numbers of dimensions. Each selected axis is
    applied to every array independently, following NumPy's axis-indexing
    rules; negative axes are therefore relative to each array's own rank.

    An empty ``arrays`` list or ``axes`` tuple produces ``True``.
    """
    selected_axes = axes if isinstance(axes, tuple) else (axes,)
    return all(len({array.shape[axis] for array in arrays}) <= 1 for axis in selected_axes)
