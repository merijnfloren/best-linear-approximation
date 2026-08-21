import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._exceptions import (
    InsufficientExperimentsError,
    RealizationsTruncatedWarning,
)

CANONICAL_SIGNAL_NDIM = 5


def to_experiment_layout(
    signal: NDArray[np.floating[Any]],
    nu: int,
) -> NDArray[np.floating[Any]]:
    """Convert a signal from realization layout to five-dimensional experiment layout.

    Splits the realization axis of a signal with shape
    ``(n_samples, n_channels, n_realizations[, n_periods])`` into input and experiment
    axes using column-major ordering. Adds a singleton period axis if none is present.
    The resulting shape is ``(n_samples, n_channels, nu, n_experiments, n_periods)``.

    Raises an ``InsufficientExperimentsError`` if ``n_realizations < nu``. Warns if
    ``n_realizations`` is not divisible by ``nu`` and discards the remaining realizations.
    """
    n_realizations = signal.shape[2]
    n_experiments = n_realizations // nu
    if n_experiments == 0:
        msg = (
            f"The number of realizations ({n_realizations}) is less than the number of "
            f"input channels ({nu}). No frequency response can be estimated."
        )
        raise InsufficientExperimentsError(msg)

    n_effective_realizations = n_experiments * nu
    if n_effective_realizations < n_realizations:
        msg = (
            f"The number of realizations ({n_realizations}) is not a multiple of "
            f"the number of input channels ({nu}). Only the first "
            f"{n_effective_realizations} realizations will be used for estimation."
        )
        warnings.warn(msg, RealizationsTruncatedWarning, stacklevel=2)

    signal = signal[:, :, :n_effective_realizations, ...]
    signal = signal.reshape(*signal.shape[:2], nu, n_experiments, *signal.shape[3:], order="F")

    return signal if signal.ndim == CANONICAL_SIGNAL_NDIM else signal[..., None]


# def to_realization_layout(
#     signal: TimeDomainSignal,
# ) -> TimeDomainSignal:
#     """Convert a time-domain signal from experiment layout to realization layout.

#     Transforms a signal of shape ``(n_points, n_channels, nu, n_experiments, ...)``
#     into one of shape ``(n_points, n_channels, n_realizations, ...)`` by merging
#     the experiment and input axes using column-major ordering.
#     """
#     if signal.ndim < 4:
#         msg = f"Expected a signal with at least 4 dimensions, got {signal.ndim}D."
#         raise InvalidSignalRanksError(msg)

#     return signal.reshape(*signal.shape[:2], -1, *signal.shape[4:], order="F")


# def move_matrix_axes_to_end(array: NDArray[Any]) -> NDArray[Any]:
#     """Transform an ``(n, n_rows, n_cols, ...)`` array into ``(n, ..., n_rows, n_cols)``."""
#     minimum_rank = 3
#     if array.ndim < minimum_rank:
#         msg = f"Expected an array with at least {minimum_rank} dimensions, got {array.ndim}."
#         raise InvalidSignalRanksError(msg)

#     return np.moveaxis(array, (1, 2), (-2, -1))
