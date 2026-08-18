from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._exceptions import InvalidSignalRanksError


# def to_experiment_layout(array: NDArray[Any], nu: int) -> NDArray[Any]:
#     """Split the realization axis of ``array`` into input and experiment axes.

#     Replaces axis 2, whose size is ``n_realizations``, with two axes of
#     sizes ``nu`` and ``n_realizations // nu``, respectively. The new axes
#     are inserted at the original position of axis 2, increasing the array's
#     rank by one. Realizations are assigned using column-major ordering.

#     If ``n_realizations`` is not divisible by ``nu``, the last
#     ``n_realizations % nu`` realizations are discarded.
#     """
#     n_realizations = array.shape[2]
#     n_experiments = n_realizations // nu

#     n_effective_realizations = n_experiments * nu
#     array = array[:, :, :n_effective_realizations, ...]
#     return  array.reshape(*array.shape[:2], nu, n_experiments, *array.shape[3:], order="F")


# def move_matrix_axes_to_end(array: NDArray[Any]) -> NDArray[Any]:
#     """Transform an ``(n, n_rows, n_cols, ...)`` array into ``(n, ..., n_rows, n_cols)``."""
#     minimum_rank = 3
#     if array.ndim < minimum_rank:
#         msg = f"Expected an array with at least {minimum_rank} dimensions, got {array.ndim}."
#         raise InvalidSignalRanksError(msg)

#     return np.moveaxis(array, (1, 2), (-2, -1))
