from typing import Any

import numpy as np
from numpy.typing import NDArray


def rightsolve(
    left: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
    right: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
) -> NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]]:
    """Solve a right-sided linear system over the final two axes.

    Computes ``solution`` such that ``solution @ right == left``. Leading axes are
    broadcast according to NumPy's rules. ``left`` has shape
    ``(..., n_rows, n_cols)`` and square ``right`` has shape
    ``(..., n_cols, n_cols)``. If ``right`` has matrix shape ``(1, 1)``, scalar
    division is used instead of a matrix solve.
    """
    if right.shape[-2:] == (1, 1):
        return np.divide(left, right)

    transposed_right = np.swapaxes(right, -1, -2)
    transposed_left = np.swapaxes(left, -1, -2)
    transposed_solution = np.linalg.solve(transposed_right, transposed_left)

    return np.swapaxes(transposed_solution, -1, -2)
