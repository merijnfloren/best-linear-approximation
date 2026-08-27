from typing import Any

import numpy as np
from numpy.typing import NDArray


def right_solve(left: NDArray[Any], right: NDArray[Any]) -> NDArray[Any]:
    """Solve a right-sided linear system over the final two axes.

    Computes ``solution`` such that ``solution @ right == left``. Leading axes are
    broadcast according to NumPy's rules. ``left`` has shape ``(..., n_rows, n_cols)``
    and square ``right`` has shape ``(..., n_cols, n_cols)``. If ``right`` has matrix
    shape ``(1, 1)``, scalar division is used instead of a matrix solve.
    """
    if right.shape[-2:] == (1, 1):
        return left / right

    return np.linalg.solve(right.mT, left.mT).mT


def vec(array: NDArray[Any]) -> NDArray[Any]:
    """Vectorize the final two matrix axes of an array.

    Stacks each matrix column by column while preserving leading axes. An input of shape
    ``(..., n_rows, n_cols)`` produces an array of shape ``(..., n_rows * n_cols)``.
    """
    return array.reshape(*array.shape[:-2], -1, order="F")
