from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def kronecker_product(left: NDArray[Any], right: NDArray[Any]) -> NDArray[Any]:
    """Compute Kronecker products of the final two axes of two arrays.

    The leading axes of ``left`` and ``right`` must match exactly. ``left`` has
    shape ``(..., n_rows_left, n_cols_left)`` and ``right`` has shape
    ``(..., n_rows_right, n_cols_right)``. The output has shape
    ``(..., n_rows_left * n_rows_right, n_cols_left * n_cols_right)``.
    """
    if left.shape[:-2] != right.shape[:-2]:
        msg = (
            f"Leading axes of left {left.shape[:-2]} and right {right.shape[:-2]} must match, "
            f"got {left.shape} and {right.shape}."
        )
        raise ValueError(msg)

    n_rows_left, n_cols_left = left.shape[-2:]
    n_rows_right, n_cols_right = right.shape[-2:]

    kronecker_tensor = left[..., :, None, :, None] * right[..., None, :, None, :]

    return kronecker_tensor.reshape(
        *left.shape[:-2],
        n_rows_left * n_rows_right,
        n_cols_left * n_cols_right,
    )


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
