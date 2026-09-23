from typing import Any

import numpy as np
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


def solve_left_kronecker_product(
    left: NDArray[Any],
    right: NDArray[Any],
) -> NDArray[Any]:
    """Solve a Kronecker-product equation for its right factor.

    Computes ``solution`` such that ``kron(left, solution) == right``.
    Leading  axes are broadcast according to NumPy's rules. Square ``left``
    has shape ``(..., n_left, n_left)`` and square ``right`` has shape
    ``(..., n_left * n_channels, n_left * n_channels)``. The returned
    matrix has shape ``(..., n_channels, n_channels)``.
    """
    n_channels_left, n_cols_left = left.shape[-2:]
    n_channels_right, n_cols_right = right.shape[-2:]

    if n_channels_left != n_cols_left:
        msg = "The final two axes of left must be square."
        raise ValueError(msg)

    if n_channels_right != n_cols_right:
        msg = "The final two axes of right must be square."
        raise ValueError(msg)

    n_channels, remainder = divmod(n_channels_right, n_channels_left)
    if remainder:
        msg = (
            "The dimensions of right must be divisible by the dimensions of left, "
            f"got {right.shape[-2:]} and {left.shape[-2:]}."
        )
        raise ValueError(msg)

    batch_shape = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = np.broadcast_to(left, (*batch_shape, n_channels_left, n_cols_left))
    right = np.broadcast_to(right, (*batch_shape, n_channels_right, n_cols_right))

    right_blocks = right.reshape(
        *batch_shape,
        n_channels_left,
        n_channels * n_channels_left * n_channels,
    )
    solution_blocks = np.linalg.solve(left, right_blocks)
    solution_blocks = solution_blocks.reshape(
        *batch_shape,
        n_channels_left,
        n_channels,
        n_channels_left,
        n_channels,
    )

    return np.trace(solution_blocks, axis1=-4, axis2=-2) / n_channels_left


def vec(array: NDArray[Any]) -> NDArray[Any]:
    """Vectorize the final two matrix axes of an array.

    Stacks each matrix column by column while preserving leading axes. An input of shape
    ``(..., n_rows, n_cols)`` produces an array of shape ``(..., n_rows * n_cols)``.
    """
    return array.reshape(*array.shape[:-2], -1, order="F")
