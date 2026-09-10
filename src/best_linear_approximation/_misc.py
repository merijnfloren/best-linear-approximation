from typing import Any

import numpy as np
from numpy.typing import NDArray


def rms(
    signal: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> NDArray[np.floating[Any]]:
    """Compute root-mean-square values along the specified axes.

    If ``axis`` is ``None``, compute one RMS value over the entire signal.
    """
    return np.sqrt(np.mean(np.abs(signal) ** 2, axis=axis, keepdims=keepdims))


def standardize_channels(
    signal: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Standardize channels of an ``(n_leading, n_channels, ...)`` signal."""
    reduction_axes = (0, *range(2, signal.ndim))
    means = np.mean(signal, axis=reduction_axes, keepdims=True)
    standard_deviations = signal.std(axis=reduction_axes, keepdims=True)

    return np.divide(
        signal - means,
        standard_deviations,
        out=np.zeros_like(signal),
        where=standard_deviations != 0,
    )
