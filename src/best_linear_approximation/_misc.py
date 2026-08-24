from typing import Any

import numpy as np
from numpy.typing import NDArray


def rms(
    signal: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
    axis: int | tuple[int, ...],
) -> NDArray[np.floating[Any]]:
    """Compute the root-mean-square along the specified axis or axes."""
    return np.sqrt(np.mean(np.abs(signal) ** 2, axis=axis))


def standardize_channels(
    signal: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Standardize channels of an ``(n_leading, n_channels, ...)`` signal."""
    reduction_axes = (0, *range(2, signal.ndim))
    means = signal.mean(axis=reduction_axes, keepdims=True)
    standard_deviations = signal.std(axis=reduction_axes, keepdims=True)

    return np.divide(
        signal - means,
        standard_deviations,
        out=np.zeros_like(signal),
        where=standard_deviations != 0,
    )
