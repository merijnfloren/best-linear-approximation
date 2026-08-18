from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._config import TimeDomainSignal


def standardize_channels(
    signal: TimeDomainSignal,
) -> tuple[TimeDomainSignal, NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Standardize each channel in ``signal`` to zero mean and unit variance.

    ``signal`` must have shape ``(n_samples, n_channels, ...)``. For each channel,
    the mean and standard deviation are computed over the sample axis and all trailing
    axes. Constant channels are returned as zeros.

    Assumes that ``signal`` has already been validated as finite, real-valued data
    with at least two dimensions.

    Returns the standardized signal, followed by the per-channel means and standard
    deviations, both with shape ``(n_channels,)``.
    """
    reduction_axes = (0, *range(2, signal.ndim))
    means = signal.mean(axis=reduction_axes, keepdims=True)
    standard_deviations = signal.std(axis=reduction_axes, keepdims=True)
    standardized_signal = np.divide(
        signal - means,
        standard_deviations,
        out=np.zeros_like(signal),
        where=standard_deviations != 0,
    )
    return standardized_signal, means.reshape(-1), standard_deviations.reshape(-1)
