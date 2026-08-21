from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._typing import FrequencyDomainSignal, TimeDomainSignal


def rms(
    signal: TimeDomainSignal | FrequencyDomainSignal,
    axis: int | tuple[int, ...],
) -> NDArray[np.floating[Any]]:
    """Compute the root-mean-square along the specified axis or axes."""
    return np.sqrt(np.mean(np.abs(signal) ** 2, axis=axis))


def standardize_channels(signal: TimeDomainSignal) -> TimeDomainSignal:
    """Standardize each channel in ``signal`` to zero mean and unit variance."""
    reduction_axes = (0, *range(2, signal.ndim))
    means = signal.mean(axis=reduction_axes, keepdims=True)
    standard_deviations = signal.std(axis=reduction_axes, keepdims=True)
    standardized_signal = np.divide(
        signal - means,
        standard_deviations,
        out=np.zeros_like(signal),
        where=standard_deviations != 0,
    )
    return TimeDomainSignal(standardized_signal)
