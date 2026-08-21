from typing import Any, NewType

import numpy as np
from numpy.typing import NDArray

TimeDomainSignal = NewType(
    "TimeDomainSignal",
    NDArray[np.floating[Any]],
)
"""A validated time-domain signal in five-dimensional experiment layout.

The shape is ``(n_samples, n_channels, nu, n_experiments, n_periods)``, and
every axis has nonzero length.
"""


FrequencyDomainSignal = NewType(
    "FrequencyDomainSignal",
    NDArray[np.complexfloating[Any, Any]],
)
"""A validated frequency-domain signal in five-dimensional experiment layout.

The shape is ``(n_bins, n_channels, nu, n_experiments, n_periods)``, and every
axis has nonzero length.
"""


ExcitedBins = NewType("ExcitedBins", NDArray[np.int_])
"""A validated array of excited frequency bins.

The shape is ``(n_excited_bins,)``, where ``n_excited_bins <= (n_samples - 1) // 2``.
The values index excited ``rfft`` bins and exclude DC and Nyquist.
"""


SamplingFrequencyHz = NewType("SamplingFrequencyHz", float)
"""A validated sampling frequency in hertz.

The value is finite and strictly positive.
"""
