from __future__ import annotations

from typing import Any, NewType

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating[Any, Any]]
"""An array with a complex floating-point dtype."""


RealArray = NDArray[np.floating[Any]]
"""An array with a real floating-point dtype."""


FrequencyDomainSignal = NewType(
    "FrequencyDomainSignal",
    ComplexArray,
)
"""A validated frequency-domain signal in five-dimensional experiment layout.

The shape is ``(n_bins, n_channels, nu, n_experiments, n_periods)``, and
every axis has nonzero length. ``n_bins`` can either be all ``rfft`` bins, or
a subset of bins.
"""

TimeDomainSignal = NewType(
    "TimeDomainSignal",
    RealArray,
)
"""A validated time-domain signal in five-dimensional experiment layout.

The shape is ``(n_samples, n_channels, nu, n_experiments, n_periods)``, and
every axis has nonzero length.
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
