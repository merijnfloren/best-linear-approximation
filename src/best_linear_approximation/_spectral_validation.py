from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._exceptions import NoExcitedBinsError
from ._misc import TimeDomainSignal, standardize_channels

MultisineType = Literal["full", "odd", "even", "special-odd", "special-even", "special-mixed"]


_MIN_FIXED_POINT_FREQUENCY_HZ = 0.1
_MAX_FIXED_POINT_FREQUENCY_HZ = 10_000


def detect_excited_bins(
    excitation: TimeDomainSignal,
    fs: float,
    relative_threshold: float,
) -> NDArray[np.int_]:
    """Detect the excited ``rfft`` bins of a time-domain excitation signal.

    The excitation is standardized independently for each channel along the sample axis
    to zero mean and unit variance. A one-sided FFT is then computed along the sample axis,
    after which the spectral magnitudes are averaged over all remaining axes. Excited bins
    are identified as those whose averaged magnitude exceeds a relative threshold with
    respect to the maximum averaged magnitude over all bins.

    Prints a summary of the detected bins, including the number of excited bins,
    the frequency range, and whether all bins in that range are excited or if
    some are missing.

    Assumes that all arguments have already been validated.

    Parameters
    ----------
    excitation : TimeDomainSignal, shape ``(n_samples, n_channels, ...)``
        Real-valued time-domain input ``u`` or reference signal ``r``.
        The latter is preferred because it is typically cleaner.
    fs : float
        Sampling frequency in Hz.
    relative_threshold : float
        Relative threshold in the open interval ``(0, 1)``.

    Returns
    -------
    NDArray[np.int_]
        Indices of the excited ``rfft`` bins, shape ``(n_excited_bins,)``, with
        ``n_excited_bins <= (n_samples - 1) // 2``. DC and Nyquist are never
        returned, even if they exceed the threshold.

    """
    excitation = standardize_channels(excitation)[0]

    magnitude = np.abs(np.fft.rfft(excitation, axis=0))
    magnitude_avg = magnitude.mean(axis=tuple(range(1, magnitude.ndim)))

    excited_bins = np.flatnonzero(magnitude_avg > relative_threshold * magnitude_avg.max())

    # # Exclude the DC bin (standardization should have removed it, but just in case)
    # excited_bins = excited_bins[excited_bins > 0]

    # Exclude the Nyquist bin (only applies to even-length signals)
    n_samples = excitation.shape[0]
    if n_samples % 2 == 0:
        excited_bins = excited_bins[excited_bins < n_samples // 2]

    if excited_bins.size == 0:
        msg = (
            "No excited frequency bins were found. "
            "Try lowering the relative threshold to make detection less strict."
        )
        raise NoExcitedBinsError(msg)

    multisine_type = _classify_multisine_type(excited_bins)
    _ = _print_excited_bins_summary(multisine_type, excited_bins, n_samples, fs)
    return excited_bins


def _classify_multisine_type(excited_bins: NDArray[np.int_]) -> MultisineType:
    """Classify the type of multisine excitation based on the excited bins."""
    first_bin, last_bin = excited_bins[0], excited_bins[-1]

    all_bins = np.arange(first_bin, last_bin + 1)
    if np.array_equal(excited_bins, all_bins):
        return "full"
    elif np.all(excited_bins % 2 == 1):
        all_odd_bins = np.arange(first_bin, last_bin + 1, 2)
        if np.array_equal(excited_bins, all_odd_bins):
            return "odd"
        else:
            return "special-odd"
    elif np.all(excited_bins % 2 == 0):
        all_even_bins = np.arange(first_bin, last_bin + 1, 2)
        if np.array_equal(excited_bins, all_even_bins):
            return "even"
        else:
            return "special-even"
    else:
        return "special-mixed"
    
    
def _print_excited_bins_summary(
    multisine_type: MultisineType,
    excited_bins: NDArray[np.int_],
    n_samples: int,
    fs: float,
) -> None:
    """Print a compact description of the selected bins."""
    first_bin, last_bin = excited_bins[0], excited_bins[-1]
    first_frequency = _format_frequency(first_bin * fs / n_samples)
    last_frequency = _format_frequency(last_bin * fs / n_samples)

    summary_prefix = (
        f"Detected {excited_bins.size} excited frequency bins from "
        f"{first_frequency} Hz to {last_frequency} Hz;"
    )

    if multisine_type == "full":
        summary = f"{summary_prefix} every bin in that interval is excited."
    elif multisine_type == "odd":
        summary = f"{summary_prefix} every odd bin in that interval is excited."
    elif multisine_type == "even":
        summary = f"{summary_prefix} every even bin in that interval is excited."
    elif multisine_type == "special-odd":
        summary = (
            f"{summary_prefix} all excited bins are odd, but some odd bins "
            "in that interval are missing."
        )
    elif multisine_type == "special-even":
        summary = (
            f"{summary_prefix} all excited bins are even, but some even bins "
            "in that interval are missing."
        )
    else:  # multisine_type == "special-mixed"
        summary = (
            f"{summary_prefix} excitation spans odd and even bins, but some bins "
            "in that interval are missing."
        )

    print(summary)  # noqa: T201


def _format_frequency(frequency_hz: float) -> str:
    """Format frequencies readably over ordinary and very small/large scales."""
    if frequency_hz == 0 or (
        _MIN_FIXED_POINT_FREQUENCY_HZ <= abs(frequency_hz) < _MAX_FIXED_POINT_FREQUENCY_HZ
    ):
        return f"{frequency_hz:.2f}"
    return f"{frequency_hz:.2e}"
