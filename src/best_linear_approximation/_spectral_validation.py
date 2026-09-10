from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._exceptions import NoExcitedBinsError
from best_linear_approximation._misc import standardize_channels
from best_linear_approximation._typing import ExcitedBins, SamplingFrequencyHz

MultisineType = Literal["full", "odd", "even", "special-odd", "special-even", "special-mixed"]


MIN_FIXED_POINT_FREQUENCY_HZ = 0.1
MAX_FIXED_POINT_FREQUENCY_HZ = 10_000


def validate_sampling_frequency(fs: float) -> SamplingFrequencyHz:
    """Validate the supplied sampling frequency.

    Raises a ``TypeError`` if the sampling frequency is not a real number, and a
    ``ValueError`` if it is not finite and strictly positive.
    """
    if isinstance(fs, bool) or not isinstance(fs, Real):
        msg = f"Sampling frequency must be a real number, got {type(fs).__name__} = {fs!r}."
        raise TypeError(msg)

    if not np.isfinite(fs) or fs <= 0:
        msg = f"Sampling frequency must be finite and strictly positive, got {fs!r}."
        raise ValueError(msg)

    return SamplingFrequencyHz(fs)


def resolve_excited_bins(
    excited_bins: NDArray[np.int_] | float,
    signal: NDArray[np.floating[Any]],
    fs: SamplingFrequencyHz,
) -> ExcitedBins:
    """Validate and resolve the supplied excited bins.

    If ``excited_bins`` is an array, it is validated against the invariants of
    :func:`detect_excited_bins`. If it is a floating-point scalar, it is validated
    to lie in ``(0, 1)`` and then used as the relative threshold for detecting the
    bins from ``signal``.

    Raises a ``TypeError`` for unsupported input types and a ``ValueError`` for
    values that violate these requirements.
    """
    if isinstance(excited_bins, np.ndarray):
        if excited_bins.ndim != 1 or not np.issubdtype(excited_bins.dtype, np.integer):
            msg = "If supplied as an array, excited_bins must be a 1D array of integers."
            raise ValueError(msg)

        if excited_bins.size == 0:
            msg = "If supplied as an array, excited_bins must not be empty."
            raise ValueError(msg)

        max_excited_bin = (signal.shape[0] - 1) // 2
        if np.any(excited_bins < 1) or np.any(excited_bins > max_excited_bin):
            msg = (
                "If supplied as an array, excited_bins must contain only non-DC, non-Nyquist "
                f"rfft bins in [1, {max_excited_bin}], got values from {excited_bins.min()} "
                f"to {excited_bins.max()}."
            )
            raise ValueError(msg)

        if np.any(np.diff(excited_bins) <= 0):
            msg = (
                "If supplied as an array, excited_bins must be strictly increasing "
                "and contain no duplicates."
            )
            raise ValueError(msg)

        return ExcitedBins(excited_bins)

    if isinstance(excited_bins, (float, np.floating)):
        relative_threshold = excited_bins
        if not (0 < relative_threshold < 1):
            msg = "If supplied as a float, excited_bins must be in the range (0, 1)."
            raise ValueError(msg)

        return detect_excited_bins(signal, fs, relative_threshold)

    msg = (
        "excited_bins must be either a 1D integer array or a float in (0, 1), "
        f"got {type(excited_bins).__name__} = {excited_bins!r}."
    )
    raise TypeError(msg)


def detect_excited_bins(
    signal: NDArray[np.floating[Any]],
    fs: SamplingFrequencyHz,
    relative_threshold: float,
    *,
    print_summary: bool = True,
) -> ExcitedBins:
    """Detect the excited ``rfft`` bins of a time-domain excitation signal.

    The excitation signal is standardized independently for each channel along the sample
    axis to zero mean and unit variance. A one-sided FFT is then computed along the sample
    axis, after which the spectral magnitudes are averaged over all remaining axes. Excited
    bins are identified as those whose averaged magnitude exceeds a relative threshold
    with respect to the maximum averaged magnitude over all bins.

    Parameters
    ----------
    signal : NDArray[np.floating[Any]]
        Real-valued time-domain input ``u`` or reference signal ``r`` with shape
        ``(n_samples, n_channels, ...)``. Prefer ``r`` when available, as it typically
        has a higher signal-to-noise ratio.
    fs : SamplingFrequencyHz
        Sampling frequency in Hz.
    relative_threshold : float
        Relative threshold in the open interval ``(0, 1)``.
    print_summary : bool, optional
        Whether to print a summary which includes the number of excited bins, the frequency
        range, and the type of multisine. Default is ``True``.

    Returns
    -------
    ExcitedBins
        Indices of the excited ``rfft`` bins, shape ``(n_excited_bins,)``, with
        ``n_excited_bins <= (n_samples - 1) // 2``. DC and Nyquist are never
        returned, even if they exceed the threshold.

    Raises
    ------
    NoExcitedBinsError
        If no excited bins are found.

    """
    signal = standardize_channels(signal)

    magnitude = np.abs(np.fft.rfft(signal, axis=0))[1:]  # exclude DC bin
    magnitude_avg = magnitude.mean(axis=tuple(range(1, magnitude.ndim)))

    excited_bins = np.flatnonzero(magnitude_avg > relative_threshold * magnitude_avg.max())
    excited_bins += 1  # shift to account for excluded DC bin

    # Exclude Nyquist bin (only applies to even-length signals)
    n_samples = signal.shape[0]
    if n_samples % 2 == 0:
        excited_bins = excited_bins[excited_bins < n_samples // 2]

    if excited_bins.size == 0:
        msg = (
            "No excited frequency bins were found. "
            "Try lowering the relative threshold to make detection less strict."
        )
        raise NoExcitedBinsError(msg)

    excited_bins = ExcitedBins(excited_bins)

    if print_summary:
        multisine_type = _classify_multisine_type(excited_bins)
        _print_excited_bins_summary(multisine_type, excited_bins, n_samples, fs)

    return excited_bins


def _classify_multisine_type(excited_bins: ExcitedBins) -> MultisineType:
    """Classify the type of multisine excitation based on the excited bins."""
    first_bin, last_bin = excited_bins[0], excited_bins[-1]

    all_bins = np.arange(first_bin, last_bin + 1)
    if np.array_equal(excited_bins, all_bins):
        return "full"
    if np.all(excited_bins % 2 == 1):
        all_odd_bins = np.arange(first_bin, last_bin + 1, 2)
        if np.array_equal(excited_bins, all_odd_bins):
            return "odd"
        return "special-odd"
    if np.all(excited_bins % 2 == 0):
        all_even_bins = np.arange(first_bin, last_bin + 1, 2)
        if np.array_equal(excited_bins, all_even_bins):
            return "even"
        return "special-even"
    return "special-mixed"


def _print_excited_bins_summary(
    multisine_type: MultisineType,
    excited_bins: ExcitedBins,
    n_samples: int,
    fs: SamplingFrequencyHz,
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
        summary = (
            f"{summary_prefix} every bin in that interval is excited ({multisine_type} multisine)."
        )
    elif multisine_type == "odd":
        summary = (
            f"{summary_prefix} every odd bin in that interval is excited "
            f"({multisine_type} multisine)."
        )
    elif multisine_type == "even":
        summary = (
            f"{summary_prefix} every even bin in that interval is excited "
            f"({multisine_type} multisine)."
        )
    elif multisine_type == "special-odd":
        summary = (
            f"{summary_prefix} all excited bins are odd, but some odd bins "
            f"in that interval are missing ({multisine_type} multisine)."
        )
    elif multisine_type == "special-even":
        summary = (
            f"{summary_prefix} all excited bins are even, but some even bins "
            f"in that interval are missing ({multisine_type} multisine)."
        )
    else:  # multisine_type == "special-mixed"
        summary = (
            f"{summary_prefix} excitation spans odd and even bins, but some bins "
            f"in that interval are missing ({multisine_type} multisine)."
        )

    print(summary)  # noqa: T201


def _format_frequency(frequency_hz: float) -> str:
    """Format frequencies readably over ordinary and very small/large scales."""
    if frequency_hz == 0 or (
        MIN_FIXED_POINT_FREQUENCY_HZ <= frequency_hz < MAX_FIXED_POINT_FREQUENCY_HZ
    ):
        return f"{frequency_hz:.2f}"
    return f"{frequency_hz:.2e}"
