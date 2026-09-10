from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._misc import rms
from best_linear_approximation._spectral_validation import validate_sampling_frequency

MIN_NUMBER_OF_SAMPLES = 4  # keep in sync with docstrings


@dataclass
class FrequencyInfo:
    """Metadata for the frequency content of a multisine signal.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    f_res : float
        Frequency resolution in Hz.
    f_min : float
        Minimum frequency in Hz (snapped to grid).
    f_max : float
        Maximum frequency in Hz (snapped to grid).
    freqs : NDArray[np.floating[Any]]
        Non-negative frequencies in Hz, shape (n_samples // 2 + 1,).
    excited_bins : NDArray[np.int_]
        Indices of excited bins in freqs.
    non_excited_bins : NDArray[np.int_]
        Indices of non-excited bins in freqs.

    """

    fs: float
    f_res: float
    f_min: float
    f_max: float
    freqs: NDArray[np.floating[Any]]
    excited_bins: NDArray[np.int_]
    non_excited_bins: NDArray[np.int_]


@dataclass
class RandomPhaseMultisine:
    """A random-phase multisine signal with associated metadata.

    Attributes
    ----------
    u : NDArray[np.floating[Any]]
        The time-domain multisine signal; depending on the generation method,
        it either has shape ``(n_samples, nu, n_realizations)`` or
        ``(n_samples, nu, nu, n_experiments)``, where ``n_samples`` is the number
        of samples, ``nu`` is the number of input channels, ``n_realizations`` is
        the number of independent phase realizations, and ``n_experiments`` is the
        number of orthogonal multisine experiments.
    freq : FrequencyInfo
        Frequency metadata of the multisine signal.
    amplitude : float or tuple of float
        The requested root-mean-square (RMS) amplitude of the multisine signal.
        If a float, the RMS amplitude is the same for all channels and realizations.
        If a tuple, it contains one RMS amplitude per input channel.
    seed : int
        The random seed used to generate the multisine signal.

    """

    u: NDArray[np.floating[Any]]
    freq: FrequencyInfo
    amplitude: float | tuple[float, ...]
    seed: int


def random_phase_multisine(  # noqa: PLR0913
    n_samples: int,
    fs: float,
    *,
    amplitude: float | tuple[float, ...] = 1.0,
    nu: int = 1,
    n_realizations: int = 1,
    f_min: float | None = None,
    f_max: float | None = None,
    seed: int = 42,
) -> RandomPhaseMultisine:
    """Generate a random-phase multisine signal.

    Parameters
    ----------
    n_samples : int
        Number of samples in each realization.
    fs : float
        Sampling frequency in Hz.
    amplitude : float or tuple of float, optional
        Desired root-mean-square (RMS) amplitude. A float sets the RMS amplitude
        over all channels and realizations. A tuple supplies one RMS amplitude
        per input channel and must have length ``nu``. Default is ``1.0``.
    nu : int, optional
        Number of input channels. Default is ``1``.
    n_realizations : int, optional
        Number of independent phase realizations per input channel. Default is ``1``.
    f_min : float, optional
        Lowest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the first non-DC frequency bin.
    f_max : float, optional
        Highest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the bin below the Nyquist frequency.
    seed : int, optional
        Seed for the random phase generator. Default is ``42``.

    Returns
    -------
    RandomPhaseMultisine
        The generated signal of shape ``(n_samples, nu, n_realizations)``, the
        corresponding frequency information, the requested RMS amplitude, and
        the random seed used.

    Raises
    ------
    TypeError
        If ``fs`` is not real; if ``n_samples``, ``nu``,
        ``n_realizations``, or ``seed`` is not an integer; or if ``amplitude``
        is neither a float nor a tuple of floats.
    ValueError
        If ``fs`` is non-finite or not strictly positive; if an integer count
        is not strictly positive; if ``n_samples`` is less than four; if
        ``amplitude`` is non-finite or not strictly positive; if the
        tuple-valued ``amplitude`` does not have length ``nu``; if ``f_max`` is
        at or above the Nyquist frequency; or if the snapped minimum frequency
        exceeds the snapped maximum frequency.

    """
    fs = validate_sampling_frequency(fs)
    _validate_strictly_positive_int(n_samples, "n_samples")
    _validate_strictly_positive_int(nu, "nu")
    _validate_strictly_positive_int(n_realizations, "n_realizations")
    _validate_strictly_positive_int(seed, "seed")
    _validate_amplitude(amplitude, nu)

    freq = _create_frequency_info(n_samples, fs, f_min, f_max)
    rng = np.random.default_rng(seed)

    n_freqs = freq.freqs.size

    # Create multisine in frequency domain
    spectrum = np.zeros((n_freqs, nu, n_realizations), dtype=complex)
    spectrum[freq.excited_bins] = 1 + 0j

    phase = 1j * rng.uniform(0, 2 * np.pi, (n_freqs, nu, n_realizations))
    spectrum *= np.exp(phase)

    # Convert to time domain
    u = np.fft.irfft(spectrum, n=n_samples, axis=0)

    u = _ensure_requested_amplitude(u, amplitude)

    return RandomPhaseMultisine(u=u, freq=freq, amplitude=amplitude, seed=seed)


def random_phase_orthogonal_multisine(  # noqa: PLR0913
    n_samples: int,
    fs: float,
    nu: int,
    *,
    amplitude: float | tuple[float, ...] = 1.0,
    n_experiments: int = 1,
    independent_subexperiments: bool = True,
    f_min: float | None = None,
    f_max: float | None = None,
    seed: int = 42,
) -> RandomPhaseMultisine:
    """Generate random-phase orthogonal multisine signals.

    Parameters
    ----------
    n_samples : int
        Number of samples in each experiment.
    fs : float
        Sampling frequency in Hz.
    nu : int
        Number of input channels.
    amplitude : float or tuple of float, optional
        Desired root-mean-square (RMS) amplitude. A float sets the RMS amplitude
        over all channels and experiments. A tuple supplies one RMS amplitude
        per input channel and must have length ``nu``. Default is ``1.0``.
    n_experiments : int, optional
        Number of orthogonal multisine experiments. Default is ``1``.
    independent_subexperiments : bool, optional
        Whether to apply an independently drawn additional phase to each of the
        ``nu`` orthogonal subexperiments within every experiment. This allows
        estimation of the covariance of stochastic nonlinear output distortions
        without compromising orthogonality. Default is ``True``.
    f_min : float, optional
        Lowest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the first non-DC frequency bin.
    f_max : float, optional
        Highest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the bin below the Nyquist frequency.
    seed : int, optional
        Seed for the random phase generator. Default is ``42``.

    Returns
    -------
    RandomPhaseMultisine
        The generated signal of shape
        ``(n_samples, nu, n_subexperiments, n_experiments)``, where
        ``n_subexperiments == nu``, the corresponding frequency information,
        the requested RMS amplitude, and the random seed used.

    Raises
    ------
    TypeError
        If ``fs`` is not real; if ``n_samples``, ``nu``, ``n_experiments``, or
        ``seed`` is not an integer; or if ``amplitude`` is neither a float nor
        a tuple of floats.
    ValueError
        If ``fs`` is non-finite or not strictly positive; if an integer count
        is not strictly positive; if ``n_samples`` is less than four; if
        ``amplitude`` is non-finite or not strictly positive; if the
        tuple-valued ``amplitude`` does not have length ``nu``; if ``f_max`` is
        at or above the Nyquist frequency; or if the snapped minimum frequency
        exceeds the snapped maximum frequency.

    """
    fs = validate_sampling_frequency(fs)
    _validate_strictly_positive_int(n_samples, "n_samples")
    _validate_strictly_positive_int(nu, "nu")
    _validate_strictly_positive_int(n_experiments, "n_experiments")
    _validate_strictly_positive_int(seed, "seed")
    _validate_amplitude(amplitude, nu)

    freq = _create_frequency_info(n_samples, fs, f_min, f_max)
    rng = np.random.default_rng(seed)

    n_freqs = freq.freqs.size

    # Create multisine in frequency domain
    dft_matrix = np.fft.fft(np.eye(nu)) / np.sqrt(nu)
    spectrum = np.zeros((n_freqs, nu, nu, n_experiments), dtype=complex)
    spectrum[freq.excited_bins] = dft_matrix[None, :, :, None]

    phase = 1j * rng.uniform(0, 2 * np.pi, (n_freqs, nu, n_experiments))
    spectrum *= np.exp(phase)[:, :, None, :]

    if independent_subexperiments:
        experiment_phase = 1j * rng.uniform(0, 2 * np.pi, (n_freqs, nu, n_experiments))
        spectrum *= np.exp(experiment_phase)[:, None, :, :]

    # Convert to time domain
    u = np.fft.irfft(spectrum, n=n_samples, axis=0)

    u = _ensure_requested_amplitude(u, amplitude)

    return RandomPhaseMultisine(u=u, freq=freq, amplitude=amplitude, seed=seed)


def _validate_amplitude(amplitude: float | tuple[float, ...], nu: int) -> None:
    if isinstance(amplitude, tuple):
        if len(amplitude) != nu:
            msg = f"amplitude must contain one value per input channel, got {amplitude!r}."
            raise ValueError(msg)
        if not all(isinstance(value, float) for value in amplitude):
            msg = f"amplitude must be a tuple of floats, got {amplitude!r}."
            raise TypeError(msg)
        if any(not np.isfinite(value) or value <= 0 for value in amplitude):
            msg = (
                f"amplitude must contain only finite, strictly positive values, "
                f"got {amplitude!r}."
            )
            raise ValueError(msg)
    elif not isinstance(amplitude, float):
        msg = f"amplitude must be a float or a tuple of floats, got {amplitude!r}."
        raise TypeError(msg)
    elif not np.isfinite(amplitude) or amplitude <= 0:
        msg = f"amplitude must be finite and strictly positive, got {amplitude!r}."
        raise ValueError(msg)


def _validate_strictly_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = (
            f"{name} must be a strictly positive integer, got {type(value).__name__} = "
            f"{value!r}."
        )
        raise TypeError(msg)
    if value <= 0:
        msg = f"{name} must be a strictly positive integer, got {value!r}."
        raise ValueError(msg)


def _create_frequency_info(
    n_samples: int,
    fs: float,
    f_min: float | None,
    f_max: float | None,
) -> FrequencyInfo:
    if n_samples < MIN_NUMBER_OF_SAMPLES:
        msg = (
            f"The number of samples must be at least {MIN_NUMBER_OF_SAMPLES} to "
            f"allow for a meaningful frequency range, got n_samples={n_samples!r}."
        )
        raise ValueError(msg)

    f_res = fs / n_samples
    f_min_idx = 1 if f_min is None else int(np.round(f_min / f_res))
    f_max_idx = n_samples // 2 - 1 if f_max is None else int(np.round(f_max / f_res))
    f_min_idx = max(f_min_idx, 1)  # ensure that DC is not excited

    if f_max_idx >= n_samples // 2:
        msg = (
            f"The maximum frequency must be less than the Nyquist frequency "
            f"(fs / 2), got f_max={f_max!r}, fs={fs!r}."
        )
        raise ValueError(msg)

    if f_min_idx > f_max_idx:
        msg = (
            f"The minimum frequency must be less than or equal to the maximum "
            f"frequency, got f_min={f_min!r}, f_max={f_max!r}."
        )
        raise ValueError(msg)

    n_freqs = n_samples // 2 + 1
    freqs = np.arange(n_freqs) * f_res
    excited_bins = np.arange(f_min_idx, f_max_idx + 1)
    non_excited_bins = np.setdiff1d(np.arange(n_freqs), excited_bins)

    return FrequencyInfo(
        fs=fs,
        f_res=f_res,
        f_min=f_min_idx * f_res,
        f_max=f_max_idx * f_res,
        freqs=freqs,
        excited_bins=excited_bins,
        non_excited_bins=non_excited_bins,
    )


def _ensure_requested_amplitude(
    u: NDArray[np.floating[Any]],
    requested_amplitude: float | tuple[float, ...],
) -> NDArray[np.floating[Any]]:
    if isinstance(requested_amplitude, float):
        current_amplitude = rms(u)
        return u * (requested_amplitude / current_amplitude)

    channel_axis = 1
    nu = u.shape[channel_axis]

    # Compute one RMS value per channel over samples and all trailing axes
    reduction_axes = (0, *range(channel_axis + 1, u.ndim))
    current_amplitudes = rms(u, axis=reduction_axes, keepdims=True)

    # Keep the channel axis while broadcasting across every reduced axis
    n_trailing_axes = u.ndim - channel_axis - 1
    amplitude_shape = (1, nu) + (1,) * n_trailing_axes
    requested_amplitudes = np.asarray(requested_amplitude).reshape(amplitude_shape)
    return u * (requested_amplitudes / current_amplitudes)
