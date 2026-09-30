from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from best_linear_approximation._array_shapes import add_period_axis, to_experiment_layout
from best_linear_approximation._exceptions import (
    NoiseCovarianceUnavailableWarning,
    PossibleExcitationAmplitudeMismatchWarning,
    PossiblePeriodMismatchWarning,
    TotalCovarianceUnavailableWarning,
)
from best_linear_approximation._misc import rms, standardize_channels
from best_linear_approximation._signal_validation import (
    ContractType,
    SignalContract,
    validate_signal_contract,
)
from best_linear_approximation._spectral_validation import (
    resolve_excited_bins,
    validate_sampling_frequency,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from best_linear_approximation._typing import (
        ExcitedBins,
        RealArray,
        SamplingFrequencyHz,
        TimeDomainSignal,
    )

# Tuned to pass all tests in "tests/test_argument_preparation.py"
MINIMUM_RELATIVE_PERIOD_MISMATCH = 0.025  # keep in sync with docstring
MINIMUM_RELATIVE_REALIZATION_MISMATCH = 0.025  # keep in sync with docstring


@overload
def prepare_arguments(
    r: None,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
    contracts: Mapping[ContractType, SignalContract],
) -> tuple[
    None,
    TimeDomainSignal,
    TimeDomainSignal,
    SamplingFrequencyHz,
    ExcitedBins,
]: ...


@overload
def prepare_arguments(
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
    contracts: Mapping[ContractType, SignalContract],
) -> tuple[
    TimeDomainSignal,
    TimeDomainSignal,
    TimeDomainSignal,
    SamplingFrequencyHz,
    ExcitedBins,
]: ...


def prepare_arguments(  # noqa: PLR0913, PLR0917
    r: RealArray | None,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
    contracts: Mapping[ContractType, SignalContract],
) -> tuple[
    TimeDomainSignal | None,
    TimeDomainSignal,
    TimeDomainSignal,
    SamplingFrequencyHz,
    ExcitedBins,
]:
    """Validate and resolve all arguments.

    Ensures the signals conform to a supported contract and transforms them into the
    canonical five-dimensional experiment layout, verifies the sampling frequency,
    and resolves the excited bins. Warns when the data is insufficient to estimate the
    noise or total covariance, and when adjacent output periods differ substantially
    or excitation amplitudes change between realizations.

    If ``excited_bins`` is an array, it is validated directly. If it is a float, it is
    interpreted as a threshold for detecting the excited bins, using ``r`` if available
    and ``u`` otherwise.
    """
    contract_type = validate_signal_contract(r, u, y, contracts)

    nu = u.shape[1]
    if contract_type is ContractType.REALIZATION:
        r = to_experiment_layout(r, nu) if r is not None else None
        u = to_experiment_layout(u, nu)
        y = to_experiment_layout(y, nu)
    else:
        r = add_period_axis(r) if r is not None else None
        u = add_period_axis(u)

    r = cast("TimeDomainSignal", r) if r is not None else None
    u = cast("TimeDomainSignal", u)
    y = cast("TimeDomainSignal", y)

    fs = validate_sampling_frequency(fs)
    excited_bins = resolve_excited_bins(excited_bins, r if r is not None else u, fs)

    n_experiments, n_periods = y.shape[-2:]

    if n_experiments == 1:
        _warn_total_covariance_unavailable()
    if n_periods == 1:
        _warn_noise_covariance_unavailable()

    max_bin = excited_bins[-1]
    if n_experiments > 1:
        _warn_if_excitation_amplitude_mismatch(r if r is not None else u, max_bin)
    if n_periods > 1:
        _warn_if_output_spectra_mismatch(y, max_bin)

    return r, u, y, fs, excited_bins


def _warn_if_output_spectra_mismatch(y: RealArray, max_bin: int) -> None:
    """Warn when the aggregate spectral mismatch between adjacent periods exceeds 2.5%.

    Requires ``n_periods > 1``.

    Only ``rfft`` bins from DC through ``max_bin`` are used. An appropriate choice of
    ``max_bin`` reduces the influence of higher-frequency measurement noise and
    emphasizes mismatches arising from drift and initial-condition effects.
    """
    n_periods = y.shape[-1]
    if n_periods <= 1:
        msg = f"y must have more than one period, got n_periods={n_periods}."
        raise ValueError(msg)

    y = standardize_channels(y)

    stop_bin = max_bin + 1
    spectrum = np.fft.rfft(y, axis=0)[:stop_bin]
    later_spectrum = spectrum[..., 1:]
    spectral_difference = later_spectrum - spectrum[..., :-1]

    reduction_axis = tuple(range(y.ndim - 1))  # all axes except the last (period) axis
    difference_rms = rms(spectral_difference, axis=reduction_axis)
    reference_rms = rms(later_spectrum, axis=reduction_axis)

    relative_difference = np.divide(
        difference_rms,
        reference_rms,
        out=np.where(difference_rms == 0, 0.0, np.inf),
        where=reference_rms != 0,
    )

    exceeds_threshold = relative_difference > MINIMUM_RELATIVE_PERIOD_MISMATCH
    if np.any(exceeds_threshold):
        values = ", ".join(f"{value:.2%}" for value in relative_difference)
        affected_pairs = np.flatnonzero(exceeds_threshold)
        n_period_pairs = n_periods - 1
        n_affected_pairs = affected_pairs.size
        if n_period_pairs == 1:
            scope = "the"
            period_pair_label = "period pair"
            difference_label = "difference"
        else:
            scope = (
                f"all {n_period_pairs}"
                if n_affected_pairs == n_period_pairs
                else f"{n_affected_pairs} of {n_period_pairs}"
            )
            period_pair_label = "period pairs"
            difference_label = "differences"
        msg = (
            f"The relative spectral mismatch between {scope} adjacent {period_pair_label} "
            f"exceeds the threshold of {MINIMUM_RELATIVE_PERIOD_MISMATCH:.2%}. This may "
            f"indicate transients, drift, changing excitation, or period-alignment "
            f"issues. Aggregated relative {difference_label}: "
            f"[{values}]."
        )
        warnings.warn(msg, PossiblePeriodMismatchWarning, stacklevel=2)


def _warn_if_excitation_amplitude_mismatch(
    signal: RealArray,
    max_bin: int,
) -> None:
    """Warn when excitation magnitude changes between adjacent realizations.

    Requires ``n_experiments > 1``.

    Only ``rfft`` bins from DC through ``max_bin`` are used. The input and experiment
    axes are merged into realization layout using column-major ordering. Adjacent
    realizations are compared separately for every physical input channel. Each
    channel is standardized before comparison, so fixed amplitude differences between
    channels do not trigger a warning.
    """
    n_experiments = signal.shape[-2]
    if n_experiments <= 1:
        msg = (
            "signal must contain more than one experiment, "
            f"got n_experiments={n_experiments}."
        )
        raise ValueError(msg)

    signal = standardize_channels(signal)

    stop_bin = max_bin + 1
    spectrum = np.fft.rfft(signal, axis=0)[:stop_bin]
    n_bins, n_channels, nu, _, n_periods = spectrum.shape
    n_realizations = nu * n_experiments
    realization_spectrum = spectrum.reshape(
        n_bins,
        n_channels,
        n_realizations,
        n_periods,
        order="F",
    )
    realization_magnitude_spectrum = np.abs(realization_spectrum)

    later_magnitude_spectrum = realization_magnitude_spectrum[..., 1:, :]
    spectral_difference = later_magnitude_spectrum - realization_magnitude_spectrum[..., :-1, :]

    reduction_axes = (0, 3)
    difference_rms = rms(spectral_difference, axis=reduction_axes)
    reference_rms = rms(later_magnitude_spectrum, axis=reduction_axes)
    relative_difference = np.divide(
        difference_rms,
        reference_rms,
        out=np.where(difference_rms == 0, 0.0, np.inf),
        where=reference_rms != 0,
    )

    exceeds_threshold = relative_difference > MINIMUM_RELATIVE_REALIZATION_MISMATCH
    if np.any(exceeds_threshold):
        values = ", ".join(f"{value:.2%}" for value in relative_difference.ravel())
        affected_pairs = np.flatnonzero(exceeds_threshold)
        n_realization_pairs = n_realizations - 1
        n_affected_pairs = affected_pairs.size
        n_channel_realization_pairs = n_channels * n_realization_pairs
        if n_channel_realization_pairs == 1:
            scope = "the"
            channel_realization_pair_label = "channel-realization pair"
            difference_label = "difference"
        else:
            scope = (
                f"all {n_channel_realization_pairs}"
                if n_affected_pairs == n_channel_realization_pairs
                else f"{n_affected_pairs} of {n_channel_realization_pairs}"
            )
            channel_realization_pair_label = "channel-realization pairs"
            difference_label = "differences"
        msg = (
            f"The relative mismatch between {scope} adjacent {channel_realization_pair_label} "
            f"exceeds the threshold of {MINIMUM_RELATIVE_REALIZATION_MISMATCH:.2%}. This "
            "may indicate an excitation-amplitude change between realizations. Aggregated relative "
            f"{difference_label}: [{values}]."
        )
        warnings.warn(msg, PossibleExcitationAmplitudeMismatchWarning, stacklevel=2)


def _warn_noise_covariance_unavailable() -> None:
    msg = "Only a single period is provided, so the noise covariance cannot be estimated."
    warnings.warn(msg, NoiseCovarianceUnavailableWarning, stacklevel=2)


def _warn_total_covariance_unavailable() -> None:
    msg = (
        "Only a single experiment is provided, so the total covariance "
        "(noise plus nonlinear distortions) cannot be estimated."
    )
    warnings.warn(msg, TotalCovarianceUnavailableWarning, stacklevel=2)
