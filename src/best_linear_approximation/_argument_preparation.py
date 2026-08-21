import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._array_shapes import to_experiment_layout
from best_linear_approximation._exceptions import PossibleTransientWarning
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
from best_linear_approximation._typing import (
    ExcitedBins,
    FrequencyDomainSignal,
    SamplingFrequencyHz,
    TimeDomainSignal,
)

# Tuned to pass all tests in "tests/test_argument_preparation.py"
MINIMUM_RELATIVE_PERIOD_MISMATCH = 0.025  # keep in sync with docstring


# def prepare_arguments_direct(
#     u: NDArray[np.floating[Any]],
#     y: NDArray[np.floating[Any]],
#     fs: float,
#     excited_bins: NDArray[np.int_] | float,
#     contracts: Mapping[ContractType, SignalContract],
# ) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
#     """Validate and resolve all arguments.

#     Ensures the signals conform to a supported contract and transforms them into the
#     canonical five-dimensional experiment layout, verifies the sampling frequency,
#     and resolves the excited bins.

#     If ``excited_bins`` is an array, it is validated directly. If it is a float, it is
#     used as the threshold to detect the excited bins from ``u``.
#     """
#     contract_type = validate_signal_contract(None, u, y, contracts)

#     if contract_type == "realization":
#         nu = u.shape[1]
#         u = to_experiment_layout(u, nu)
#         y = to_experiment_layout(y, nu)

#     u = TimeDomainSignal(u)
#     y = TimeDomainSignal(y)

#     fs = validate_sampling_frequency(fs)
#     excited_bins = resolve_excited_bins(excited_bins, u, fs)

#     return u, y, fs, excited_bins


# def prepare_arguments_indirect(  # noqa: PLR0913, PLR0917
#     r: NDArray[np.floating[Any]],
#     u: NDArray[np.floating[Any]],
#     y: NDArray[np.floating[Any]],
#     fs: float,
#     excited_bins: NDArray[np.int_] | float,
#     contracts: Mapping[ContractType, SignalContract],
# ) -> tuple[TimeDomainSignal, TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
#     """Validate and resolve all arguments.

#     Ensures the signals conform to a supported contract and transforms them into the
#     canonical five-dimensional experiment layout, verifies the sampling frequency,
#     and resolves the excited bins.

#     If ``excited_bins`` is an array, it is validated directly. If it is a float, it is
#     used as the threshold to detect the excited bins from ``r``.
#     """
#     contract_type = validate_signal_contract(r, u, y, contracts)

#     if contract_type == "realization":
#         nu = u.shape[1]
#         r = to_experiment_layout(r, nu)
#         u = to_experiment_layout(u, nu)
#         y = to_experiment_layout(y, nu)

#     r = TimeDomainSignal(r)
#     u = TimeDomainSignal(u)
#     y = TimeDomainSignal(y)

#     fs = validate_sampling_frequency(fs)
#     excited_bins = resolve_excited_bins(excited_bins, r, fs)

#     return r, u, y, fs, excited_bins


def prepare_arguments(  # noqa: PLR0913, PLR0917
    r: NDArray[np.floating[Any]] | None,
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float,
    contracts: Mapping[ContractType, SignalContract],
) -> tuple[
    TimeDomainSignal | None,
    TimeDomainSignal,
    TimeDomainSignal,
    SamplingFrequencyHz,
    ExcitedBins
    ]:
    """Validate and resolve all arguments.

    Ensures the signals conform to a supported contract and transforms them into the
    canonical five-dimensional experiment layout, verifies the sampling frequency,
    and resolves the excited bins.

    If ``excited_bins`` is an array, it is validated directly. If it is a float, it is
    interpreted as a threshold for detecting the excited bins, using ``r`` if available
    and ``u`` otherwise.
    """
    contract_type = validate_signal_contract(r, u, y, contracts)

    if contract_type == "realization":
        nu = u.shape[1]
        r = to_experiment_layout(r, nu) if r is not None else None
        u = to_experiment_layout(u, nu)
        y = to_experiment_layout(y, nu)

    r = TimeDomainSignal(r) if r is not None else None
    u = TimeDomainSignal(u)
    y = TimeDomainSignal(y)

    fs = validate_sampling_frequency(fs)
    excited_bins = resolve_excited_bins(excited_bins, r if r is not None else u, fs)

    return r, u, y, fs, excited_bins


def warn_for_possible_transients(y: TimeDomainSignal, max_bin: int) -> None:
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
    later_spectrum = FrequencyDomainSignal(spectrum[..., 1:])
    spectral_difference = FrequencyDomainSignal(later_spectrum - spectrum[..., :-1])

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
            f"indicate non-steady-state behavior. Aggregated relative {difference_label}: "
            f"[{values}]."
        )
        warnings.warn(msg, PossibleTransientWarning, stacklevel=2)
