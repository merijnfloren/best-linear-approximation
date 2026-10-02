from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from best_linear_approximation._exceptions import (
    InsufficientExperimentsError,
    RealizationsTruncatedWarning,
)

if TYPE_CHECKING:
    from best_linear_approximation._typing import (
        FrequencyDomainSignal,
        RealArray,
        TimeDomainSignal,
    )

CANONICAL_SIGNAL_NDIM = 5
MINIMUM_SIGNAL_NDIM = 3
EXPERIMENT_LAYOUT_WITHOUT_PERIOD_NDIM = 4


def to_experiment_layout(
    signal: RealArray,
    nu: int,
    *,
    warn_on_truncation: bool = True,
) -> TimeDomainSignal:
    """Convert a signal from realization layout to five-dimensional experiment layout.

    Splits the realization axis of a signal with shape
    ``(n_samples, n_channels, n_realizations[, n_periods])`` into input and experiment
    axes using column-major ordering. Adds a singleton period axis if none is present.
    The resulting shape is ``(n_samples, n_channels, nu, n_experiments, n_periods)``.

    Raises an ``InsufficientExperimentsError`` if ``n_realizations < nu``. If
    ``warn_on_truncation`` is true, warns when ``n_realizations`` is not divisible by
    ``nu`` and the remaining realizations are discarded.
    """
    n_realizations = signal.shape[2]
    n_experiments = n_realizations // nu
    if n_experiments == 0:
        msg = (
            f"The number of realizations ({n_realizations}) is less than the number of "
            f"input channels ({nu}). No frequency response can be estimated."
        )
        raise InsufficientExperimentsError(msg)

    n_effective_realizations = n_experiments * nu
    if warn_on_truncation and n_effective_realizations < n_realizations:
        msg = (
            f"The number of realizations ({n_realizations}) is not a multiple of "
            f"the number of input channels ({nu}). Only the first "
            f"{n_effective_realizations} realizations will be used for estimation."
        )
        warnings.warn(msg, RealizationsTruncatedWarning, stacklevel=2)

    signal = signal[:, :, :n_effective_realizations, ...]
    signal = signal.reshape(*signal.shape[:2], nu, n_experiments, *signal.shape[3:], order="F")

    return cast(
        "TimeDomainSignal",
        signal
        if signal.ndim == CANONICAL_SIGNAL_NDIM
        else add_period_axis(signal),
    )


def add_period_axis(signal: RealArray) -> RealArray:
    """Add a singleton period axis to a four-dimensional experiment signal.

    Five-dimensional signals are returned unchanged.
    """
    return signal[..., None] if signal.ndim == EXPERIMENT_LAYOUT_WITHOUT_PERIOD_NDIM else signal


@overload
def as_batched_matrices(signal: TimeDomainSignal) -> TimeDomainSignal: ...


@overload
def as_batched_matrices(signal: FrequencyDomainSignal) -> FrequencyDomainSignal: ...


def as_batched_matrices(
    signal: TimeDomainSignal | FrequencyDomainSignal,
) -> TimeDomainSignal | FrequencyDomainSignal:
    """Arrange a signal as matrices for batched linear algebra.

    Transforms an array with shape ``(n_leading, n_rows, n_cols, ...)`` into
    one with shape ``(n_leading, ..., n_rows, n_cols)``.
    """
    if signal.ndim < MINIMUM_SIGNAL_NDIM:
        msg = (
            f"Expected a signal with at least {MINIMUM_SIGNAL_NDIM} dimensions, got {signal.ndim}."
        )
        raise ValueError(msg)

    return np.moveaxis(signal, (1, 2), (-2, -1))
