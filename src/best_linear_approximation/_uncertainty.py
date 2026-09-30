"""Uncertainty result type and frequency-domain integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from best_linear_approximation._typing import ComplexArray, RealArray


@dataclass(frozen=True)
class Uncertainty:
    """Uncertainty associated with an estimate evaluated at frequency lines.

    Attributes
    ----------
    cov : ComplexArray or None
        Joint covariance matrix at every frequency. ``None`` when the available
        experiment design cannot estimate the joint covariance.
    var : RealArray or None
        Component-wise variance of the frequency-domain quantity. When ``cov``
        is available, this is its diagonal reshaped to the quantity's value
        dimensions using column-wise ordering.
    std : RealArray or None
        Component-wise standard deviation, equal to ``sqrt(var)``.
    as_percentage : RealArray or None
        RMS uncertainty relative to the estimate's RMS value, expressed as a
        percentage. The frequency axis is reduced, giving one value per
        component with shape ``var.shape[1:]``.
    as_power_ratio_db : RealArray or None
        Estimate's RMS value relative to its RMS uncertainty, expressed in dB.
        The frequency axis is reduced, giving one value per component with
        shape ``var.shape[1:]``.

    """

    cov: ComplexArray | None
    _marginal_var_shape: tuple[int, ...]
    _signal_power: RealArray | None = None
    _frequency_weights: RealArray | None = None

    @classmethod
    def from_cov(
        cls,
        covariance: ComplexArray,
        marginal_var_shape: tuple[int, ...],
        *,
        spectrum: ComplexArray | None = None,
        n_samples: int | None = None,
        frequency_bins: NDArray[np.int_] | None = None,
    ) -> Self:
        """Create covariance-backed uncertainty for a frequency-domain quantity."""
        if spectrum is None or n_samples is None or frequency_bins is None:
            return cls(covariance, marginal_var_shape)

        frequency_weights = _compute_frequency_weights(n_samples, frequency_bins)
        signal_power = _compute_signal_power(
            spectrum,
            frequency_weights,
            marginal_var_shape,
        )
        return cls(covariance, marginal_var_shape, signal_power, frequency_weights)

    @classmethod
    def unavailable(cls) -> Self:
        """Create uncertainty with no estimable covariance or variance."""
        return cls(None, ())

    @property
    def var(self) -> RealArray | None:
        """Component-wise variance of the frequency-domain quantity."""
        if self.cov is None:
            return None

        variance = np.diagonal(self.cov, axis1=-2, axis2=-1).real
        return variance.reshape(self.cov.shape[0], *self._marginal_var_shape, order="F")

    @property
    def std(self) -> RealArray | None:
        """Component-wise standard deviation of the frequency-domain quantity."""
        if self.var is None:
            return None
        return np.sqrt(self.var)

    @property
    def as_percentage(self) -> RealArray | None:
        """RMS uncertainty relative to the estimate's RMS value in percent."""
        disturbance_power = self._disturbance_power
        if self._signal_power is None or disturbance_power is None:
            return None
        return 100 * np.sqrt(disturbance_power / self._signal_power)

    @property
    def as_power_ratio_db(self) -> RealArray | None:
        """Ratio in dB of the estimate's RMS value to its RMS uncertainty."""
        disturbance_power = self._disturbance_power
        if self._signal_power is None or disturbance_power is None:
            return None
        return 10 * np.log10(self._signal_power / disturbance_power)

    @property
    def _disturbance_power(self) -> RealArray | None:
        """Integrate marginal spectral variance over the uncertainty frequency bins."""
        if self.var is None or self._frequency_weights is None:
            return None
        return np.einsum("f,f...->...", self._frequency_weights, self.var)


def _compute_frequency_weights(
    n_samples: int,
    frequency_bins: NDArray[np.int_],
) -> RealArray:
    """Return Parseval weights for selected bins of a real ``rfft`` spectrum."""
    weights = np.full(frequency_bins.size, 2.0)
    dc_indices = np.flatnonzero(frequency_bins == 0)
    weights[dc_indices] = 1.0

    if n_samples % 2 == 0:
        nyquist_bin = n_samples // 2
        nyquist_indices = np.flatnonzero(frequency_bins == nyquist_bin)
        weights[nyquist_indices] = 1.0

    return weights


def _compute_signal_power(
    spectrum: ComplexArray,
    frequency_weights: RealArray,
    marginal_var_shape: tuple[int, ...],
) -> RealArray:
    """Compute weighted frequency-domain power, averaging sample dimensions."""
    power = np.einsum("f,f...->...", frequency_weights, np.abs(spectrum) ** 2)
    n_value_dimensions = len(marginal_var_shape)
    sample_axes = tuple(range(n_value_dimensions, power.ndim))
    return np.mean(power, axis=sample_axes) if sample_axes else power
