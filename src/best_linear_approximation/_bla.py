"""Result types for best linear approximation estimates."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._covariance import project_onto_positive_semidefinite
from best_linear_approximation._typing import ComplexArray, RealArray


@dataclass(frozen=True)
class FrequencyInfo:
    """Metadata for the frequency content of a best linear approximation.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    f_res : float
        Frequency resolution in Hz.
    f_min : float
        Lowest excited frequency in Hz.
    f_max : float
        Highest excited frequency in Hz.
    freqs : RealArray
        Non-negative DFT frequencies in Hz.
    excited_bins : NDArray[np.int_]
        Indices of the excited DFT frequencies.
    non_excited_bins : NDArray[np.int_]
        Indices of the non-excited DFT frequencies.

    """

    fs: float
    f_res: float
    f_min: float
    f_max: float
    freqs: RealArray
    excited_bins: NDArray[np.int_]
    non_excited_bins: NDArray[np.int_]


@dataclass(frozen=True)
class SpectralUncertainty:
    """Uncertainty of a frequency-domain quantity.

    Attributes
    ----------
    cov : ComplexArray or None
        Joint covariance matrix at every frequency. ``None`` when the available
        experiment design cannot estimate the joint covariance.
    var : RealArray or None
        Component-wise variance of the frequency-domain quantity. When ``cov``
        is available, this is its diagonal reshaped to the quantity's value
        dimensions using column-wise ordering. Otherwise, it is a pooled
        component-wise variance estimate when that is identifiable without a
        joint covariance matrix.
    std : RealArray or None
        Component-wise standard deviation, equal to ``sqrt(var)``.

    """

    cov: ComplexArray | None
    _value_shape: tuple[int, ...]
    _pooled_var: RealArray | None

    def __post_init__(self) -> None:
        """Validate that uncertainty has a single source for marginal variances."""
        if self.cov is not None and self._pooled_var is not None:
            msg = "cov and _pooled_var cannot both be provided."
            raise ValueError(msg)

        if self._pooled_var is not None and self._pooled_var.shape[1:] != self._value_shape:
            msg = "_pooled_var dimensions must match _value_shape."
            raise ValueError(msg)

    @classmethod
    def from_cov(
        cls,
        covariance: ComplexArray,
        value_shape: tuple[int, ...],
    ) -> Self:
        """Create covariance-backed uncertainty for a frequency-domain quantity."""
        return cls(covariance, value_shape, None)

    @classmethod
    def from_pooled_var(cls, variance: RealArray) -> Self:
        """Create uncertainty from a pooled component-wise variance estimate."""
        return cls(None, variance.shape[1:], variance)

    @classmethod
    def unavailable(cls, value_shape: tuple[int, ...]) -> Self:
        """Create uncertainty with no estimable covariance or variance."""
        return cls(None, value_shape, None)

    @property
    def var(self) -> RealArray | None:
        """Component-wise variance of the frequency-domain quantity."""
        if self.cov is None:
            return self._pooled_var

        variance = np.diagonal(self.cov, axis1=-2, axis2=-1).real
        return variance.reshape(self.cov.shape[0], *self._value_shape, order="F")

    @property
    def std(self) -> RealArray | None:
        """Component-wise standard deviation of the frequency-domain quantity."""
        if self.var is None:
            return None

        return np.sqrt(self.var)


@dataclass(frozen=True)
class FrequencyResponse:
    """Frequency response estimate and its uncertainty estimates.

    Attributes
    ----------
    value : ComplexArray
        Frequency response, with shape ``(n_excited_bins, ny, nu)``.
    total : SpectralUncertainty
        Total distortion uncertainty of ``value``.
    noise : SpectralUncertainty
        Noise distortion uncertainty of ``value``.
    nonlinear : SpectralUncertainty
        Nonlinear distortion uncertainty of ``value``.

    """

    value: ComplexArray
    total: SpectralUncertainty
    noise: SpectralUncertainty
    nonlinear: SpectralUncertainty


@dataclass(frozen=True)
class NonparametricBLA:
    """Nonparametric best linear approximation estimate.

    Attributes
    ----------
    freq : FrequencyInfo
        Frequency metadata for the estimate.
    G : FrequencyResponse
        Frequency response estimate and its covariance and variance estimates.

    """

    freq: FrequencyInfo
    G: FrequencyResponse


def create_frequency_info(
    n_samples: int,
    fs: float,
    excited_bins: NDArray[np.int_],
) -> FrequencyInfo:
    """Create frequency metadata from validated sampling and excitation data."""
    n_freqs = n_samples // 2 + 1
    f_res = fs / n_samples
    freqs = np.arange(n_freqs) * f_res
    non_excited_bins = np.setdiff1d(np.arange(n_freqs), excited_bins)
    f_min = float(freqs[excited_bins[0]])
    f_max = float(freqs[excited_bins[-1]])
    return FrequencyInfo(
        fs=fs,
        f_res=f_res,
        f_min=f_min,
        f_max=f_max,
        freqs=freqs,
        excited_bins=excited_bins,
        non_excited_bins=non_excited_bins,
    )


def create_bla(
    frequency_info: FrequencyInfo,
    frequency_response: ComplexArray,
    cov_total: ComplexArray | None,
    cov_noise: ComplexArray | None,
) -> NonparametricBLA:
    """Create a nonparametric BLA and derive its nonlinear-distortion covariance."""
    if cov_total is None or cov_noise is None:
        cov_nonlinear = None
    else:
        cov_nonlinear = cov_total - cov_noise
        cov_nonlinear = project_onto_positive_semidefinite(cov_nonlinear)

    value_shape = frequency_response.shape[1:]
    total = _create_spectral_uncertainty(cov_total, value_shape)
    noise = _create_spectral_uncertainty(cov_noise, value_shape)
    nonlinear = _create_spectral_uncertainty(cov_nonlinear, value_shape)
    frequency_response_estimate = FrequencyResponse(
        frequency_response,
        total,
        noise,
        nonlinear,
    )
    return NonparametricBLA(frequency_info, frequency_response_estimate)


def _create_spectral_uncertainty(
    covariance: ComplexArray | None,
    value_shape: tuple[int, ...],
) -> SpectralUncertainty:
    """Create uncertainty from an optional covariance."""
    if covariance is None:
        return SpectralUncertainty.unavailable(value_shape)

    return SpectralUncertainty.from_cov(covariance, value_shape)
