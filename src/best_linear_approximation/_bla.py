"""Result types for best linear approximation estimates."""

from dataclasses import dataclass
from enum import Enum
from typing import Self

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._covariance import project_onto_positive_semidefinite
from best_linear_approximation._typing import (
    ComplexArray,
    ExcitedBins,
    RealArray,
    SamplingFrequencyHz,
    TimeDomainSignal,
)


@dataclass(frozen=True)
class FrequencyDomainUncertainty:
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
    _marginal_var_shape: tuple[int, ...]

    @classmethod
    def from_cov(cls, covariance: ComplexArray, marginal_var_shape: tuple[int, ...]) -> Self:
        """Create covariance-backed uncertainty for a frequency-domain quantity."""
        return cls(covariance, marginal_var_shape)

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


@dataclass(frozen=True)
class _FrequencyDomainEstimate:
    value: ComplexArray
    total: FrequencyDomainUncertainty
    nonlinear: FrequencyDomainUncertainty
    noise: FrequencyDomainUncertainty


@dataclass(frozen=True)
class FrequencyResponse(_FrequencyDomainEstimate):
    """Frequency response estimate and its uncertainty estimates."""


@dataclass(frozen=True)
class InputSpectrum(_FrequencyDomainEstimate):
    """Spectrum estimate and its uncertainty estimates."""


@dataclass(frozen=True)
class OutputSpectrum(_FrequencyDomainEstimate):
    """Spectrum estimate and its uncertainty estimates."""

    total_equation_error: FrequencyDomainUncertainty


class EstimationMethod(Enum):
    """Method used to estimate a best linear approximation."""

    ROBUST_DIRECT_KNOWN_INPUT = "robust_direct_known_input"
    ROBUST_DIRECT_NOISY_INPUT = "robust_direct_noisy_input"
    ROBUST_INDIRECT_CLOSED_LOOP = "robust_indirect_closed_loop"
    ROBUST_INDIRECT_KNOWN_REFERENCE = "robust_indirect_known_reference"


@dataclass(frozen=True)
class ExperimentInfo:
    """Metadata describing the recordings used to estimate a BLA.

    Attributes
    ----------
    estimation_method : EstimationMethod
        Method used to estimate the BLA.
    n_samples : int
        Number of time-domain samples in each period.
    nu : int
        Number of input channels.
    ny : int
        Number of output channels.
    n_experiments : int
        Number of experiments.
    n_periods : int
        Number of output periods per experiment.
    u_shape : tuple of int
        Canonical time-domain shape of the input signal.
    y_shape : tuple of int
        Canonical time-domain shape of the output signal.
    r_shape : tuple of int or None
        Canonical time-domain shape of the reference signal, when available.

    """

    estimation_method: EstimationMethod
    n_samples: int
    nu: int
    ny: int
    n_experiments: int
    n_periods: int
    u_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    r_shape: tuple[int, ...] | None = None

    @property
    def n_realizations(self) -> int:
        """Number of effective realizations after arranging data by input direction."""
        return self.nu * self.n_experiments

    @classmethod
    def from_signals(
        cls,
        estimation_method: EstimationMethod,
        u: TimeDomainSignal,
        y: TimeDomainSignal,
        r: TimeDomainSignal | None = None,
    ) -> Self:
        """Create recording metadata from canonical time-domain signals."""
        n_samples, ny, nu, n_experiments, n_periods = y.shape
        return cls(
            estimation_method=estimation_method,
            n_samples=n_samples,
            nu=nu,
            ny=ny,
            n_experiments=n_experiments,
            n_periods=n_periods,
            u_shape=u.shape,
            y_shape=y.shape,
            r_shape=r.shape if r is not None else None,
        )


@dataclass(frozen=True)
class Spectra:
    """Spectra used to estimate a BLA and their available spectrum-level uncertainty."""

    U: InputSpectrum
    Y: OutputSpectrum
    R: InputSpectrum | None = None


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
class NonparametricBLA:
    """Nonparametric best linear approximation estimate.

    Attributes
    ----------
    G : FrequencyResponse
        Frequency response estimate and its uncertainty estimates.
    spectra : Spectra
        Input, output, and optional reference spectra with their uncertainty estimates.
    freq : FrequencyInfo
        Frequency metadata for the estimate.
    experiment : ExperimentInfo
        Recording metadata and estimation method.

    """
    G: FrequencyResponse
    spectra: Spectra
    freq: FrequencyInfo
    experiment: ExperimentInfo


def create_frequency_info(
    n_samples: int,
    fs: SamplingFrequencyHz,
    excited_bins: ExcitedBins,
) -> FrequencyInfo:
    """Create frequency metadata from validated sampling and excitation data."""
    n_bins = n_samples // 2 + 1
    f_res = fs / n_samples
    freqs = np.arange(n_bins) * f_res
    non_excited_bins = np.setdiff1d(np.arange(n_bins), excited_bins)
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


def create_bla_frequency_response(
    frequency_response: ComplexArray,
    cov_total: ComplexArray | None,
    cov_noise: ComplexArray | None,
) -> FrequencyResponse:
    """Create a frequency response estimate from its value and covariances."""
    cov_nonlinear = None
    if cov_total is not None and cov_noise is not None:
        cov_nonlinear = project_onto_positive_semidefinite(cov_total - cov_noise)

    marginal_var_shape = frequency_response.shape[1:]
    total = _create_frequency_domain_uncertainty(cov_total, marginal_var_shape)
    nonlinear = _create_frequency_domain_uncertainty(cov_nonlinear, marginal_var_shape)
    noise = _create_frequency_domain_uncertainty(cov_noise, marginal_var_shape)
    return FrequencyResponse(frequency_response, total, nonlinear, noise)


def _create_frequency_domain_uncertainty(
    covariance: ComplexArray | None,
    marginal_var_shape: tuple[int, ...],
) -> FrequencyDomainUncertainty:
    """Create available or unavailable frequency-domain uncertainty."""
    if covariance is None:
        return FrequencyDomainUncertainty.unavailable()

    return FrequencyDomainUncertainty.from_cov(covariance, marginal_var_shape)
