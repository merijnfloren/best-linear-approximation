"""Result types for best linear approximation estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Self

import numpy as np

from best_linear_approximation._covariance import project_onto_positive_semidefinite
from best_linear_approximation._uncertainty import Uncertainty

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from best_linear_approximation._spectra import Spectra
    from best_linear_approximation._typing import (
        ComplexArray,
        ExcitedBins,
        RealArray,
        SamplingFrequencyHz,
        TimeDomainSignal,
    )


@dataclass(frozen=True)
class FrequencyResponse:
    """Frequency response estimate and its uncertainty estimates."""

    value: ComplexArray
    noise: Uncertainty
    nonlinear: Uncertainty
    total: Uncertainty


class EstimationMethod(Enum):
    """Method used to estimate a best linear approximation."""

    ROBUST_DIRECT_KNOWN_INPUT = "robust_direct_known_input"
    ROBUST_DIRECT_NOISY_INPUT = "robust_direct_noisy_input"
    ROBUST_INDIRECT = "robust_indirect"
    ROBUST_INDIRECT_KNOWN_REFERENCE = "robust_indirect_known_reference"
    ROBUST_INDIRECT_CLOSED_LOOP = "robust_indirect_closed_loop"


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
    G_bla: ComplexArray,
    G_total_cov: ComplexArray | None,
    G_noise_cov: ComplexArray | None,
    n_samples: int,
    excited_bins: ExcitedBins,
) -> FrequencyResponse:
    """Create a frequency response estimate from its value and covariances."""
    nonlinear_cov = None
    if G_total_cov is not None and G_noise_cov is not None:
        nonlinear_cov = project_onto_positive_semidefinite(G_total_cov - G_noise_cov)

    marginal_var_shape = G_bla.shape[1:]
    noise = _create_frequency_domain_uncertainty(
        G_noise_cov,
        marginal_var_shape,
        G_bla,
        n_samples,
        excited_bins,
    )
    nonlinear = _create_frequency_domain_uncertainty(
        nonlinear_cov,
        marginal_var_shape,
        G_bla,
        n_samples,
        excited_bins,
    )
    total = _create_frequency_domain_uncertainty(
        G_total_cov,
        marginal_var_shape,
        G_bla,
        n_samples,
        excited_bins,
    )
    return FrequencyResponse(G_bla, noise, nonlinear, total)


def _create_frequency_domain_uncertainty(
    covariance: ComplexArray | None,
    marginal_var_shape: tuple[int, ...],
    spectrum: ComplexArray,
    n_samples: int,
    frequency_bins: ExcitedBins,
) -> Uncertainty:
    if covariance is None:
        return Uncertainty.unavailable()

    return Uncertainty.from_cov(
        covariance,
        marginal_var_shape,
        spectrum=spectrum,
        n_samples=n_samples,
        frequency_bins=frequency_bins,
    )
