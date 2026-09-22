"""Result types for best linear approximation estimates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
class FrequencyResponse:
    """Frequency response estimate and its covariance estimates.

    Attributes
    ----------
    value : ComplexArray
        Frequency response, with shape ``(n_excited_bins, ny, nu)``.
    cov_total : ComplexArray or None
        Total covariance of ``value``. ``None`` when fewer than two experiments
        are available. Has shape ``(n_excited_bins, ny * nu, ny * nu)``.
    cov_noise : ComplexArray or None
        Noise covariance of ``value``. ``None`` when fewer than two periods are
        available. Has shape ``(n_excited_bins, ny * nu, ny * nu)``.
    cov_nonlinear : ComplexArray or None
        Nonlinear-distortion covariance, equal to ``cov_total - cov_noise``.
        ``None`` when either covariance is unavailable. Has shape
        ``(n_excited_bins, ny * nu, ny * nu)``.
    var_total, var_noise, var_nonlinear : RealArray or None
        Per-frequency variances of each response element, with shape
        ``(n_excited_bins, ny, nu)``. Each is the corresponding covariance
        diagonal, reshaped from column-wise vectorization.
    std_total, std_noise, std_nonlinear : RealArray or None
        Per-frequency standard deviations of each response element, with shape
        ``(n_excited_bins, ny, nu)``.

    """

    value: ComplexArray
    cov_total: ComplexArray | None
    cov_noise: ComplexArray | None
    cov_nonlinear: ComplexArray | None

    @property
    def var_total(self) -> RealArray | None:
        """Total variance for every frequency-response element."""
        return self._covariance_diagonal(self.cov_total)

    @property
    def var_noise(self) -> RealArray | None:
        """Noise variance for every frequency-response element."""
        return self._covariance_diagonal(self.cov_noise)

    @property
    def var_nonlinear(self) -> RealArray | None:
        """Nonlinear-distortion variance for every frequency-response element."""
        return self._covariance_diagonal(self.cov_nonlinear)

    @property
    def std_total(self) -> RealArray | None:
        """Total standard deviation for every frequency-response element."""
        return self._standard_deviation(self.var_total)

    @property
    def std_noise(self) -> RealArray | None:
        """Noise standard deviation for every frequency-response element."""
        return self._standard_deviation(self.var_noise)

    @property
    def std_nonlinear(self) -> RealArray | None:
        """Nonlinear-distortion standard deviation for every response element."""
        return self._standard_deviation(self.var_nonlinear)

    def _covariance_diagonal(
        self,
        covariance: ComplexArray | None,
    ) -> RealArray | None:
        """Reshape a covariance diagonal into the response matrix layout."""
        if covariance is None:
            return None

        variances = np.diagonal(covariance, axis1=-2, axis2=-1).real
        return variances.reshape(*self.value.shape[:-2], *self.value.shape[-2:], order="F")

    @staticmethod
    def _standard_deviation(
        variance: RealArray | None,
    ) -> RealArray | None:
        """Return the standard deviation associated with a variance array."""
        if variance is None:
            return None

        return np.sqrt(variance)


@dataclass(frozen=True)
class BLA:
    """Best linear approximation estimate.

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
) -> BLA:
    """Create a BLA result and derive its nonlinear-distortion covariance."""
    if cov_total is None or cov_noise is None:
        cov_G_nonlinear = None
    else:
        cov_G_nonlinear = cov_total - cov_noise

        # Remove negative-eigenvalue artifacts to enforce positive semidefiniteness
        hermitian_covariance = (cov_G_nonlinear + cov_G_nonlinear.conj().mT) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian_covariance)
        nonnegative_eigenvalues = np.maximum(eigenvalues, 0)
        cov_G_nonlinear = (
            eigenvectors * nonnegative_eigenvalues[..., None, :]
        ) @ eigenvectors.conj().mT

    frequency_response_estimate = FrequencyResponse(
        frequency_response,
        cov_total,
        cov_noise,
        cov_G_nonlinear,
    )
    return BLA(frequency_info, frequency_response_estimate)
