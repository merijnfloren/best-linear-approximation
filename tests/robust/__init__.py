"""Shared fixtures and assertions for robust-method tests."""

from dataclasses import dataclass

import numpy as np
import pytest
from multisine import (
    RandomPhaseMultisine,
    random_phase_multisine,
    random_phase_orthogonal_multisine,
)
from numpy.typing import NDArray

from best_linear_approximation import NonparametricBLA
from best_linear_approximation._array_shapes import as_batched_matrices, to_experiment_layout
from best_linear_approximation._exceptions import (
    NoiseCovarianceUnavailableWarning,
    TotalCovarianceUnavailableWarning,
)
from best_linear_approximation._typing import ComplexArray, RealArray

covariance_availability_cases = pytest.mark.parametrize(
    ("n_experiments", "n_periods"),
    [(1, 2), (2, 1)],
    ids=["one_experiment", "one_period"],
)

bla_recovery_cases = pytest.mark.parametrize(
    ("ny", "nu"),
    [(1, 1), (3, 2)],
    ids=["siso", "rectangular_mimo"],
)

bla_recovery_seeds = pytest.mark.parametrize("seed", [7, 19])

bla_excitation_cases = pytest.mark.parametrize(
    "orthogonal",
    [True, False],
    ids=["orthogonal", "nonorthogonal"],
)

bla_disturbance_cases = pytest.mark.parametrize(
    ("nonlinear_std", "noise_std"),
    [(0.0, 0.01), (0.02, 0.0), (0.02, 0.01)],
    ids=["noise_only", "nonlinear_only", "mixed"],
)


@dataclass
class TestSetup:
    """Shared multisine excitation and true frequency response for a test."""

    multisine: RandomPhaseMultisine
    U: ComplexArray
    G_true: ComplexArray
    n_experiments: int
    n_periods: int
    n_samples: int


def generate_correlation_matrix(
    rng: np.random.Generator,
    n_excited_bins: int,
    n_channels: int,
) -> ComplexArray:
    """Generate per-frequency complex correlation matrices with unit diagonal."""
    shape = (n_excited_bins, n_channels, n_channels)
    mixing_matrix = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    covariance = mixing_matrix @ mixing_matrix.conj().mT
    scales = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1).real)
    normalization = scales[:, :, None] * scales[:, None, :]
    return covariance / normalization


def generate_covariance_availability_signals(
    n_experiments: int,
    n_periods: int,
) -> tuple[RealArray, RealArray, RealArray]:
    """Generate a single-input, single-output data set with one excited bin."""
    n_samples = 8
    samples = np.arange(n_samples)
    period = np.sin(2 * np.pi * samples / n_samples)
    r = np.broadcast_to(period[:, None, None], (n_samples, 1, n_experiments))
    u = np.broadcast_to(r[..., None], (n_samples, 1, n_experiments, n_periods))
    y = 2 * u
    return r, u, y


def covariance_unavailable_warning(n_experiments: int) -> type[Warning]:
    """Return the warning for the covariance unavailable with this experiment count."""
    if n_experiments == 1:
        return TotalCovarianceUnavailableWarning
    return NoiseCovarianceUnavailableWarning


def sample_disturbances(
    covariance: ComplexArray,
    rng: np.random.Generator,
    n_experiments: int,
    n_periods: int,
    nu: int,
) -> ComplexArray:
    """Sample recordings with the prescribed channel covariance."""
    n_excited_bins, n_channels = covariance.shape[:2]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    shape = (n_excited_bins, n_experiments, n_periods, n_channels, nu)
    independent = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    return factor[:, None, None] @ independent


def to_time_domain(
    spectrum: ComplexArray,
    n_samples: int,
    excited_bins: NDArray[np.int_],
) -> RealArray:
    """Convert excited spectra to real periods in the public realization layout."""
    n_freqs = n_samples // 2 + 1
    full_spectrum = np.zeros((n_freqs, *spectrum.shape[1:]), dtype=complex)
    full_spectrum[excited_bins] = spectrum
    signal = np.fft.irfft(full_spectrum, n=n_samples, axis=0)
    n_samples, n_experiments, n_periods, n_channels, nu = signal.shape
    n_realizations = nu * n_experiments
    return signal.transpose(0, 3, 4, 1, 2).reshape(
        n_samples,
        n_channels,
        n_realizations,
        n_periods,
        order="F",
    )


def generate_test_setup(
    ny: int,
    nu: int,
    seed: int,
    orthogonal: bool,
    *,
    n_experiments: int = 2048,
    n_periods: int = 8,
) -> TestSetup:
    """Generate the shared multisine excitation and true frequency response."""
    n_samples = 32
    fs = 128.0
    if orthogonal:
        multisine = random_phase_orthogonal_multisine(
            n_samples,
            fs,
            nu,
            n_experiments=n_experiments,
            f_min=8.0,
            f_max=36.0,
            seed=seed,
        )
        u = multisine.u
    else:
        max_condition_number = 10.0
        n_candidate_experiments = 8 * n_experiments
        n_realizations = n_candidate_experiments * nu
        multisine = random_phase_multisine(
            n_samples,
            fs,
            nu=nu,
            n_realizations=n_realizations,
            f_min=8.0,
            f_max=36.0,
            seed=seed,
        )
        u_candidates = to_experiment_layout(multisine.u, nu)[..., 0]
        excited_bins = multisine.freq.excited_bins
        U_candidates = as_batched_matrices(np.fft.rfft(u_candidates, axis=0)[excited_bins])
        condition_numbers = np.linalg.cond(U_candidates)
        well_conditioned = np.all(condition_numbers <= max_condition_number, axis=0)
        experiment_indices = np.flatnonzero(well_conditioned)[:n_experiments]
        assert experiment_indices.size == n_experiments
        u = u_candidates[:, :, :, experiment_indices]

    excited_bins = multisine.freq.excited_bins
    n_excited_bins = excited_bins.size
    G_rng = np.random.default_rng(seed)
    G_shape = (n_excited_bins, ny, nu)
    G_true = G_rng.normal(scale=0.5, size=G_shape) + 1j * G_rng.normal(scale=0.5, size=G_shape)
    U = as_batched_matrices(np.fft.rfft(u, axis=0)[excited_bins])
    return TestSetup(multisine, U, G_true, n_experiments, n_periods, n_samples)


def assert_bla_recovery_and_covariances(
    bla: NonparametricBLA,
    G_true: ComplexArray,
    G_total_cov_expected: ComplexArray,
    G_noise_cov_expected: ComplexArray,
    n_experiments: int,
    n_periods: int,
) -> None:
    """Verify BLA recovery and its covariance estimates."""
    n_total_covariance_dof = n_experiments - 1
    n_noise_covariance_dof = n_experiments * (n_periods - 1)
    assert bla.G.total.cov is not None
    assert bla.G.noise.cov is not None
    assert bla.G.nonlinear.cov is not None
    _assert_plant_recovery(bla.G.value, G_true, G_total_cov_expected)
    _assert_covariance(bla.G.total.cov, G_total_cov_expected, n_total_covariance_dof)
    _assert_covariance(bla.G.noise.cov, G_noise_cov_expected, n_noise_covariance_dof)
    expected_shape = G_true.shape
    assert bla.G.total.var is not None
    assert bla.G.noise.var is not None
    assert bla.G.nonlinear.var is not None
    assert bla.G.total.std is not None
    assert bla.G.noise.std is not None
    assert bla.G.nonlinear.std is not None
    assert bla.G.total.var.shape == expected_shape
    assert bla.G.noise.var.shape == expected_shape
    assert bla.G.nonlinear.var.shape == expected_shape
    assert bla.G.total.std.shape == expected_shape
    assert bla.G.noise.std.shape == expected_shape
    assert bla.G.nonlinear.std.shape == expected_shape
    for covariance, variances in (
        (bla.G.total.cov, bla.G.total.var),
        (bla.G.noise.cov, bla.G.noise.var),
        (bla.G.nonlinear.cov, bla.G.nonlinear.var),
    ):
        expected_variances = np.diagonal(covariance, axis1=-2, axis2=-1).real
        expected_variances = expected_variances.reshape(expected_shape, order="F")
        np.testing.assert_array_equal(variances, expected_variances)
    np.testing.assert_array_equal(bla.G.total.std, np.sqrt(bla.G.total.var))
    np.testing.assert_array_equal(bla.G.noise.std, np.sqrt(bla.G.noise.var))
    np.testing.assert_array_equal(bla.G.nonlinear.std, np.sqrt(bla.G.nonlinear.var))
    nonlinear_eigenvalues = np.linalg.eigvalsh(bla.G.nonlinear.cov)
    np.testing.assert_array_less(-1e-24, nonlinear_eigenvalues)


def _assert_plant_recovery(
    G_estimated: ComplexArray,
    G_true: ComplexArray,
    expected_covariance: ComplexArray,
) -> None:
    """Check recovery using the prescribed covariance of the final BLA estimate."""
    assert G_estimated.shape == G_true.shape
    errors = (G_estimated - G_true).swapaxes(-1, -2).reshape(G_estimated.shape[0], -1)
    variances = np.diagonal(expected_covariance, axis1=-2, axis2=-1).real
    tolerance = 6 * np.sqrt(np.maximum(variances, 0)) + 1e-12
    np.testing.assert_array_less(np.abs(errors), tolerance)


def _assert_covariance(
    estimated: ComplexArray,
    expected: ComplexArray,
    n_degrees_of_freedom: int,
) -> None:
    """Check entries, allowing for finite-sample covariance scatter."""
    assert estimated.shape == expected.shape
    np.testing.assert_allclose(estimated, estimated.conj().mT, atol=1e-24)
    variances = np.maximum(np.diagonal(expected, axis1=-2, axis2=-1).real, 0)
    entry_scales = np.sqrt(variances[:, :, None] * variances[:, None, :])
    tolerance = 6 * entry_scales / np.sqrt(n_degrees_of_freedom) + 1e-24
    np.testing.assert_array_less(np.abs(estimated - expected), tolerance)
