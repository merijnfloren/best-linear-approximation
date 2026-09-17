from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._covariance import propagate_covariance
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation.multisine._multisine import (
    RandomPhaseMultisine,
    random_phase_orthogonal_multisine,
)
from best_linear_approximation.robust._direct_methods import known_input, noisy_input

pytestmark = pytest.mark.parametrize(
    ("ny", "nu"),
    [(1, 1), (3, 2)],
    ids=["siso", "rectangular_mimo"],
)


@dataclass
class _TestSetup:
    multisine: RandomPhaseMultisine
    U: NDArray[np.complexfloating[Any, Any]]
    G_true: NDArray[np.complexfloating[Any, Any]]
    n_experiments: int
    n_periods: int
    n_samples: int


@pytest.mark.parametrize("seed", [7, 19])
@pytest.mark.parametrize(
    ("nonlinear_std", "noise_std"),
    [(0.0, 0.01), (0.02, 0.0), (0.02, 0.01)],
    ids=["noise_only", "nonlinear_only", "mixed"],
)
def test_known_input_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
) -> None:
    """Verify known-input BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = _generate_test_setup(ny, nu, seed)
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    U = setup.U
    G_true = setup.G_true
    Y_linear = setup.G_true[:, None] @ U

    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    cov_Y_nonlinear = nonlinear_var * _generate_correlation_matrix(rng, n_excited_bins, ny)
    cov_Y_noise = noise_var * _generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = _sample_disturbances(cov_Y_nonlinear, rng, n_experiments, 1, nu)
    Y_noise = _sample_disturbances(cov_Y_noise, rng, n_experiments, n_periods, nu)

    Y = Y_linear[:, :, None] + Y_nonlinear + Y_noise

    u = _to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y = _to_time_domain(Y, n_samples, multisine.freq.excited_bins)

    G_estimated, cov_G_total_estimated, cov_G_noise_estimated = known_input(
        u,
        y,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )

    cov_G_total_expected, cov_G_noise_expected = _compute_oracle_covariances(
        U,
        cov_Y_noise,
        cov_Y_nonlinear,
        n_periods,
        known_input=True,
    )

    _assert_bla_recovery_and_covariances(
        G_estimated,
        G_true,
        cov_G_total_estimated,
        cov_G_noise_estimated,
        cov_G_total_expected,
        cov_G_noise_expected,
        n_experiments,
        n_periods,
    )


@pytest.mark.parametrize("seed", [7, 19])
@pytest.mark.parametrize(
    ("nonlinear_std", "noise_std"),
    [(0.0, 0.01), (0.02, 0.0), (0.02, 0.01)],
    ids=["noise_only", "nonlinear_only", "mixed"],
)
def test_noisy_input_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
) -> None:
    """Verify noisy-input BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = _generate_test_setup(ny, nu, seed)
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size
    n_channels = ny + nu

    U = setup.U
    G_true = setup.G_true
    Y_linear = G_true[:, None] @ U

    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    cov_Z_nonlinear = nonlinear_var * _generate_correlation_matrix(rng, n_excited_bins, n_channels)
    cov_Z_noise = noise_var * _generate_correlation_matrix(rng, n_excited_bins, n_channels)
    Z_nonlinear = _sample_disturbances(cov_Z_nonlinear, rng, n_experiments, 1, nu)
    Z_noise = _sample_disturbances(cov_Z_noise, rng, n_experiments, n_periods, nu)
    Y_nonlinear, U_nonlinear = Z_nonlinear[..., :ny, :], Z_nonlinear[..., ny:, :]
    Y_noise, U_noise = Z_noise[..., :ny, :], Z_noise[..., ny:, :]

    U_measured = U[:, :, None] + U_nonlinear + U_noise
    Y = Y_linear[:, :, None] + G_true[:, None, None] @ U_nonlinear
    Y = Y + Y_nonlinear + Y_noise

    u = _to_time_domain(U_measured, n_samples, multisine.freq.excited_bins)
    y = _to_time_domain(Y, n_samples, multisine.freq.excited_bins)
    G_estimated, cov_G_total_estimated, cov_G_noise_estimated = noisy_input(
        u,
        y,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )

    cov_G_total_expected, cov_G_noise_expected = _compute_oracle_covariances(
        U,
        cov_Z_noise,
        cov_Z_nonlinear,
        n_periods,
        G_true=G_true,
    )

    _assert_bla_recovery_and_covariances(
        G_estimated,
        G_true,
        cov_G_total_estimated,
        cov_G_noise_estimated,
        cov_G_total_expected,
        cov_G_noise_expected,
        n_experiments,
        n_periods,
    )


def _generate_correlation_matrix(
    rng: np.random.Generator,
    n_excited_bins: int,
    n_channels: int,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Generate per-frequency complex correlation matrices with unit diagonal."""
    shape = (n_excited_bins, n_channels, n_channels)
    mixing_matrix = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    covariance = mixing_matrix @ mixing_matrix.conj().mT
    scales = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1).real)
    normalization = scales[:, :, None] * scales[:, None, :]
    return covariance / normalization


def _sample_disturbances(
    covariance: NDArray[np.complexfloating[Any, Any]],
    rng: np.random.Generator,
    n_experiments: int,
    n_periods: int,
    nu: int,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Sample independent recordings with the prescribed channel covariance.

    Returns shape (n_excited_bins, n_experiments, n_periods, n_channels, nu).
    A singleton period axis holds nonlinear variation fixed across all periods.
    Covariances may be zero; the eigenvalue factorization also handles that case.
    """
    n_excited_bins, n_channels = covariance.shape[:2]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    shape = (n_excited_bins, n_experiments, n_periods, n_channels, nu)
    independent = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    return factor[:, None, None] @ independent


def _to_time_domain(
    spectrum: NDArray[np.complexfloating[Any, Any]],
    n_samples: int,
    excited_bins: NDArray[np.int_],
) -> NDArray[np.float64]:
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


def _generate_test_setup(
    ny: int,
    nu: int,
    seed: int,
) -> _TestSetup:
    """Generate the shared multisine excitation and true frequency response."""
    n_experiments = 2048
    n_samples = 32
    fs = 128.0
    n_periods = 8
    multisine = random_phase_orthogonal_multisine(
        n_samples,
        fs,
        nu,
        n_experiments=n_experiments,
        f_min=8.0,
        f_max=36.0,
        seed=seed,
    )
    excited_bins = multisine.freq.excited_bins
    n_excited_bins = excited_bins.size
    plant_rng = np.random.default_rng(seed)
    plant_shape = (n_excited_bins, ny, nu)
    G_true = plant_rng.normal(scale=0.5, size=plant_shape) + 1j * plant_rng.normal(
        scale=0.5,
        size=plant_shape,
    )
    U = as_batched_matrices(np.fft.rfft(multisine.u, axis=0)[excited_bins])
    return _TestSetup(
        multisine,
        U,
        G_true,
        n_experiments,
        n_periods,
        n_samples,
    )


def _compute_oracle_covariances(
    U: NDArray[np.complexfloating[Any, Any]],
    cov_noise: NDArray[np.complexfloating[Any, Any]],
    cov_nonlinear: NDArray[np.complexfloating[Any, Any]],
    n_periods: int,
    G_true: NDArray[np.complexfloating[Any, Any]] | None = None,
    *,
    known_input: bool = False,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]],
]:
    """Compute expected total and noise BLA covariances.

    Based on Eq. (2-77) in Pintelon, R., and Schoukens, J. (2012).
    *System Identification: A Frequency Domain Approach*, 2nd ed.,
    Wiley-IEEE Press, ISBN 978-0-470-64037-1.
    """
    n_excited_bins, n_experiments, nu, _ = U.shape
    U_inv_transpose = np.linalg.solve(U, np.eye(nu)).mT
    if known_input:
        ny = cov_noise.shape[-1]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        V = I_ny
    else:
        assert G_true is not None
        ny = G_true.shape[1]
        n_channels = ny + nu

        # Input nonlinearities drive the plant before both signals are measured
        nonlinear_transform = np.broadcast_to(
            np.eye(n_channels, dtype=complex),
            (n_excited_bins, n_channels, n_channels),
        ).copy()
        nonlinear_transform[:, :ny, ny:] = G_true
        cov_nonlinear = nonlinear_transform @ cov_nonlinear @ nonlinear_transform.conj().mT
        
        G_true_per_experiment = np.broadcast_to(
            G_true[:, None],
            (n_excited_bins, n_experiments, ny, nu),
        )
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        V = np.concatenate((I_ny, -G_true_per_experiment), axis=-1)

    jacobian = kronecker_product(U_inv_transpose, V)
    I_nu = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu))

    cov_noise = kronecker_product(I_nu, cov_noise)
    cov_nonlinear = kronecker_product(I_nu, cov_nonlinear)
    cov_G_noise_expected = (
        np.mean(
            propagate_covariance(cov_noise[:, None], jacobian),
            axis=1,
        )
        / n_experiments
        / n_periods
    )
    cov_G_nonlinear_expected = (
        np.mean(
            propagate_covariance(cov_nonlinear[:, None], jacobian),
            axis=1,
        )
        / n_experiments
    )
    cov_G_total_expected = cov_G_nonlinear_expected + cov_G_noise_expected
    return cov_G_total_expected, cov_G_noise_expected


def _assert_bla_recovery_and_covariances(
    G_estimated: NDArray[np.complexfloating[Any, Any]],
    G_true: NDArray[np.complexfloating[Any, Any]],
    cov_G_total_estimated: NDArray[np.complexfloating[Any, Any]] | None,
    cov_G_noise_estimated: NDArray[np.complexfloating[Any, Any]] | None,
    cov_G_total_expected: NDArray[np.complexfloating[Any, Any]],
    cov_G_noise_expected: NDArray[np.complexfloating[Any, Any]],
    n_experiments: int,
    n_periods: int,
) -> None:
    """Verify BLA recovery and total and noise covariance estimates."""
    n_total_covariance_dof = n_experiments - 1
    n_noise_covariance_dof = n_experiments * (n_periods - 1)
    assert cov_G_total_estimated is not None
    assert cov_G_noise_estimated is not None
    _assert_plant_recovery(G_estimated, G_true, cov_G_total_expected)
    _assert_covariance(cov_G_total_estimated, cov_G_total_expected, n_total_covariance_dof)
    _assert_covariance(cov_G_noise_estimated, cov_G_noise_expected, n_noise_covariance_dof)


def _assert_plant_recovery(
    G_estimated: NDArray[np.complexfloating[Any, Any]],
    G_true: NDArray[np.complexfloating[Any, Any]],
    expected_covariance: NDArray[np.complexfloating[Any, Any]],
) -> None:
    """Check recovery using the prescribed covariance of the final BLA estimate."""
    assert G_estimated.shape == G_true.shape
    errors = (G_estimated - G_true).swapaxes(-1, -2).reshape(G_estimated.shape[0], -1)
    variances = np.diagonal(expected_covariance, axis1=-2, axis2=-1).real
    tolerance = 6 * np.sqrt(np.maximum(variances, 0)) + 1e-12
    np.testing.assert_array_less(np.abs(errors), tolerance)


def _assert_covariance(
    estimated: NDArray[np.complexfloating[Any, Any]],
    expected: NDArray[np.complexfloating[Any, Any]],
    n_degrees_of_freedom: int,
) -> None:
    """Check all complex entries, allowing for finite-sample covariance scatter."""
    assert estimated.shape == expected.shape
    np.testing.assert_allclose(estimated, estimated.conj().mT, atol=1e-24)
    variances = np.maximum(np.diagonal(expected, axis1=-2, axis2=-1).real, 0)
    entry_scales = np.sqrt(variances[:, :, None] * variances[:, None, :])
    tolerance = 6 * entry_scales / np.sqrt(n_degrees_of_freedom) + 1e-24
    np.testing.assert_array_less(np.abs(estimated - expected), tolerance)
