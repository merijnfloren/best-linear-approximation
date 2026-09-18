from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from best_linear_approximation._covariance import propagate_covariance
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation.robust._direct_methods import known_input, noisy_input

from . import (
    assert_bla_recovery_and_covariances,
    bla_disturbance_cases,
    bla_recovery_cases,
    bla_recovery_seeds,
    covariance_availability_cases,
    covariance_unavailable_warning,
    generate_correlation_matrix,
    generate_covariance_availability_signals,
    generate_test_setup,
    sample_disturbances,
    to_time_domain,
)


@covariance_availability_cases
def test_known_input_covariances_require_repetitions(
    n_experiments: int,
    n_periods: int,
) -> None:
    """Verify known-input covariance availability and its warning."""
    r, _, y = generate_covariance_availability_signals(n_experiments, n_periods)
    warning = covariance_unavailable_warning(n_experiments)

    with pytest.warns(warning):
        G, cov_G_total, cov_G_noise = known_input(r, y, fs=8.0, excited_bins=np.array([1]))

    assert G.shape == (1, 1, 1)
    assert cov_G_total is None if n_experiments == 1 else cov_G_noise is None


@covariance_availability_cases
def test_noisy_input_covariances_require_repetitions(
    n_experiments: int,
    n_periods: int,
) -> None:
    """Verify noisy-input covariance availability and its warning."""
    _, u, y = generate_covariance_availability_signals(n_experiments, n_periods)
    warning = covariance_unavailable_warning(n_experiments)

    with pytest.warns(warning):
        G, cov_G_total, cov_G_noise = noisy_input(u, y, fs=8.0, excited_bins=np.array([1]))

    assert G.shape == (1, 1, 1)
    assert cov_G_total is None if n_experiments == 1 else cov_G_noise is None


@bla_recovery_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_known_input_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
) -> None:
    """Verify known-input BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed)
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    U = setup.U
    G_true = setup.G_true
    Y_linear = setup.G_true[:, None] @ U

    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    cov_Y_nonlinear = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    cov_Y_noise = noise_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(cov_Y_nonlinear, rng, n_experiments, 1, nu)
    Y_noise = sample_disturbances(cov_Y_noise, rng, n_experiments, n_periods, nu)

    Y = Y_linear[:, :, None] + Y_nonlinear + Y_noise

    u = to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)

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

    assert_bla_recovery_and_covariances(
        G_estimated,
        G_true,
        cov_G_total_estimated,
        cov_G_noise_estimated,
        cov_G_total_expected,
        cov_G_noise_expected,
        n_experiments,
        n_periods,
    )


@bla_recovery_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_noisy_input_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
) -> None:
    """Verify noisy-input BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed)
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
    cov_Z_nonlinear = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, n_channels)
    cov_Z_noise = noise_var * generate_correlation_matrix(rng, n_excited_bins, n_channels)
    Z_nonlinear = sample_disturbances(cov_Z_nonlinear, rng, n_experiments, 1, nu)
    Z_noise = sample_disturbances(cov_Z_noise, rng, n_experiments, n_periods, nu)
    Y_nonlinear, U_nonlinear = Z_nonlinear[..., :ny, :], Z_nonlinear[..., ny:, :]
    Y_noise, U_noise = Z_noise[..., :ny, :], Z_noise[..., ny:, :]

    U_measured = U[:, :, None] + U_nonlinear + U_noise
    Y = Y_linear[:, :, None] + G_true[:, None, None] @ U_nonlinear
    Y = Y + Y_nonlinear + Y_noise

    u = to_time_domain(U_measured, n_samples, multisine.freq.excited_bins)
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)
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

    assert_bla_recovery_and_covariances(
        G_estimated,
        G_true,
        cov_G_total_estimated,
        cov_G_noise_estimated,
        cov_G_total_expected,
        cov_G_noise_expected,
        n_experiments,
        n_periods,
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
