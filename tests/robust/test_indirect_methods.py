from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from best_linear_approximation._covariance import propagate_covariance
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation.robust._indirect_methods import closed_loop, known_reference

from . import (
    assert_bla_recovery_and_covariances,
    bla_recovery_cases,
    covariance_availability_cases,
    covariance_unavailable_warning,
    generate_correlation_matrix,
    generate_covariance_availability_signals,
    generate_test_setup,
    sample_disturbances,
    to_time_domain,
)


@covariance_availability_cases
def test_closed_loop_covariances_require_repetitions(
    n_experiments: int,
    n_periods: int,
) -> None:
    """Verify known-reference covariance availability and its warning."""
    r, u, y = generate_covariance_availability_signals(n_experiments, n_periods)
    warning = covariance_unavailable_warning(n_experiments)

    with pytest.warns(warning):
        G, cov_G_total, cov_G_noise = closed_loop(
            r,
            u,
            y,
            fs=8.0,
            excited_bins=np.array([1]),
        )

    assert G.shape == (1, 1, 1)
    assert cov_G_total is None if n_experiments == 1 else cov_G_noise is None


def test_closed_loop_is_identical_to_known_reference() -> None:
    """Verify that closed loop returns the known-reference result unchanged."""
    r, u, y = generate_covariance_availability_signals(2, 2)
    G_closed_loop, cov_G_total_closed_loop, cov_G_noise_closed_loop = closed_loop(
        r,
        u,
        y,
        fs=8.0,
        excited_bins=np.array([1]),
    )
    G_known_reference, cov_G_total_known_reference, cov_G_noise_known_reference = known_reference(
        r,
        u,
        y,
        fs=8.0,
        excited_bins=np.array([1]),
    )

    assert cov_G_total_closed_loop is not None
    assert cov_G_noise_closed_loop is not None
    assert cov_G_total_known_reference is not None
    assert cov_G_noise_known_reference is not None
    np.testing.assert_array_equal(G_closed_loop, G_known_reference)
    np.testing.assert_array_equal(cov_G_total_closed_loop, cov_G_total_known_reference)
    np.testing.assert_array_equal(cov_G_noise_closed_loop, cov_G_noise_known_reference)


@bla_recovery_cases
@pytest.mark.parametrize("seed", [7, 19])
@pytest.mark.parametrize(
    ("nonlinear_std", "noise_std"),
    [(0.0, 0.01), (0.02, 0.0), (0.02, 0.01)],
    ids=["noise_only", "nonlinear_only", "mixed"],
)
def test_closed_loop_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
) -> None:
    """Verify closed-loop BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed)
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    R = setup.U
    G_true = setup.G_true

    controller_shape = (n_excited_bins, nu, ny)
    controller = rng.normal(size=controller_shape) + 1j * rng.normal(size=controller_shape)
    loop_gain_norm = np.linalg.svd(controller @ G_true, compute_uv=False)[:, 0]
    controller_scale = 0.25 / np.maximum(loop_gain_norm, 0.25)
    controller = controller * controller_scale[:, None, None]

    # With an outer reference of zero, R is an additive input excitation
    loop_matrix_input = np.eye(nu) + controller @ G_true
    input_sensitivity = np.linalg.solve(loop_matrix_input, np.eye(nu))
    U_linear = input_sensitivity[:, None] @ R
    Y_linear = G_true[:, None] @ U_linear

    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    cov_Y_nonlinear = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    cov_Y_noise = noise_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(cov_Y_nonlinear, rng, n_experiments, 1, nu)
    Y_noise = sample_disturbances(cov_Y_noise, rng, n_experiments, n_periods, nu)

    input_control_sensitivity = input_sensitivity @ controller
    U_nonlinear = -input_control_sensitivity[:, None, None] @ Y_nonlinear
    U_noise = -input_control_sensitivity[:, None, None] @ Y_noise
    U_measured = U_linear[:, :, None] + U_nonlinear + U_noise
    Y =  Y_linear[:, :, None]+ G_true[:, None, None] @ (U_nonlinear + U_noise)
    Y = Y + Y_nonlinear + Y_noise

    r = to_time_domain(R[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    u = to_time_domain(U_measured, n_samples, multisine.freq.excited_bins)
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)

    G_estimated, cov_G_total_estimated, cov_G_noise_estimated = closed_loop(
        r,
        u,
        y,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )

    loop_matrix_output = np.eye(ny) + G_true @ controller
    output_sensitivity = np.linalg.solve(loop_matrix_output, np.eye(ny))

    disturbance_transform = np.concatenate(
        (output_sensitivity, -input_control_sensitivity),
        axis=-2,
    )
    cov_Z_nonlinear = disturbance_transform @ cov_Y_nonlinear @ disturbance_transform.conj().mT
    cov_Z_noise = disturbance_transform @ cov_Y_noise @ disturbance_transform.conj().mT

    U_R = np.mean(U_linear @ R.conj().mT, axis=1)

    cov_G_total_expected, cov_G_noise_expected, = _compute_indirect_oracle_covariances(
        U_R,
        R,
        G_true,
        cov_Z_noise,
        cov_Z_nonlinear,
        n_periods,
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


def _compute_indirect_oracle_covariances(
    U_R: NDArray[np.complexfloating[Any, Any]],
    R: NDArray[np.complexfloating[Any, Any]],
    G_true: NDArray[np.complexfloating[Any, Any]],
    cov_Z_noise: NDArray[np.complexfloating[Any, Any]],
    cov_Z_nonlinear: NDArray[np.complexfloating[Any, Any]],
    n_periods: int,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]],
]:
    """Compute expected total and noise BLA covariances for indirect data."""
    n_excited_bins, n_experiments, nu, _ = R.shape
    ny = G_true.shape[1]
    n_channels = ny + nu

    U_R_inv_transpose = np.linalg.solve(U_R, np.eye(nu)).mT
    I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
    V = np.concatenate((I_ny, -G_true), axis=-1)
    jacobian = kronecker_product(U_R_inv_transpose, V)
    I_nu = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu))

    cov_Z_noise = kronecker_product(I_nu, cov_Z_noise)
    cov_Z_nonlinear = kronecker_product(I_nu, cov_Z_nonlinear)

    # Total covariance is estimated after projecting each recording onto R
    I_channels = np.broadcast_to(
        np.eye(n_channels), (n_excited_bins, n_experiments, n_channels, n_channels),
    )
    reference_transform = kronecker_product(R.conj(), I_channels)

    cov_Z_total = cov_Z_nonlinear + cov_Z_noise / n_periods
    cov_Z_R_total = (
        np.mean(
            propagate_covariance(cov_Z_total[:, None], reference_transform),
            axis=1,
        )
        / n_experiments
    )
    cov_G_total_expected = propagate_covariance(cov_Z_R_total, jacobian)

    # Noise covariance is estimated before projecting each recording onto R
    cov_G_noise_expected = propagate_covariance(cov_Z_noise / n_experiments / n_periods, jacobian)
    return cov_G_total_expected, cov_G_noise_expected
