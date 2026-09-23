import numpy as np

from best_linear_approximation import (
    known_input_compute_output_nonlinear_covariance,
    known_input_compute_output_nonlinear_pooled_variance,
)
from best_linear_approximation.robust import known_input
from tests.robust import (
    _assert_covariance,
    generate_correlation_matrix,
    generate_test_setup,
    sample_disturbances,
    to_time_domain,
)


def test_known_input_output_nonlinear_covariance_recovers_full_orthogonal_covariance() -> None:
    """Verify full random orthogonal input recovers the true output covariance."""
    rng = np.random.default_rng(7)
    ny = 2
    nu = 2
    setup = generate_test_setup(
        ny,
        nu,
        seed=19,
        orthogonal=True,
    )
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    input_channel_amplitudes = np.array([0.5, 2.0])
    U = input_channel_amplitudes[None, None, :, None] * setup.U
    G_true = setup.G_true
    Y_linear = G_true[:, None] @ U

    cov_Y_nonlinear = generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(cov_Y_nonlinear, rng, n_experiments, 1, nu)
    Y = Y_linear[:, :, None] + Y_nonlinear
    Y = np.broadcast_to(Y, (n_excited_bins, n_experiments, n_periods, ny, nu))

    u = to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)
    bla = known_input(
        u,
        y,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )
    assert bla.G.nonlinear.cov is not None

    cov_Y_nonlinear_estimate = known_input_compute_output_nonlinear_covariance(
        U,
        bla.G.nonlinear.cov,
    )

    degrees_of_freedom = n_experiments - 1
    _assert_covariance(cov_Y_nonlinear_estimate, cov_Y_nonlinear, degrees_of_freedom)


def test_known_input_pooled_variance_matches_full_orthogonal_covariance_diagonal() -> None:
    """Verify pooled and full-orthogonal output nonlinear variances agree."""
    rng = np.random.default_rng(7)
    ny = 2
    nu = 2
    setup = generate_test_setup(
        ny,
        nu,
        seed=19,
        orthogonal=True,
    )
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    input_channel_amplitudes = np.array([0.5, 2.0])
    U = input_channel_amplitudes[None, None, :, None] * setup.U
    G_true = setup.G_true
    Y_linear = G_true[:, None] @ U

    cov_Y_nonlinear = generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(cov_Y_nonlinear, rng, n_experiments, 1, nu)
    Y = Y_linear[:, :, None] + Y_nonlinear
    Y = np.broadcast_to(Y, (n_excited_bins, n_experiments, n_periods, ny, nu))

    u = to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)
    bla = known_input(
        u,
        y,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )
    assert bla.G.nonlinear.cov is not None

    cov_Y_nonlinear_clean = known_input_compute_output_nonlinear_covariance(
        U,
        bla.G.nonlinear.cov,
    )
    pooled_variance = known_input_compute_output_nonlinear_pooled_variance(
        U,
        Y,
        bla.G.value,
    )

    expected_variance = np.diagonal(cov_Y_nonlinear_clean, axis1=-2, axis2=-1).real
    np.testing.assert_allclose(pooled_variance, expected_variance, rtol=0.01)
