import numpy as np

from best_linear_approximation._covariance import propagate_covariance
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation._typing import ComplexArray
from best_linear_approximation.robust import known_input, noisy_input

from . import (
    assert_bla_recovery_and_covariances,
    assert_covariance,
    bla_disturbance_cases,
    bla_recovery_excitation_cases,
    bla_recovery_seeds,
    generate_correlation_matrix,
    generate_test_setup,
    sample_disturbances,
    to_time_domain,
)


def test_known_input_mimo_experiment_and_realization_layouts_are_equivalent() -> None:
    """Verify known-input estimation gives the same result for both public layouts."""
    ny = 2
    nu = 2
    n_periods = 2
    setup = generate_test_setup(
        ny,
        nu,
        seed=7,
        orthogonal=True,
        n_experiments=2,
        n_periods=n_periods,
    )
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size
    U = setup.U
    Y = setup.G_true[:, None] @ U
    Y = np.broadcast_to(Y[:, :, None], (n_excited_bins, n_experiments, n_periods, ny, nu))

    u_experiment = to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y_experiment = to_time_domain(Y, n_samples, multisine.freq.excited_bins)

    n_realizations = nu * n_experiments
    u_realization = u_experiment.reshape(n_samples, nu, n_realizations, order="F")
    y_realization = y_experiment.reshape(n_samples, ny, n_realizations, n_periods, order="F")

    bla_experiment = known_input(
        u_experiment,
        y_experiment,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )
    bla_realization = known_input(
        u_realization,
        y_realization,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )

    np.testing.assert_array_equal(bla_experiment.G.value, bla_realization.G.value)
    np.testing.assert_array_equal(bla_experiment.G.total.cov, bla_realization.G.total.cov)


@bla_recovery_excitation_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_known_input_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> None:
    """Verify known-input BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed, orthogonal)
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    U = setup.U
    G_true = setup.G_true
    Y_linear = setup.G_true[:, None] @ U

    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    Y_nonlinear_cov = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_noise_cov = noise_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(Y_nonlinear_cov, rng, n_experiments, 1, nu)
    Y_noise = sample_disturbances(Y_noise_cov, rng, n_experiments, n_periods, nu)

    Y = Y_linear[:, :, None] + Y_nonlinear + Y_noise

    u = to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)

    bla = known_input(u, y, multisine.freq.fs, multisine.freq.excited_bins)

    G_total_cov_expected, G_noise_cov_expected = _compute_oracle_covariances(
        U,
        Y_noise_cov,
        Y_nonlinear_cov,
        n_periods,
        known_input=True,
    )

    assert_bla_recovery_and_covariances(
        bla,
        G_true,
        G_total_cov_expected,
        G_noise_cov_expected,
        n_experiments,
        n_periods,
    )


@bla_recovery_excitation_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_known_input_recovers_output_spectrum_uncertainties(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> None:
    """Verify known-input estimation recovers output spectrum uncertainties."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed, orthogonal)
    multisine = setup.multisine
    n_experiments = setup.n_experiments
    n_periods = setup.n_periods
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    U = setup.U
    Y_linear = setup.G_true[:, None] @ U
    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    Y_nonlinear_cov = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_noise_cov = noise_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(Y_nonlinear_cov, rng, n_experiments, 1, nu)
    Y_noise = sample_disturbances(Y_noise_cov, rng, n_experiments, n_periods, nu)
    Y = Y_linear[:, :, None] + Y_nonlinear + Y_noise

    u = to_time_domain(U[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)
    bla = known_input(u, y, multisine.freq.fs, multisine.freq.excited_bins)

    assert bla.spectra.R is None
    assert bla.spectra.U.total.cov is None
    assert bla.spectra.U.nonlinear.cov is None
    assert bla.spectra.U.noise.cov is None

    Y_spectrum = bla.spectra.Y
    assert Y_spectrum.total.cov is not None
    assert Y_spectrum.nonlinear.cov is not None
    assert Y_spectrum.noise.cov is not None

    n_total_covariance_dof = n_experiments - 1
    n_noise_covariance_dof = n_experiments * nu * (n_periods - 1)
    assert_covariance(
        Y_spectrum.total.cov,
        Y_nonlinear_cov + Y_noise_cov,
        n_total_covariance_dof,
    )
    if nonlinear_std > 0:
        assert_covariance(
            Y_spectrum.nonlinear.cov,
            Y_nonlinear_cov,
            n_total_covariance_dof,
        )
    else:
        nonlinear_eigenvalues = np.linalg.eigvalsh(Y_spectrum.nonlinear.cov)
        np.testing.assert_array_less(-1e-20, nonlinear_eigenvalues)
    assert_covariance(
        Y_spectrum.noise.cov[multisine.freq.excited_bins],
        Y_noise_cov,
        n_noise_covariance_dof,
    )


@bla_recovery_excitation_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_noisy_input_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> None:
    """Verify noisy-input BLA recovery and its total and noise covariances."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed, orthogonal)
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
    Z_nonlinear_cov = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, n_channels)
    Z_noise_cov = noise_var * generate_correlation_matrix(rng, n_excited_bins, n_channels)
    Z_nonlinear = sample_disturbances(Z_nonlinear_cov, rng, n_experiments, 1, nu)
    Z_noise = sample_disturbances(Z_noise_cov, rng, n_experiments, n_periods, nu)
    Y_nonlinear, U_nonlinear = Z_nonlinear[..., :ny, :], Z_nonlinear[..., ny:, :]
    Y_noise, U_noise = Z_noise[..., :ny, :], Z_noise[..., ny:, :]

    U_measured = U[:, :, None] + U_nonlinear + U_noise
    Y = Y_linear[:, :, None] + G_true[:, None, None] @ U_nonlinear
    Y = Y + Y_nonlinear + Y_noise

    u = to_time_domain(U_measured, n_samples, multisine.freq.excited_bins)
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)
    bla = noisy_input(
        u,
        y,
        multisine.freq.fs,
        multisine.freq.excited_bins,
    )

    G_total_cov_expected, G_noise_cov_expected = _compute_oracle_covariances(
        U,
        Z_noise_cov,
        Z_nonlinear_cov,
        n_periods,
        G_true=G_true,
    )

    assert_bla_recovery_and_covariances(
        bla,
        G_true,
        G_total_cov_expected,
        G_noise_cov_expected,
        n_experiments,
        n_periods,
    )


@bla_recovery_excitation_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_noisy_input_recovers_spectrum_uncertainties(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> None:
    """Verify noisy-input estimation recovers residual output spectrum uncertainties."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed, orthogonal)
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
    Z_nonlinear_cov = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, n_channels)
    Z_noise_cov = noise_var * generate_correlation_matrix(rng, n_excited_bins, n_channels)
    Z_nonlinear = sample_disturbances(Z_nonlinear_cov, rng, n_experiments, 1, nu)
    Z_noise = sample_disturbances(Z_noise_cov, rng, n_experiments, n_periods, nu)
    Y_nonlinear, U_nonlinear = Z_nonlinear[..., :ny, :], Z_nonlinear[..., ny:, :]
    Y_noise, U_noise = Z_noise[..., :ny, :], Z_noise[..., ny:, :]

    U_measured = U[:, :, None] + U_nonlinear + U_noise
    Y = Y_linear[:, :, None] + G_true[:, None, None] @ U_nonlinear
    Y = Y + Y_nonlinear + Y_noise

    u = to_time_domain(U_measured, n_samples, multisine.freq.excited_bins)
    y = to_time_domain(Y, n_samples, multisine.freq.excited_bins)
    bla = noisy_input(u, y, multisine.freq.fs, multisine.freq.excited_bins)

    U_spectrum = bla.spectra.U
    assert U_spectrum.total.cov is None
    assert U_spectrum.nonlinear.cov is None
    assert U_spectrum.noise.cov is not None

    Y_spectrum = bla.spectra.Y
    assert Y_spectrum.total.cov is not None
    assert Y_spectrum.nonlinear.cov is not None
    assert Y_spectrum.noise.cov is not None

    Y_nonlinear_cov = Z_nonlinear_cov[:, :ny, :ny]
    Y_noise_cov = Z_noise_cov[:, :ny, :ny]
    U_noise_cov = Z_noise_cov[:, ny:, ny:]
    n_total_covariance_dof = n_experiments - 1
    n_noise_covariance_dof = n_experiments * nu * (n_periods - 1)
    assert_covariance(
        Y_spectrum.total.cov,
        Y_nonlinear_cov + Y_noise_cov,
        n_total_covariance_dof,
    )
    if nonlinear_std > 0:
        assert_covariance(
            Y_spectrum.nonlinear.cov,
            Y_nonlinear_cov,
            n_total_covariance_dof,
        )
    else:
        nonlinear_eigenvalues = np.linalg.eigvalsh(Y_spectrum.nonlinear.cov)
        np.testing.assert_array_less(-1e-20, nonlinear_eigenvalues)

    assert_covariance(
        Y_spectrum.noise.cov[multisine.freq.excited_bins],
        Y_noise_cov,
        n_noise_covariance_dof,
    )
    assert_covariance(
        U_spectrum.noise.cov[multisine.freq.excited_bins],
        U_noise_cov,
        n_noise_covariance_dof,
    )


def _compute_oracle_covariances(
    U: ComplexArray,
    noise_cov: ComplexArray,
    nonlinear_cov: ComplexArray,
    n_periods: int,
    G_true: ComplexArray | None = None,
    *,
    known_input: bool = False,
) -> tuple[
    ComplexArray,
    ComplexArray,
]:
    """Compute expected total and noise BLA covariances.

    Based on Eq. (2-77) in Pintelon, R., and Schoukens, J. (2012).
    *System Identification: A Frequency Domain Approach*, 2nd ed.,
    Wiley-IEEE Press, ISBN 978-0-470-64037-1.
    """
    n_excited_bins, n_experiments, nu, _ = U.shape
    U_inv_transpose = np.linalg.solve(U, np.eye(nu)).mT
    if known_input:
        ny = noise_cov.shape[-1]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        residual_transform = I_ny
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
        nonlinear_cov = nonlinear_transform @ nonlinear_cov @ nonlinear_transform.conj().mT

        G_true_per_experiment = np.broadcast_to(
            G_true[:, None],
            (n_excited_bins, n_experiments, ny, nu),
        )
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        residual_transform = np.concatenate((I_ny, -G_true_per_experiment), axis=-1)

    jacobian = kronecker_product(U_inv_transpose, residual_transform)
    I_nu = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu))

    noise_cov = kronecker_product(I_nu, noise_cov)
    nonlinear_cov = kronecker_product(I_nu, nonlinear_cov)
    G_noise_cov_expected = (
        np.mean(
            propagate_covariance(noise_cov[:, None], jacobian),
            axis=1,
        )
        / n_experiments
        / n_periods
    )
    G_nonlinear_cov_expected = (
        np.mean(
            propagate_covariance(nonlinear_cov[:, None], jacobian),
            axis=1,
        )
        / n_experiments
    )
    G_total_cov_expected = G_nonlinear_cov_expected + G_noise_cov_expected
    return G_total_cov_expected, G_noise_cov_expected
