import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._covariance import propagate_covariance
from best_linear_approximation._exceptions import (
    PossibleExcitationAmplitudeMismatchWarning,
    PossibleTransientWarning,
)
from best_linear_approximation._linear_algebra import kronecker_product
from best_linear_approximation._typing import ComplexArray, RealArray
from best_linear_approximation.robust import closed_loop, known_reference, noisy_input

from . import (
    assert_bla_recovery_and_covariances,
    assert_covariance,
    bla_disturbance_cases,
    bla_recovery_cases,
    bla_recovery_excitation_cases,
    bla_recovery_seeds,
    generate_correlation_matrix,
    generate_covariance_availability_signals,
    generate_test_setup,
    sample_disturbances,
    to_time_domain,
)


def test_closed_loop_is_identical_to_known_reference() -> None:
    """Verify that closed loop returns the known-reference result unchanged."""
    r, u, y = generate_covariance_availability_signals(2, 2)
    bla_closed_loop = closed_loop(
        r,
        u,
        y,
        fs=8.0,
        excited_bins=np.array([1]),
    )
    bla_known_reference = known_reference(
        r,
        u,
        y,
        fs=8.0,
        excited_bins=np.array([1]),
    )

    assert bla_closed_loop.G.total.cov is not None
    assert bla_closed_loop.G.noise.cov is not None
    assert bla_closed_loop.G.nonlinear.cov is not None
    assert bla_known_reference.G.total.cov is not None
    assert bla_known_reference.G.noise.cov is not None
    assert bla_known_reference.G.nonlinear.cov is not None
    np.testing.assert_array_equal(bla_closed_loop.G.value, bla_known_reference.G.value)
    np.testing.assert_array_equal(bla_closed_loop.G.total.cov, bla_known_reference.G.total.cov)
    np.testing.assert_array_equal(bla_closed_loop.G.noise.cov, bla_known_reference.G.noise.cov)
    np.testing.assert_array_equal(
        bla_closed_loop.G.nonlinear.cov,
        bla_known_reference.G.nonlinear.cov,
    )


@bla_recovery_cases
@bla_recovery_seeds
def test_known_reference_reduces_input_measurement_bias(
    ny: int,
    nu: int,
    seed: int,
) -> None:
    """Verify a clean reference reduces open-loop input measurement bias."""
    n_trials = 16
    n_experiments = 512
    n_periods = 4
    input_noise_std = 7.0
    output_noise_std = 0.2
    maximum_relative_noisy_input_bias = 0.1
    maximum_relative_bias_ratio = 0.4
    rng = np.random.default_rng(seed)

    setup = generate_test_setup(
        ny,
        nu,
        seed,
        orthogonal=True,
        n_experiments=n_experiments,
        n_periods=n_periods,
    )
    multisine = setup.multisine
    n_samples = setup.n_samples
    n_excited_bins = multisine.freq.excited_bins.size

    R = setup.U
    G_true = setup.G_true
    actuator = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu)).astype(complex)
    actuator += 0.2 * (rng.normal(size=actuator.shape) + 1j * rng.normal(size=actuator.shape))
    U_true = actuator[:, None] @ R
    Y_true = G_true[:, None] @ U_true

    U_noise_cov = input_noise_std**2 * generate_correlation_matrix(rng, n_excited_bins, nu)
    Y_noise_cov = output_noise_std**2 * generate_correlation_matrix(rng, n_excited_bins, ny)
    r = to_time_domain(R[:, :, None], n_samples, multisine.freq.excited_bins)[..., 0]
    known_reference_errors = []
    noisy_input_errors = []

    for _ in range(n_trials):
        U_noise = sample_disturbances(U_noise_cov, rng, n_experiments, n_periods, nu)
        Y_noise = sample_disturbances(Y_noise_cov, rng, n_experiments, n_periods, nu)
        u = to_time_domain(
            U_true[:, :, None] + U_noise,
            n_samples,
            multisine.freq.excited_bins,
        )
        y = to_time_domain(
            Y_true[:, :, None] + Y_noise,
            n_samples,
            multisine.freq.excited_bins,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PossibleExcitationAmplitudeMismatchWarning)
            warnings.simplefilter("ignore", PossibleTransientWarning)
            known_reference_estimate = known_reference(
                r,
                u,
                y,
                multisine.freq.fs,
                multisine.freq.excited_bins,
            ).G.value
            noisy_input_estimate = noisy_input(
                u,
                y,
                multisine.freq.fs,
                multisine.freq.excited_bins,
            ).G.value
        known_reference_errors.append(known_reference_estimate - G_true)
        noisy_input_errors.append(noisy_input_estimate - G_true)

    true_response_norm = np.linalg.norm(G_true)
    relative_known_reference_bias = (
        np.linalg.norm(np.mean(known_reference_errors, axis=0)) / true_response_norm
    )
    relative_noisy_input_bias = (
        np.linalg.norm(np.mean(noisy_input_errors, axis=0)) / true_response_norm
    )

    assert relative_noisy_input_bias < maximum_relative_noisy_input_bias
    assert relative_known_reference_bias < maximum_relative_bias_ratio * relative_noisy_input_bias


@bla_recovery_excitation_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_closed_loop_recovers_plant_and_propagated_covariances(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> None:
    """Verify closed-loop BLA recovery and its total and noise covariances."""
    test_data = _generate_closed_loop_test_data(ny, nu, seed, nonlinear_std, noise_std, orthogonal)
    bla = closed_loop(
        test_data.r,
        test_data.u,
        test_data.y,
        test_data.fs,
        test_data.excited_bins,
    )
    assert_bla_recovery_and_covariances(
        bla,
        test_data.G_true,
        test_data.G_total_cov,
        test_data.G_noise_cov,
        test_data.n_experiments,
        test_data.n_periods,
    )


@bla_recovery_excitation_cases
@bla_recovery_seeds
@bla_disturbance_cases
def test_closed_loop_recovers_spectrum_uncertainties(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> None:
    """Verify closed-loop recovery of reference, input, and output spectra."""
    test_data = _generate_closed_loop_test_data(ny, nu, seed, nonlinear_std, noise_std, orthogonal)
    bla = closed_loop(
        test_data.r,
        test_data.u,
        test_data.y,
        test_data.fs,
        test_data.excited_bins,
    )

    R_spectrum = bla.spectra.R
    assert R_spectrum is not None
    assert R_spectrum.noise.cov is None
    assert R_spectrum.nonlinear.cov is None
    assert R_spectrum.total.cov is None

    U_spectrum = bla.spectra.U
    Y_spectrum = bla.spectra.Y
    assert U_spectrum.noise.cov is not None
    assert U_spectrum.nonlinear.cov is not None
    assert U_spectrum.total.cov is not None
    assert Y_spectrum.noise.cov is not None
    assert Y_spectrum.nonlinear.cov is not None
    assert Y_spectrum.total.cov is not None
    assert Y_spectrum.total_equation_error.cov is not None

    n_total_covariance_dof = test_data.n_experiments - 1
    n_noise_covariance_dof = test_data.n_experiments * nu * (test_data.n_periods - 1)

    assert_covariance(
        U_spectrum.noise.cov[test_data.excited_bins],
        test_data.U_noise_cov,
        n_noise_covariance_dof,
    )
    assert_covariance(
        Y_spectrum.noise.cov[test_data.excited_bins],
        test_data.Y_noise_cov,
        n_noise_covariance_dof,
    )

    if nonlinear_std > 0:
        assert_covariance(
            U_spectrum.nonlinear.cov,
            test_data.U_nonlinear_cov,
            n_total_covariance_dof,
        )
        assert_covariance(
            Y_spectrum.nonlinear.cov,
            test_data.Y_nonlinear_cov,
            n_total_covariance_dof,
        )
    else:
        U_nonlinear_eigenvalues = np.linalg.eigvalsh(U_spectrum.nonlinear.cov)
        Y_nonlinear_eigenvalues = np.linalg.eigvalsh(Y_spectrum.nonlinear.cov)
        np.testing.assert_array_less(-1e-20, U_nonlinear_eigenvalues)
        np.testing.assert_array_less(-1e-20, Y_nonlinear_eigenvalues)

    assert_covariance(
        U_spectrum.total.cov,
        test_data.U_nonlinear_cov + test_data.U_noise_cov,
        n_total_covariance_dof,
    )
    assert_covariance(
        Y_spectrum.total.cov,
        test_data.Y_nonlinear_cov + test_data.Y_noise_cov,
        n_total_covariance_dof,
    )


def _compute_indirect_oracle_covariances(
    U_R: ComplexArray,
    R: ComplexArray,
    G_true: ComplexArray,
    Z_noise_cov: ComplexArray,
    Z_nonlinear_cov: ComplexArray,
    n_periods: int,
) -> tuple[
    ComplexArray,
    ComplexArray,
]:
    """Compute expected total and noise BLA covariances for indirect data.

    Based on Eq. (2-77) in Pintelon, R., and Schoukens, J. (2012).
    *System Identification: A Frequency Domain Approach*, 2nd ed.,
    Wiley-IEEE Press, ISBN 978-0-470-64037-1.
    """
    n_excited_bins, n_experiments, nu, _ = R.shape
    ny = G_true.shape[1]
    n_channels = ny + nu

    U_R_inv_transpose = np.linalg.solve(U_R, np.eye(nu)).mT
    I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
    residual_transform = np.concatenate((I_ny, -G_true), axis=-1)
    jacobian = kronecker_product(U_R_inv_transpose, residual_transform)
    I_nu = np.broadcast_to(np.eye(nu), (n_excited_bins, nu, nu))

    Z_noise_cov = kronecker_product(I_nu, Z_noise_cov)
    Z_nonlinear_cov = kronecker_product(I_nu, Z_nonlinear_cov)

    # Total covariance is estimated after projecting each recording onto R
    I_channels = np.broadcast_to(
        np.eye(n_channels), (n_excited_bins, n_experiments, n_channels, n_channels),
    )
    reference_transform = kronecker_product(R.conj(), I_channels)

    Z_total_cov = Z_nonlinear_cov + Z_noise_cov / n_periods
    Z_R_total_cov = (
        np.mean(
            propagate_covariance(Z_total_cov[:, None], reference_transform),
            axis=1,
        )
        / n_experiments
    )
    G_total_cov_expected = propagate_covariance(Z_R_total_cov, jacobian)

    # Noise covariance is projected onto R before averaging over experiments
    Z_R_noise_cov = (
        np.mean(
            propagate_covariance(Z_noise_cov[:, None], reference_transform),
            axis=1,
        )
        / n_experiments
        / n_periods
    )
    G_noise_cov_expected = propagate_covariance(Z_R_noise_cov, jacobian)
    return G_total_cov_expected, G_noise_cov_expected


@dataclass(frozen=True)
class _ClosedLoopTestData:
    r: RealArray
    u: RealArray
    y: RealArray
    fs: float
    excited_bins: NDArray[np.int_]
    G_true: ComplexArray
    G_total_cov: ComplexArray
    G_noise_cov: ComplexArray
    U_nonlinear_cov: ComplexArray
    U_noise_cov: ComplexArray
    Y_nonlinear_cov: ComplexArray
    Y_noise_cov: ComplexArray
    n_experiments: int
    n_periods: int


def _generate_closed_loop_test_data(
    ny: int,
    nu: int,
    seed: int,
    nonlinear_std: float,
    noise_std: float,
    orthogonal: bool,
) -> _ClosedLoopTestData:
    """Generate a closed-loop data set and its spectral covariance oracles."""
    rng = np.random.default_rng(seed)
    setup = generate_test_setup(ny, nu, seed, orthogonal)
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

    loop_matrix_input = np.eye(nu) + controller @ G_true
    input_sensitivity = np.linalg.solve(loop_matrix_input, np.eye(nu))
    U_linear = input_sensitivity[:, None] @ R
    Y_linear = G_true[:, None] @ U_linear

    nonlinear_var, noise_var = nonlinear_std**2, noise_std**2
    Y_nonlinear_cov = nonlinear_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_noise_cov = noise_var * generate_correlation_matrix(rng, n_excited_bins, ny)
    Y_nonlinear = sample_disturbances(Y_nonlinear_cov, rng, n_experiments, 1, nu)
    Y_noise = sample_disturbances(Y_noise_cov, rng, n_experiments, n_periods, nu)

    # Generate closed-loop-correlated input and output disturbances
    input_control_sensitivity = input_sensitivity @ controller
    U_nonlinear = -input_control_sensitivity[:, None, None] @ Y_nonlinear
    U_noise = -input_control_sensitivity[:, None, None] @ Y_noise
    U_measured = U_linear[:, :, None] + U_nonlinear + U_noise
    Y = Y_linear[:, :, None] + G_true[:, None, None] @ (U_nonlinear + U_noise)
    Y = Y + Y_nonlinear + Y_noise

    loop_matrix_output = np.eye(ny) + G_true @ controller
    output_sensitivity = np.linalg.solve(loop_matrix_output, np.eye(ny))
    disturbance_transform = np.concatenate(
        (output_sensitivity, -input_control_sensitivity),
        axis=-2,
    )
    Z_nonlinear_cov = disturbance_transform @ Y_nonlinear_cov @ disturbance_transform.conj().mT
    Z_noise_cov = disturbance_transform @ Y_noise_cov @ disturbance_transform.conj().mT
    U_R = np.mean(U_linear @ R.conj().mT, axis=1)
    G_total_cov, G_noise_cov = _compute_indirect_oracle_covariances(
        U_R,
        R,
        G_true,
        Z_noise_cov,
        Z_nonlinear_cov,
        n_periods,
    )

    excited_bins = multisine.freq.excited_bins
    r = to_time_domain(R[:, :, None], n_samples, excited_bins)[..., 0]
    u = to_time_domain(U_measured, n_samples, excited_bins)
    y = to_time_domain(Y, n_samples, excited_bins)
    return _ClosedLoopTestData(
        r=r,
        u=u,
        y=y,
        fs=multisine.freq.fs,
        excited_bins=excited_bins,
        G_true=G_true,
        G_total_cov=G_total_cov,
        G_noise_cov=G_noise_cov,
        U_nonlinear_cov=Z_nonlinear_cov[:, ny:, ny:],
        U_noise_cov=Z_noise_cov[:, ny:, ny:],
        Y_nonlinear_cov=Y_nonlinear_cov,
        Y_noise_cov=Z_noise_cov[:, :ny, :ny],
        n_experiments=n_experiments,
        n_periods=n_periods,
    )
