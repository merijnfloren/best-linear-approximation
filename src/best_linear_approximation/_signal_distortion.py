"""Signal-level nonlinear-distortion calculations."""

import numpy as np

from best_linear_approximation._covariance import (
    compute_sample_covariance,
    project_onto_positive_semidefinite,
)
from best_linear_approximation._linear_algebra import solve_left_kronecker_product
from best_linear_approximation._typing import ComplexArray, RealArray


def known_input_compute_output_nonlinear_covariance(
    U: ComplexArray,  # noqa: N803
    G_cov_nonlinear: ComplexArray | None,  # noqa: N803
) -> ComplexArray:
    """Refer known-input BLA nonlinear uncertainty to the output signal level.

    Based on Pintelon, R., and Schoukens, J. (2012).
    *System Identification: A Frequency Domain Approach*, 2nd ed.,
    Wiley-IEEE Press, ISBN 978-0-470-64037-1. Specifically, Eq. (2-77) is
    inverted to compute ``cov(Y_S) = V cov(Z_S) V^H`` given the BLA's nonlinear
    covariance estimate, assuming full random orthogonal multisines (Eq. (3-31)).
    """
    n_experiments = U.shape[1]
    nu = U.shape[-1]

    # Compute experiment-averaged (U U^H)^(-T)
    U_gram_per_experiment = U @ U.conj().mT
    U_gram_inv_transpose_per_experiment = np.linalg.solve(U_gram_per_experiment, np.eye(nu)).mT
    U_gram_inv_transpose = np.mean(U_gram_inv_transpose_per_experiment, axis=1)

    cov_Y_nonlinear = solve_left_kronecker_product(
        U_gram_inv_transpose,
        n_experiments * G_cov_nonlinear,
    )
    return project_onto_positive_semidefinite(cov_Y_nonlinear)


def known_input_compute_output_nonlinear_pooled_variance(
    U: ComplexArray,  # noqa: N803
    Y: ComplexArray,  # noqa: N803
    G: ComplexArray,  # noqa: N803
) -> RealArray:
    """Compute pooled output nonlinear variance for a known input.

    Estimates the nonlinear output residual of every experiment and input
    direction directly from ``Y - G U``. It subtracts the output-noise
    covariance estimated from period repetitions, then averages marginal
    nonlinear variances over input directions. Full random orthogonal
    multisines are not required.

    At least two experiments and two periods are required to separate the
    nonlinear and output-noise variances.
    """
    n_experiments = U.shape[1]
    n_periods = Y.shape[2]
    minimum_repetitions = 2
    if n_experiments < minimum_repetitions:
        msg = "At least two experiments are required to estimate nonlinear variance."
        raise ValueError(msg)

    if n_periods < minimum_repetitions:
        msg = "At least two periods are required to separate nonlinear and noise variance."
        raise ValueError(msg)

    # The same BLA is removed from every period of each realization
    Y_nonlinear_and_noise = Y - G[:, None, None] @ U[:, :, None]
    Y_total = np.mean(Y_nonlinear_and_noise, axis=2)

    # Estimate one output covariance per input direction across experiments
    Y_total_by_input_direction = Y_total.transpose(0, 3, 1, 2)
    cov_Y_total = compute_sample_covariance(Y_total_by_input_direction)

    # Period variations estimate the output-noise covariance of the period mean
    Y_by_input_direction = Y_nonlinear_and_noise.transpose(0, 1, 4, 2, 3)
    cov_Y_noise = np.mean(compute_sample_covariance(Y_by_input_direction), axis=1)
    cov_Y_noise = cov_Y_noise / n_periods

    cov_Y_nonlinear = project_onto_positive_semidefinite(cov_Y_total - cov_Y_noise)

    return np.mean(
        np.diagonal(cov_Y_nonlinear, axis1=-2, axis2=-1).real,
        axis=1,
    )
