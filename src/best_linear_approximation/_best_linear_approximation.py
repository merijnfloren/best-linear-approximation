from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    propagate_covariance,
)
from best_linear_approximation._frequency_response import compute_frequency_response
from best_linear_approximation._linear_algebra import kronecker_product, vec
from best_linear_approximation._typing import (
    ExcitedBins,
    TimeDomainSignal,
)


def compute_best_linear_approximation_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    """Compute the best linear approximation and its covariances from known input data."""
    n_experiments, n_periods = y.shape[-2:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_batched_matrices = as_batched_matrices(u)  # (n_samples, n_experiments, 1, nu, nu)
    y_batched_matrices = as_batched_matrices(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U = np.fft.rfft(u_batched_matrices, axis=0)[excited_bins]
    Y = np.fft.rfft(y_batched_matrices, axis=0)[excited_bins]

    # Frequency response: (n_excited_bins, n_experiments, n_periods, ny, nu)
    G_per_experiment_period = compute_frequency_response(U, Y)

    # Average over periods: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = np.mean(G_per_experiment_period, axis=2)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)

    # BLA total covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(
            vec(G_per_experiment),  # (n_excited_bins, n_experiments, ny * nu)
        ) / n_experiments
    else:
        G_cov_total = None

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_periods > 1:
        G_cov_noise = np.mean(
            compute_sample_covariance(
                vec(G_per_experiment_period),  # (n_excited_bins, n_experiments, n_periods, ny * nu)
            ) / n_periods,
            axis=1,
        ) / n_experiments
    else:
        G_cov_noise = None

    return G, G_cov_total, G_cov_noise


def compute_best_linear_approximation_noisy_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    """Compute the best linear approximation and its covariances from noisy input data."""
    ny, nu, n_experiments, n_periods = y.shape[-4:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_batched_matrices = as_batched_matrices(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_batched_matrices = as_batched_matrices(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U = np.fft.rfft(u_batched_matrices, axis=0)[excited_bins]
    Y = np.fft.rfft(y_batched_matrices, axis=0)[excited_bins]

    # Data noise covariance: (n_excited_bins, n_experiments, (ny + nu) * nu, (ny + nu) * nu)
    if n_periods > 1:
        Z = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )

        cov_Z_noise = compute_sample_covariance(
            vec(Z),  # (n_excited_bins, n_experiments, n_periods, (ny + nu) * nu)
        ) / n_periods
    else:
        cov_Z_noise = None

    # Proceed with the period sample means
    U = np.mean(U, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y = np.mean(Y, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Frequency response: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = compute_frequency_response(U, Y)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)

    # BLA total covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(
            vec(G_per_experiment),  # (n_excited_bins, n_experiments, ny * nu)
        ) / n_experiments
    else:
        G_cov_total = None

    # BLA noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if cov_Z_noise is not None:
        n_excited_bins = G.shape[0]

        # Batched U^{-T}
        U_inv_transpose = np.linalg.solve(
            U, np.eye(nu),
        ).mT if nu > 1 else 1 / U

        # Batched V = [I_ny, -G]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        V = np.concatenate((I_ny, -G_per_experiment), axis=-1)

        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        jacobian = kronecker_product(U_inv_transpose, V)

        G_cov_noise = np.mean(
            propagate_covariance(cov_Z_noise, jacobian),
            axis=1,
        ) / n_experiments
    else:
        G_cov_noise = None

    return G, G_cov_total, G_cov_noise


def compute_best_linear_approximation_indirect(
    r: TimeDomainSignal,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    """Compute the best linear approximation and its covariances from noisy input data."""
    ny, nu, n_experiments, n_periods = y.shape[-4:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    r_batched_matrices = as_batched_matrices(r)  # (n_samples, n_experiments, 1, nu, nu)
    u_batched_matrices = as_batched_matrices(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_batched_matrices = as_batched_matrices(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    R = np.fft.rfft(r_batched_matrices, axis=0)[excited_bins]
    U = np.fft.rfft(u_batched_matrices, axis=0)[excited_bins]
    Y = np.fft.rfft(y_batched_matrices, axis=0)[excited_bins]

    # Data noise covariance: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    if n_periods > 1:
        Z = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )

        cov_Z_R_noise = np.mean(
            compute_sample_covariance(
                vec(Z),  # (n_excited_bins, n_experiments, n_periods, (ny + nu) * nu)
            ) / n_periods,
            axis=1,
        ) / n_experiments
    else:
        cov_Z_R_noise = None

    # Proceed with the period sample means
    U = np.mean(U, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y = np.mean(Y, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Remove singleton reference dimension
    R = np.squeeze(R, axis=2)  # (n_excited_bins, n_experiments, nu, nu)

    # Apply phase correction to the input-output spectra
    phase_correction = R.conj() / np.abs(R)
    U_R = U * phase_correction  # (n_excited_bins, n_experiments, nu, nu)
    Y_R = Y * phase_correction  # (n_excited_bins, n_experiments, ny, nu)

    # Data total covariance: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    if n_experiments > 1:
        Z_R = np.concatenate(  # (n_excited_bins, n_experiments, ny + nu, nu)
            (Y_R, U_R),
            axis=-2,
        )

        cov_Z_R_total = compute_sample_covariance(
            vec(Z_R),  # (n_excited_bins, n_experiments, (ny + nu) * nu)
        ) / n_experiments
    else:
        cov_Z_R_total = None

    # Proceed with the experiment sample means
    U_R = np.mean(U_R, axis=1)  # (n_excited_bins, nu, nu)
    Y_R = np.mean(Y_R, axis=1)  # (n_excited_bins, ny, nu)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = compute_frequency_response(U_R, Y_R)

    # BLA covariances
    G_cov_total = None
    G_cov_noise = None
    if cov_Z_R_total is not None or cov_Z_R_noise is not None:
        n_excited_bins = G.shape[0]

        # Batched U^{-T}
        U_R_inv_transpose = np.linalg.solve(U_R, np.eye(nu)).mT if nu > 1 else 1 / U_R

        # Batched V = [I_ny, -G]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
        V = np.concatenate((I_ny, -G), axis=-1)

        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        jacobian = kronecker_product(U_R_inv_transpose, V)

        # Total covariance: (n_excited_bins, ny * nu, ny * nu)
        if cov_Z_R_total is not None:
            G_cov_total = propagate_covariance(cov_Z_R_total, jacobian)

        # Noise covariance: (n_excited_bins, ny * nu, ny * nu)
        if cov_Z_R_noise is not None:
            G_cov_noise = propagate_covariance(cov_Z_R_noise, jacobian)

    return G, G_cov_total, G_cov_noise
