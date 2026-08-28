from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._array_shapes import move_matrix_axes_to_end
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
    n_experiments, n_periods = u.shape[-2:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_matrix_last = move_matrix_axes_to_end(u)  # (n_samples, n_experiments, 1, nu, nu)
    y_matrix_last = move_matrix_axes_to_end(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U_excited = np.fft.rfft(u_matrix_last, axis=0)[excited_bins]
    Y_excited = np.fft.rfft(y_matrix_last, axis=0)[excited_bins]

    # Frequency response: (n_excited_bins, n_experiments, n_periods, ny, nu)
    G_per_experiment_period = compute_frequency_response(U_excited, Y_excited)

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
    u_matrix_last = move_matrix_axes_to_end(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_matrix_last = move_matrix_axes_to_end(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U_excited = np.fft.rfft(u_matrix_last, axis=0)[excited_bins]
    Y_excited = np.fft.rfft(y_matrix_last, axis=0)[excited_bins]

    # Data noise covariance: (n_excited_bins, n_experiments, (ny + nu) * nu, (ny + nu) * nu)
    if n_periods > 1:
        Z_excited = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y_excited, U_excited),
            axis=-2,
        )

        cov_Z_noise = compute_sample_covariance(
            vec(Z_excited),  # (n_excited_bins, n_experiments, n_periods, (ny + nu) * nu)
        ) / n_periods
    else:
        cov_Z_noise = None

    # Proceed with the period sample means
    U_excited = np.mean(U_excited, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y_excited = np.mean(Y_excited, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Frequency response: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = compute_frequency_response(U_excited, Y_excited)

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
            U_excited, np.eye(nu)
        ).mT if nu > 1 else 1 / U_excited

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
    r_matrix_last = move_matrix_axes_to_end(r)  # (n_samples, n_experiments, 1, nu, nu)
    u_matrix_last = move_matrix_axes_to_end(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_matrix_last = move_matrix_axes_to_end(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    R_excited = np.fft.rfft(r_matrix_last, axis=0)[excited_bins]
    U_excited = np.fft.rfft(u_matrix_last, axis=0)[excited_bins]
    Y_excited = np.fft.rfft(y_matrix_last, axis=0)[excited_bins]

    # Data noise covariance: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    if n_periods > 1:
        Z_excited = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y_excited, U_excited),
            axis=-2,
        )

        cov_Z_R_noise = np.mean(
            compute_sample_covariance(
                vec(Z_excited),  # (n_excited_bins, n_experiments, n_periods, (ny + nu) * nu)
            ) / n_periods,
            axis=1,
        ) / n_experiments
    else:
        cov_Z_R_noise = None

    # Proceed with the period sample means
    U_excited = np.mean(U_excited, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y_excited = np.mean(Y_excited, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Remove singleton reference dimension
    R_excited = np.squeeze(R_excited, axis=2)  # (n_excited_bins, n_experiments, nu, nu)

    # Apply phase correction to the input-output spectra
    phase_correction = R_excited.conj() / np.abs(R_excited)
    U_R_excited = U_excited * phase_correction  # (n_excited_bins, n_experiments, nu, nu)
    Y_R_excited = Y_excited * phase_correction  # (n_excited_bins, n_experiments, ny, nu)

    # Data total covariance: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    if n_experiments > 1:
        Z_R_excited = np.concatenate(  # (n_excited_bins, n_experiments, ny + nu, nu)
            (Y_R_excited, U_R_excited),
            axis=-2,
        )

        cov_Z_R_total = compute_sample_covariance(
            vec(Z_R_excited),  # (n_excited_bins, n_experiments, (ny + nu) * nu)
        ) / n_experiments
    else:
        cov_Z_R_total = None

    # Proceed with the experiment sample means
    U_R_excited = np.mean(U_R_excited, axis=1)  # (n_excited_bins, nu, nu)
    Y_R_excited = np.mean(Y_R_excited, axis=1)  # (n_excited_bins, ny, nu)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = compute_frequency_response(U_R_excited, Y_R_excited)

    # BLA covariances
    G_cov_total = None
    G_cov_noise = None
    if cov_Z_R_total is not None or cov_Z_R_noise is not None:
        n_excited_bins = G.shape[0]

        # Batched U^{-T}
        U_R_inv_transpose = np.linalg.solve(
            U_R_excited, np.eye(nu)
        ).mT if nu > 1 else 1 / U_R_excited

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
