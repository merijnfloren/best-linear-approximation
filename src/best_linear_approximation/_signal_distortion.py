# from typing import cast

# import numpy as np

# from best_linear_approximation._array_shapes import as_batched_matrices
# from best_linear_approximation._bla import SpectralUncertainty
# from best_linear_approximation._covariance import (
#     compute_sample_covariance,
#     project_onto_positive_semidefinite,
# )
# from best_linear_approximation._linear_algebra import solve_left_kronecker_product
# from best_linear_approximation._typing import ComplexArray, ExcitedBins, TimeDomainSignal


# def compute_noise_covariance(
#     signal: TimeDomainSignal,
#     excited_bins: ExcitedBins | None = None,
# ) -> ComplexArray | None:
#     """Compute the averaged covariance over periods.

#     The covariance remains in signal units; it is not scaled for a period mean. When
#     ``excited_bins`` is ``None``, the covariance is computed at every ``rfft`` bin.
#     """
#     n_periods = signal.shape[-1]
#     if n_periods < 2:
#         return None

#     spectrum = np.fft.rfft(as_batched_matrices(signal), axis=0)
#     if excited_bins is not None:
#         spectrum = spectrum[excited_bins]

#     spectrum_by_input_direction = spectrum.transpose(0, 1, 4, 2, 3)
#     covariance = np.mean(compute_sample_covariance(spectrum_by_input_direction), axis=(1, 2))
#     return cast("ComplexArray", covariance)


# def known_input_compute_output_nonlinear_covariance(
#     U: ComplexArray,  # noqa: N803
#     G_cov_nonlinear: ComplexArray | None,  # noqa: N803
# ) -> ComplexArray:
#     """Refer known-input BLA nonlinear uncertainty to the output signal level.

#     Based on Pintelon, R., and Schoukens, J. (2012).
#     *System Identification: A Frequency Domain Approach*, 2nd ed.,
#     Wiley-IEEE Press, ISBN 978-0-470-64037-1. Specifically, Eq. (2-77) is
#     inverted to compute ``cov(Y_S) = V cov(Z_S) V^H`` given the BLA's nonlinear
#     covariance estimate, assuming full random orthogonal multisines (Eq. (3-31)).
#     """
#     n_experiments = U.shape[1]
#     nu = U.shape[-1]

#     # Compute experiment-averaged (U U^H)^(-T)
#     U_gram_per_experiment = U @ U.conj().mT
#     U_gram_inv_transpose_per_experiment = np.linalg.solve(U_gram_per_experiment, np.eye(nu)).mT
#     U_gram_inv_transpose = np.mean(U_gram_inv_transpose_per_experiment, axis=1)

#     cov_Y_nonlinear = solve_left_kronecker_product(
#         U_gram_inv_transpose,
#         n_experiments * G_cov_nonlinear,
#     )
#     return project_onto_positive_semidefinite(cov_Y_nonlinear)


# def known_input_compute_output_nonlinear_covariance_from_signals(
#     u: TimeDomainSignal,
#     y: TimeDomainSignal,
#     G: ComplexArray,  # noqa: N803
#     excited_bins: ExcitedBins,
#     *,
#     independent_subexperiments: bool = False,
# ) -> SpectralUncertainty:
#     """Estimate output nonlinear covariance directly from known-input recordings.

#     Parameters
#     ----------
#     u : TimeDomainSignal
#         Known input recordings with shape
#         ``(n_samples, nu, nu, n_experiments, n_periods)``.
#     y : TimeDomainSignal
#         Output recordings with shape
#         ``(n_samples, ny, nu, n_experiments, n_periods)``.
#     G : ComplexArray
#         Best linear approximation with shape ``(n_excited_bins, ny, nu)``.
#     excited_bins : ExcitedBins
#         Excited non-DC, non-Nyquist ``rfft`` bin indices.
#     independent_subexperiments : bool, default=False
#         Whether input directions within an experiment are independent output-distortion
#         samples. If ``False``, estimates covariance across experiments separately for
#         each input direction, then averages those covariances; this requires at least
#         two experiments. If ``True``, pools experiment and input-direction samples;
#         this requires ``n_experiments * nu > 1``.

#     Returns
#     -------
#     SpectralUncertainty
#         Output nonlinear covariance with shape ``(n_excited_bins, ny, ny)``.

#     Raises
#     ------
#     ValueError
#         If there are insufficient independent samples or fewer than two periods.

#     """
#     ny, nu, n_experiments, n_periods = y.shape[1:]
#     minimum_n_realizations, minimum_n_periods = 2, 2
#     n_realizations = n_experiments * nu if independent_subexperiments else n_experiments
#     if n_realizations < minimum_n_realizations or n_periods < minimum_n_periods:
#         return SpectralUncertainty.unavailable()

#     U = np.fft.rfft(as_batched_matrices(u), axis=0)[excited_bins]
#     Y = np.fft.rfft(as_batched_matrices(y), axis=0)[excited_bins]

#     # The same BLA is removed from every period of every input direction
#     Y_total_per_experiment_and_period = (  # (n_excited_bins, n_experiments, n_periods, ny, nu)
#         Y - G[:, None, None] @ U
#     )
#     Y_total_per_experiment = np.mean(  # (n_excited_bins, n_experiments, ny, nu)
#         Y_total_per_experiment_and_period,
#         axis=2,
#     )

#     if independent_subexperiments:
#         Y_total_samples = Y_total_per_experiment.transpose(0, 1, 3, 2).reshape(
#             Y_total_per_experiment.shape[0],
#             n_realizations,
#             Y_total_per_experiment.shape[2],
#         )
#         cov_Y_total = compute_sample_covariance(Y_total_samples)
#     else:
#         # Estimate one output covariance per input direction across experiments
#         Y_total_by_input_direction = Y_total_per_experiment.transpose(0, 3, 1, 2)
#         cov_Y_total = np.mean(compute_sample_covariance(Y_total_by_input_direction), axis=1)

#     cov_Y_period = compute_noise_covariance(y, excited_bins)

#     cov_Y_nonlinear = project_onto_positive_semidefinite(cov_Y_total - cov_Y_period / n_periods)
#     return SpectralUncertainty.from_cov(cov_Y_nonlinear, ny)


