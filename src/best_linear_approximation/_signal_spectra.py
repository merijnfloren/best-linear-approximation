
# import numpy as np

# from best_linear_approximation._array_shapes import as_batched_matrices
# from best_linear_approximation._bla import FrequencyDomainUncertainty, InputSpectrum, OutputSpectrum
# from best_linear_approximation._covariance import (
#     compute_sample_covariance,
#     project_onto_positive_semidefinite,
# )
# from best_linear_approximation._typing import (
#     ComplexArray,
#     ExcitedBins,
#     FrequencyDomainSignal,
#     TimeDomainSignal,
# )

# MINIMUM_N_SAMPLES_FOR_COVARIANCE_ESTIMATION = 2


# def compute_spectra_known_input(
#     u: TimeDomainSignal,
#     y: TimeDomainSignal,
#     G_bla: ComplexArray,  # noqa: N803
#     excited_bins: ExcitedBins,
#     *,
#     independent_subexperiments: bool,
# ) -> tuple[InputSpectrum, OutputSpectrum]:
    
#     ny = y.shape[1]
#     n_periods = y.shape[-1]

#     U = FrequencyDomainSignal(np.fft.rfft(u, axis=0))
#     Y = FrequencyDomainSignal(np.fft.rfft(y, axis=0))
#     U_excited = FrequencyDomainSignal(U[excited_bins])
#     Y_excited = FrequencyDomainSignal(Y[excited_bins])

#     input_spectrum = InputSpectrum(
#         value=U,
#         total=FrequencyDomainUncertainty.unavailable(),
#         nonlinear=FrequencyDomainUncertainty.unavailable(),
#         noise=FrequencyDomainUncertainty.unavailable(),
#     )

#     Y_cov_noise = _compute_noise_covariance(Y)
#     Y_cov_total_per_experiment = _compute_output_total_covariance_known_input(
#         U_excited,
#         Y_excited,
#         G_bla,
#         independent_subexperiments=independent_subexperiments,
#     )
    
#     Y_cov_nonlinear: ComplexArray | None = None
#     Y_cov_total: ComplexArray | None = None
#     if Y_cov_total_per_experiment is not None:
#         if Y_cov_noise is None:
#             # Noise covariance is unavailable only when there is one period. In that
#             # case, period averaging does not change the total covariance, so:
#             Y_cov_total = Y_cov_total_per_experiment
#         else:
#             Y_cov_nonlinear = project_onto_positive_semidefinite(
#                 Y_cov_total_per_experiment - Y_cov_noise[excited_bins] / n_periods,
#             )
#             Y_cov_total = Y_cov_nonlinear + Y_cov_noise[excited_bins]
    
#     output_spectrum = OutputSpectrum(
#         value=Y,
#         total=(
#             FrequencyDomainUncertainty.from_cov(Y_cov_total, (ny,))
#             if Y_cov_total is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#         nonlinear=(
#             FrequencyDomainUncertainty.from_cov(Y_cov_nonlinear, (ny,))
#             if Y_cov_nonlinear is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#         noise=(
#             FrequencyDomainUncertainty.from_cov(Y_cov_noise, (ny,))
#             if Y_cov_noise is not None
#             else FrequencyDomainUncertainty.unavailable()
#         ),
#     )
#     return input_spectrum, output_spectrum


# def _compute_noise_covariance(
#     signal: FrequencyDomainSignal,
# ) -> ComplexArray | None:
#     """Compute the averaged noise covariance over periods."""
#     n_periods = signal.shape[-1]
#     if n_periods < MINIMUM_N_SAMPLES_FOR_COVARIANCE_ESTIMATION:
#         return None

#     signal = np.moveaxis(signal, 1, -1)  # (n_bins, nu, n_experiments, n_periods, ny)
#     return np.mean(compute_sample_covariance(signal), axis=(1, 2))


# def _compute_output_total_covariance_known_input(
#     U_excited: FrequencyDomainSignal,  # noqa: N803
#     Y_excited: FrequencyDomainSignal,  # noqa: N803
#     G_bla: ComplexArray,  # noqa: N803
#     *,
#     independent_subexperiments: bool = False,
# ) -> ComplexArray | None:
#     """Estimate the total covariance of the period-averaged output residuals.

#     The total covariance contains both stochastic nonlinear distortions and the
#     measurement noise remaining after period averaging:

#     ``Cov(Y_total) = Cov(Y_nonlinear) + Cov(Y_noise) / n_periods``.

#     If ``independent_subexperiments`` is true, experiments and subexperiments are
#     treated as independent realizations and covariance is estimated jointly across
#     both axes. Otherwise, covariance is estimated across experiments separately
#     for each subexperiment and then averaged over subexperiments.
#     """
#     n_excited_bins = U_excited.shape[0]
#     ny, nu, n_experiments = Y_excited.shape[1:4]
    
#     U_excited = as_batched_matrices(U_excited)
#     Y_excited = as_batched_matrices(Y_excited)

#     n_realizations = n_experiments * nu if independent_subexperiments else n_experiments
#     if n_realizations < MINIMUM_N_SAMPLES_FOR_COVARIANCE_ESTIMATION:
#         return None

#     # Period-averaged output residual: (n_excited_bins, n_experiments, ny, nu)
#     U_per_experiment = np.squeeze(U_excited, axis=2)
#     Y_per_experiment = np.mean(Y_excited, axis=2)
#     Y_residual = Y_per_experiment - G_bla[:, None] @ U_per_experiment

#     if independent_subexperiments:
#         # Estimate covariance jointly across experiments and subexperiments
#         Y_residual = Y_residual.transpose(0, 1, 3, 2).reshape(n_excited_bins, n_realizations, ny)
#         Y_cov_total = compute_sample_covariance(Y_residual)
#     else:
#         # Estimate covariance across experiments for each subexperiment, then average
#         Y_residual = Y_residual.transpose(0, 3, 1, 2)  # (n_excited_bins, nu, n_experiments, ny)
#         Y_cov_total = np.mean(compute_sample_covariance(Y_residual), axis=1)

#     return Y_cov_total
