import numpy as np

from best_linear_approximation.robust._direct_methods import noisy_input


def test_noisy_input_recovers_siso_bla_and_covariances() -> None:
    rng = np.random.default_rng(0)
    n_samples = 128
    n_experiments = 1000
    n_periods = 10
    n_positive_freqs = n_samples // 2 + 1
    excited_bins = np.arange(1, n_positive_freqs - 1)
    n_excited_freqs = excited_bins.size

    state_matrix = rng.uniform(-0.8, 0.8)
    input_matrix = rng.normal(scale=0.2)
    output_matrix = rng.normal(scale=0.2)
    feedthrough_matrix = 2 + rng.normal(scale=0.1)
    angular_frequencies = 2 * np.pi * excited_bins / n_samples
    g_bla = feedthrough_matrix + output_matrix * input_matrix / (
        np.exp(1j * angular_frequencies) - state_matrix
    )

    nonlinear_variance = rng.uniform(1e-3, 3e-3, size=n_excited_freqs)
    measurement_variance = rng.uniform(5e-5, 2e-4, size=n_excited_freqs)
    nonlinear_noise = np.sqrt(nonlinear_variance[:, None]) * (
        rng.normal(size=(n_excited_freqs, n_experiments))
        + 1j * rng.normal(size=(n_excited_freqs, n_experiments))
    ) / np.sqrt(2)
    measurement_noise = np.sqrt(measurement_variance[:, None, None]) * (
        rng.normal(size=(n_excited_freqs, n_experiments, n_periods))
        + 1j * rng.normal(size=(n_excited_freqs, n_experiments, n_periods))
    ) / np.sqrt(2)

    input_spectrum = np.zeros(
        (n_positive_freqs, 1, 1, n_experiments, n_periods),
        dtype=complex,
    )
    input_spectrum[excited_bins, 0, 0] = np.exp(
        2j * np.pi * rng.random((n_excited_freqs, n_experiments)),
    )[:, :, None]
    u = np.fft.irfft(input_spectrum, n=n_samples, axis=0)

    output_spectrum = np.zeros(
        (n_positive_freqs, 1, 1, n_experiments, n_periods),
        dtype=complex,
    )
    g_per_experiment_period = (
        g_bla[:, None, None] + nonlinear_noise[:, :, None] + measurement_noise
    )
    output_spectrum[excited_bins, 0, 0] = (
        g_per_experiment_period * input_spectrum[excited_bins, 0, 0]
    )
    y = np.fft.irfft(output_spectrum, n=n_samples, axis=0)

    estimated_g_bla, estimated_total_covariance, estimated_noise_covariance = noisy_input(
        u,
        y,
        fs=1.0,
        excited_bins=excited_bins,
    )

    expected_total_covariance = (
        nonlinear_variance + measurement_variance / n_periods
    ) / n_experiments
    expected_noise_covariance = measurement_variance / (n_periods * n_experiments)

    np.testing.assert_allclose(estimated_g_bla[:, 0, 0], g_bla, rtol=1e-2)
    assert estimated_total_covariance is not None
    np.testing.assert_allclose(
        estimated_total_covariance[:, 0, 0],
        expected_total_covariance,
        rtol=1e-1,
    )
    assert estimated_noise_covariance is not None
    np.testing.assert_allclose(
        estimated_noise_covariance[:, 0, 0],
        expected_noise_covariance,
        rtol=1e-1,
    )
