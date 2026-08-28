import numpy as np

from best_linear_approximation._covariance import compute_sample_covariance


def test_compute_sample_covariance_matches_known_complex_covariance() -> None:
    rng = np.random.default_rng(0)
    n_samples = 5000
    n_channels = 3
    mixing_matrix = rng.normal(size=(n_channels, n_channels)) + 1j * rng.normal(
        size=(n_channels, n_channels),
    )
    known_covariance = mixing_matrix @ mixing_matrix.conj().T
    standard_complex_samples = (
        rng.normal(size=(n_samples, n_channels))
        + 1j * rng.normal(size=(n_samples, n_channels))
    ) / np.sqrt(2)
    mean = rng.normal(size=n_channels) + 1j * rng.normal(size=n_channels)
    samples = standard_complex_samples @ mixing_matrix.T + mean

    sample_covariance = compute_sample_covariance(samples)

    np.testing.assert_allclose(sample_covariance, known_covariance, rtol=3e-2, atol=3e-2)
