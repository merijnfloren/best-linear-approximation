import numpy as np

from best_linear_approximation._misc import standardize_channels


def test_standardize_channels_reduces_all_axes_except_channels() -> None:
    signal = np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
            [[[5.0, 6.0], [7.0, 8.0]], [[50.0, 60.0], [70.0, 80.0]]],
        ],
    )

    standardized, means, standard_deviations = standardize_channels(signal)

    expected_means = np.array(
        [signal[:, channel, ...].ravel().mean() for channel in range(signal.shape[1])],
    )
    expected_standard_deviations = np.array(
        [signal[:, channel, ...].ravel().std() for channel in range(signal.shape[1])],
    )

    np.testing.assert_allclose(means, expected_means)
    np.testing.assert_allclose(standard_deviations, expected_standard_deviations)
    np.testing.assert_allclose(standardized.mean(axis=(0, 2, 3)), 0, atol=1e-15)
    np.testing.assert_allclose(standardized.std(axis=(0, 2, 3)), 1)

    rescaled_signal = signal.copy()
    rescaled_signal[:, 0, ...] = 3 * rescaled_signal[:, 0, ...] + 7
    rescaled_standardized, _, _ = standardize_channels(rescaled_signal)

    np.testing.assert_allclose(rescaled_standardized, standardized)


def test_standardize_channels_returns_zeros_for_constant_channels() -> None:
    standardized, means, standard_deviations = standardize_channels(np.ones((3, 1, 2)))

    np.testing.assert_array_equal(standardized, 0)
    np.testing.assert_array_equal(means, [1])
    np.testing.assert_array_equal(standard_deviations, [0])
