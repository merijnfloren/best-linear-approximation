import numpy as np
import pytest

from best_linear_approximation._array_shapes import add_period_axis, to_experiment_layout
from best_linear_approximation._exceptions import (
    InsufficientExperimentsError,
    RealizationsTruncatedWarning,
)


def test_to_experiment_layout_warns_when_realizations_are_not_divisible_by_nu() -> None:
    realization_layout = np.empty((1, 1, 5, 1))

    with pytest.warns(RealizationsTruncatedWarning):
        to_experiment_layout(realization_layout, nu=2)


def test_to_experiment_layout_uses_column_major_realization_order() -> None:
    realization_layout = np.arange(6, dtype=float).reshape(1, 1, 6, 1)

    experiment_layout = to_experiment_layout(realization_layout, nu=2)

    expected = np.array([[[[[0], [2], [4]], [[1], [3], [5]]]]], dtype=float)
    np.testing.assert_array_equal(experiment_layout, expected)


def test_to_experiment_layout_adds_singleton_period_axis_if_missing() -> None:
    realization_layout = np.empty((1, 1, 2))

    experiment_layout = to_experiment_layout(realization_layout, nu=1)

    assert experiment_layout.shape == (1, 1, 1, 2, 1)


def test_to_experiment_layout_raises_if_nu_is_larger_than_n_realizations() -> None:
    realization_layout = np.empty((1, 1, 2, 1))

    with pytest.raises(InsufficientExperimentsError):
        to_experiment_layout(realization_layout, nu=3)


def test_add_period_axis_adds_singleton_axis_only_when_needed() -> None:
    experiment_layout_without_periods = np.empty((1, 2, 3, 4))
    experiment_layout_with_periods = np.empty((1, 2, 3, 4, 5))

    result_without_periods = add_period_axis(experiment_layout_without_periods)
    result_with_periods = add_period_axis(experiment_layout_with_periods)

    assert result_without_periods.shape == (1, 2, 3, 4, 1)
    assert result_with_periods is experiment_layout_with_periods
