import warnings

import numpy as np
import pytest

from best_linear_approximation._argument_preparation import warn_for_possible_transients
from best_linear_approximation._dataloader import (
    DataBLA,
    load_f16,
    load_parallel_wiener_hammerstein,
)
from best_linear_approximation._exceptions import PossibleTransientWarning
from best_linear_approximation._typing import TimeDomainSignal


def test_warn_for_possible_transients_low_max_bin_ignores_higher_frequency_changes() -> None:
    n_samples = 16
    time = np.arange(n_samples)
    base_period = np.cos(2 * np.pi * time / n_samples)
    high_frequency_change = np.cos(2 * np.pi * 4 * time / n_samples)
    y = np.stack((base_period, base_period + high_frequency_change), axis=-1)
    y = TimeDomainSignal(y[:, None, None, None, :])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_for_possible_transients(y, max_bin=3)

    with pytest.warns(PossibleTransientWarning):
        warn_for_possible_transients(y, max_bin=4)


def test_warn_for_possible_transients_handles_different_channel_scales() -> None:
    n_samples = 16
    time = np.arange(n_samples)
    period = np.cos(2 * np.pi * 3 * time / n_samples)
    changing_channel = np.stack((1.25 * period, period), axis=-1)
    steady_channel = np.stack((period, period), axis=-1)
    y = np.stack((changing_channel, steady_channel), axis=1)
    scaled_y = y * np.array([1e-3, 1e3])
    scaled_y = TimeDomainSignal(scaled_y[:, :, None, None, :])

    with pytest.warns(PossibleTransientWarning):
        warn_for_possible_transients(scaled_y, max_bin=3)


def test_warn_for_possible_transient_distinguishes_f16_data() -> None:
    def is_flagged(data: DataBLA) -> bool:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            warnings.simplefilter("always")

            warn_for_possible_transients(data.y, max_bin=data.excited_bins[-1])

        return bool(caught)

    transient_flags = [
        is_flagged(data)
        for data in load_f16(return_transients=True).values()
    ]
    steady_state_flags = [
        is_flagged(data)
        for data in load_f16(return_transients=False).values()
    ]

    assert all(transient_flags)
    assert not all(steady_state_flags)


def test_warn_for_possible_transients_accepts_all_parallel_wiener_hammerstein_data() -> None:
    datasets = load_parallel_wiener_hammerstein()
    for data in datasets.values():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_for_possible_transients(data.y, max_bin=data.excited_bins[-1])
