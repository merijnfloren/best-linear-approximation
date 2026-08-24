import warnings

import numpy as np
import pytest

from best_linear_approximation._argument_preparation import (
    _warn_if_excitation_amplitudes_mismatch,
    _warn_if_output_spectra_mismatch,
    prepare_arguments,
)
from best_linear_approximation._array_shapes import to_experiment_layout
from best_linear_approximation._dataloader import (
    DataBLA,
    load_f16,
    load_parallel_wiener_hammerstein,
)
from best_linear_approximation._exceptions import (
    PossibleMultiAmplitudeWarning,
    PossibleTransientWarning,
)
from best_linear_approximation._typing import ExcitedBins, TimeDomainSignal
from best_linear_approximation.robust._direct_methods import KNOWN_INPUT_CONTRACTS
from best_linear_approximation.robust._indirect_methods import INDIRECT_CONTRACTS


def test_prepare_arguments_converts_direct_realization_layout() -> None:
    n_samples = 16
    n_realizations = 4
    time = np.arange(n_samples)
    excitation = np.cos(2 * np.pi * time / n_samples)
    u = np.broadcast_to(excitation[:, None, None], (n_samples, 2, n_realizations)).copy()
    y = np.broadcast_to(excitation[:, None, None, None], (n_samples, 1, n_realizations, 2)).copy()
    excited_bins = np.array([1])
    sampling_frequency = 20.0

    r, prepared_u, prepared_y, fs, prepared_excited_bins = prepare_arguments(
        None, u, y, sampling_frequency, excited_bins, KNOWN_INPUT_CONTRACTS,
    )

    assert r is None
    assert prepared_u.shape == (n_samples, 2, 2, 2, 1)
    assert prepared_y.shape == (n_samples, 1, 2, 2, 2)
    assert fs == sampling_frequency
    np.testing.assert_array_equal(prepared_excited_bins, excited_bins)


def test_prepare_arguments_converts_indirect_realization_layout() -> None:
    n_samples = 16
    n_realizations = 4
    time = np.arange(n_samples)
    reference = np.cos(2 * np.pi * time / n_samples)
    r = np.broadcast_to(reference[:, None, None], (n_samples, 1, n_realizations)).copy()
    u = np.broadcast_to(reference[:, None, None, None], (n_samples, 1, n_realizations, 2)).copy()
    y = np.broadcast_to(reference[:, None, None, None], (n_samples, 2, n_realizations, 2)).copy()
    excited_bins = np.array([1])
    sampling_frequency = 20.0

    prepared_r, prepared_u, prepared_y, fs, prepared_excited_bins = prepare_arguments(
        r, u, y, sampling_frequency, excited_bins, INDIRECT_CONTRACTS,
    )

    assert prepared_r.shape == (n_samples, 1, 1, n_realizations, 1)
    assert prepared_u.shape == (n_samples, 1, 1, n_realizations, 2)
    assert prepared_y.shape == (n_samples, 2, 1, n_realizations, 2)
    assert fs == sampling_frequency
    np.testing.assert_array_equal(prepared_excited_bins, excited_bins)


def test_warn_if_output_spectra_mismatch_low_max_bin_ignores_higher_frequency_changes() -> None:
    n_samples = 16
    time = np.arange(n_samples)
    base_period = np.cos(2 * np.pi * time / n_samples)
    high_frequency_change = np.cos(2 * np.pi * 4 * time / n_samples)
    y = np.stack((base_period, base_period + high_frequency_change), axis=-1)
    y = TimeDomainSignal(y[:, None, None, None, :])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_output_spectra_mismatch(y, max_bin=3)

    with pytest.warns(PossibleTransientWarning):
        _warn_if_output_spectra_mismatch(y, max_bin=4)


def test_warn_if_output_spectra_mismatch_handles_different_channel_scales() -> None:
    n_samples = 16
    time = np.arange(n_samples)
    period = np.cos(2 * np.pi * 3 * time / n_samples)
    changing_channel = np.stack((1.25 * period, period), axis=-1)
    steady_channel = np.stack((period, period), axis=-1)
    y = np.stack((changing_channel, steady_channel), axis=1)
    scaled_y = y * np.array([1e-3, 1e3])
    scaled_y = TimeDomainSignal(scaled_y[:, :, None, None, :])

    with pytest.warns(PossibleTransientWarning):
        _warn_if_output_spectra_mismatch(scaled_y, max_bin=3)


def test_warn_if_output_spectra_mismatch_distinguishes_f16_data() -> None:
    def is_flagged(data: DataBLA) -> bool:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            warnings.simplefilter("always")

            _warn_if_output_spectra_mismatch(data.y, max_bin=data.excited_bins[-1])

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


def test_warn_if_output_spectra_mismatch_accepts_all_wiener_hammerstein_data() -> None:
    datasets = load_parallel_wiener_hammerstein()
    for data in datasets.values():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_output_spectra_mismatch(data.y, max_bin=data.excited_bins[-1])


def test_warn_if_excitation_amplitudes_mismatch_ignores_unexcited_bins() -> None:
    n_samples = 16
    time = np.arange(n_samples)
    base_realization = np.cos(2 * np.pi * time / n_samples)
    high_frequency_change = np.cos(2 * np.pi * 4 * time / n_samples)
    signal = np.stack(
        (base_realization, base_realization + high_frequency_change),
        axis=1,
    )
    signal = TimeDomainSignal(to_experiment_layout(signal[:, None, :], nu=1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_excitation_amplitudes_mismatch(signal, ExcitedBins(np.array([1])))

    with pytest.warns(PossibleMultiAmplitudeWarning):
        _warn_if_excitation_amplitudes_mismatch(signal, ExcitedBins(np.array([4])))


def test_warn_if_excitation_amplitudes_mismatch_accepts_f16_special_odd_data() -> None:
    datasets = load_f16()
    special_odd_data = {
        name: data
        for name, data in datasets.items()
        if "SpecialOddMSine" in name
    }
    for data in special_odd_data.values():
        signal = TimeDomainSignal(to_experiment_layout(np.mean(data.r, axis=-1), nu=1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_excitation_amplitudes_mismatch(signal, data.excited_bins)


def test_warn_if_excitation_amplitudes_mismatch_flags_f16_amplitude_levels() -> None:
    datasets = load_f16()
    level2 = datasets["F16Data_SpecialOddMSine_Level2.mat"]
    level3 = datasets["F16Data_SpecialOddMSine_Level3.mat"]
    reference = np.concatenate((level2.r, level3.r), axis=2)
    signal = TimeDomainSignal(to_experiment_layout(np.mean(reference, axis=-1), nu=1))

    with pytest.warns(PossibleMultiAmplitudeWarning):
        _warn_if_excitation_amplitudes_mismatch(signal, level2.excited_bins)


def test_warn_if_excitation_amplitudes_mismatch_accepts_all_wiener_hammerstein_data() -> None:
    datasets = load_parallel_wiener_hammerstein()
    for data in datasets.values():
        signal = TimeDomainSignal(to_experiment_layout(data.u, nu=1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_excitation_amplitudes_mismatch(signal, data.excited_bins)
