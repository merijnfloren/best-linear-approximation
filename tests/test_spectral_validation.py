import numpy as np
import pytest

from best_linear_approximation._config import RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._dataloader import (
    DataBLA,
    load_f16,
    load_fine_steering_mirror,
    load_parallel_wiener_hammerstein,
    load_silverbox,
)
from best_linear_approximation._exceptions import NoExcitedBinsError
from best_linear_approximation._spectral_validation import (
    _classify_multisine_type,
    detect_excited_bins,
)

PARALLEL_WIENER_HAMMERSTEIN_MAX_EXCITED_FREQUENCY_HZ = 20_000
F16_FULL_MULTISINE_MIN_EXCITED_FREQUENCY_HZ = 2
F16_FULL_MULTISINE_MAX_EXCITED_FREQUENCY_HZ = 15
F16_SPECIAL_ODD_MULTISINE_MIN_EXCITED_FREQUENCY_HZ = 1
F16_SPECIAL_ODD_MULTISINE_MAX_EXCITED_FREQUENCY_HZ = 60
F16_MULTISINE_FREQUENCY_RELATIVE_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def parallel_wiener_hammerstein_data() -> dict[str, DataBLA]:
    return load_parallel_wiener_hammerstein()


@pytest.fixture(scope="module")
def f16_full_multisine_data() -> dict[str, DataBLA]:
    return {name: data for name, data in load_f16().items() if "FullMSine" in name}


@pytest.fixture(scope="module")
def f16_special_odd_multisine_data() -> dict[str, DataBLA]:
    return {name: data for name, data in load_f16().items() if "SpecialOddMSine" in name}


@pytest.mark.parametrize(
    ("relative_threshold", "expected_bins"),
    [
        (0.25, [7, 15]),
        (0.75, [7]),
    ],
)
def test_detect_excited_bins_respects_non_default_relative_threshold(
    relative_threshold: float,
    expected_bins: list[int],
) -> None:
    n_samples = 128
    spectrum = np.zeros(n_samples // 2 + 1, dtype=complex)
    spectrum[7] = 10
    spectrum[15] = 5
    excitation = np.fft.irfft(spectrum, n=n_samples)[:, np.newaxis]

    excited_bins = detect_excited_bins(
        excitation,
        fs=128,
        relative_threshold=relative_threshold,
    )

    np.testing.assert_array_equal(excited_bins, expected_bins)


def test_detect_excited_bins_excludes_dc_and_nyquist() -> None:
    n_samples = 128
    spectrum = np.zeros(n_samples // 2 + 1, dtype=complex)
    spectrum[0] = 4
    spectrum[7] = 3
    spectrum[-1] = 2.5
    excitation = np.fft.irfft(spectrum, n=n_samples)[:, np.newaxis]

    excited_bins = detect_excited_bins(
        excitation,
        fs=128,
        relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
    )
    np.testing.assert_array_equal(excited_bins, [7])


def test_detect_excited_bins_excludes_bins_just_below_relative_threshold() -> None:
    n_samples = 128
    spectrum = np.zeros(n_samples // 2 + 1, dtype=complex)
    spectrum[7] = 3
    spectrum[15] = 3 * RELATIVE_THRESHOLD_EXCITED_BINS - 1e-6
    excitation = np.fft.irfft(spectrum, n=n_samples)[:, np.newaxis]
    excited_bins = detect_excited_bins(
        excitation,
        fs=128,
        relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
    )
    np.testing.assert_array_equal(excited_bins, [7])


def test_detect_excited_bins_includes_bins_just_above_relative_threshold() -> None:
    n_samples = 128
    spectrum = np.zeros(n_samples // 2 + 1, dtype=complex)
    spectrum[7] = 3
    spectrum[15] = 3 * RELATIVE_THRESHOLD_EXCITED_BINS + 1e-6
    excitation = np.fft.irfft(spectrum, n=n_samples)[:, np.newaxis]
    excited_bins = detect_excited_bins(
        excitation,
        fs=128,
        relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
    )
    np.testing.assert_array_equal(excited_bins, [7, 15])


def test_detect_excited_bins_raises_when_no_bins_are_excited() -> None:
    with pytest.raises(NoExcitedBinsError):
        detect_excited_bins(
            np.ones((64, 2)),
            fs=100,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )


@pytest.mark.parametrize(
    ("excited_bins", "expected_type"),
    [
        (np.array([1, 2, 3, 4]), "full"),
        (np.array([1, 3, 5]), "odd"),
        (np.array([2, 4, 6]), "even"),
        (np.array([1, 5]), "special-odd"),
        (np.array([2, 6]), "special-even"),
        (np.array([1, 2, 4]), "special-mixed"),
    ],
)
def test_classify_multisine_type(
    excited_bins: np.ndarray,
    expected_type: str,
) -> None:
    assert _classify_multisine_type(excited_bins) == expected_type


def test_detect_excited_bins_classifies_f16_full_multisines_as_full(
    f16_full_multisine_data: dict[str, DataBLA],
) -> None:
    assert f16_full_multisine_data
    for data in f16_full_multisine_data.values():
        assert data.r is not None
        detected_bins = detect_excited_bins(
            data.r,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        assert _classify_multisine_type(detected_bins) == "full"


def test_detect_excited_bins_finds_f16_full_multisines_between_2_and_15_hz(
    f16_full_multisine_data: dict[str, DataBLA],
) -> None:
    assert f16_full_multisine_data
    for data in f16_full_multisine_data.values():
        assert data.r is not None
        detected_bins = detect_excited_bins(
            data.r,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        f_min = detected_bins[0] * data.fs / data.u.shape[0]
        f_max = detected_bins[-1] * data.fs / data.u.shape[0]

        assert np.isclose(
            f_min,
            F16_FULL_MULTISINE_MIN_EXCITED_FREQUENCY_HZ,
            rtol=F16_MULTISINE_FREQUENCY_RELATIVE_TOLERANCE,
        )
        assert np.isclose(
            f_max,
            F16_FULL_MULTISINE_MAX_EXCITED_FREQUENCY_HZ,
            rtol=F16_MULTISINE_FREQUENCY_RELATIVE_TOLERANCE,
        )


def test_detect_excited_bins_classifies_f16_special_odd_multisines_as_special_odd(
    f16_special_odd_multisine_data: dict[str, DataBLA],
) -> None:
    assert f16_special_odd_multisine_data
    for data in f16_special_odd_multisine_data.values():
        assert data.r is not None
        detected_bins = detect_excited_bins(
            data.r,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        assert _classify_multisine_type(detected_bins) == "special-odd"


def test_detect_excited_bins_finds_f16_special_odd_multisines_between_1_and_60_hz(
    f16_special_odd_multisine_data: dict[str, DataBLA],
) -> None:
    assert f16_special_odd_multisine_data
    for data in f16_special_odd_multisine_data.values():
        assert data.r is not None
        detected_bins = detect_excited_bins(
            data.r,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        f_min = detected_bins[0] * data.fs / data.u.shape[0]
        f_max = detected_bins[-1] * data.fs / data.u.shape[0]

        assert np.isclose(
            f_min,
            F16_SPECIAL_ODD_MULTISINE_MIN_EXCITED_FREQUENCY_HZ,
            rtol=F16_MULTISINE_FREQUENCY_RELATIVE_TOLERANCE,
        )
        assert np.isclose(
            f_max,
            F16_SPECIAL_ODD_MULTISINE_MAX_EXCITED_FREQUENCY_HZ,
            rtol=F16_MULTISINE_FREQUENCY_RELATIVE_TOLERANCE,
        )


def test_detect_excited_bins_matches_fine_steering_mirror_paper_bins() -> None:
    for data in load_fine_steering_mirror().values():
        detected_bins = detect_excited_bins(
            data.u,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        bins_from_paper = data.excited_bins

        np.testing.assert_array_equal(detected_bins, bins_from_paper)


def test_detect_excited_bins_matches_silverbox_paper_bins() -> None:
    silverbox_data = next(iter(load_silverbox().values()))

    detected_bins = detect_excited_bins(
        silverbox_data.u,
        fs=silverbox_data.fs,
        relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
    )
    bins_from_paper = silverbox_data.excited_bins

    np.testing.assert_array_equal(detected_bins, bins_from_paper)


def test_detect_excited_bins_finds_parallel_wiener_hammerstein_below_20_khz(
    parallel_wiener_hammerstein_data: dict[str, DataBLA],
) -> None:
    for data in parallel_wiener_hammerstein_data.values():
        detected_bins = detect_excited_bins(
            data.u,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        f_max = detected_bins[-1] * data.fs / data.u.shape[0]
        assert f_max <= PARALLEL_WIENER_HAMMERSTEIN_MAX_EXCITED_FREQUENCY_HZ


def test_detect_excited_bins_classifies_parallel_wiener_hammerstein_as_full(
    parallel_wiener_hammerstein_data: dict[str, DataBLA],
) -> None:
    for data in parallel_wiener_hammerstein_data.values():
        detected_bins = detect_excited_bins(
            data.u,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        assert _classify_multisine_type(detected_bins) == "full"


def test_detect_excited_bins_is_consistent_for_parallel_wiener_hammerstein(
    parallel_wiener_hammerstein_data: dict[str, DataBLA],
) -> None:
    first_data = next(iter(parallel_wiener_hammerstein_data.values()))
    first_detected_bins = detect_excited_bins(
        first_data.u,
        fs=first_data.fs,
        relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
    )
    for data in parallel_wiener_hammerstein_data.values():
        detected_bins = detect_excited_bins(
            data.u,
            fs=data.fs,
            relative_threshold=RELATIVE_THRESHOLD_EXCITED_BINS,
        )
        np.testing.assert_array_equal(detected_bins, first_detected_bins)
