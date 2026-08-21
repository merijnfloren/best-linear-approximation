from collections.abc import Callable

import pytest

from best_linear_approximation._dataloader import (
    DataBLA,
    load_f16,
    load_fine_steering_mirror,
    load_parallel_wiener_hammerstein,
    load_silverbox,
)

N_SIGNAL_AXES = 4
N_F16_DATASETS = 4 + 3  # full + special-odd multisines
N_FINE_STEERING_MIRROR_DATASETS = 3
N_PARALLEL_WIENER_HAMMERSTEIN_DATASETS = 5
N_SILVERBOX_DATASETS = 1


@pytest.mark.parametrize(
    "loader",
    [
        load_f16,
        load_fine_steering_mirror,
        load_silverbox,
        load_parallel_wiener_hammerstein,
    ],
)
def test_benchmark_loaders_return_valid_bla_data(
    loader: Callable[[], dict[str, DataBLA]],
) -> None:
    datasets = loader()

    assert datasets
    for data in datasets.values():
        assert data.u.ndim == data.y.ndim == N_SIGNAL_AXES
        assert data.u.shape[0] == data.y.shape[0]
        assert data.u.shape[2:] == data.y.shape[2:]
        assert data.fs > 0
        assert data.excited_bins.size > 0
        assert all(data.excited_bins >= 1)
        assert len(data.excited_bins) <= (data.u.shape[0] - 1) // 2
        assert data.r is None or data.r.shape == data.u.shape


@pytest.mark.parametrize(
    ("loader", "expected_n_datasets"),
    [
        (load_f16, N_F16_DATASETS),
        (load_fine_steering_mirror, N_FINE_STEERING_MIRROR_DATASETS),
        (load_silverbox, N_SILVERBOX_DATASETS),
        (load_parallel_wiener_hammerstein, N_PARALLEL_WIENER_HAMMERSTEIN_DATASETS),
    ],
)
def test_benchmark_loaders_return_expected_number_of_datasets(
    loader: Callable[[], dict[str, DataBLA]],
    expected_n_datasets: int,
) -> None:
    assert len(loader()) == expected_n_datasets

