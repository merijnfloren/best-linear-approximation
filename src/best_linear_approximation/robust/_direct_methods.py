from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments, warn_for_possible_transients
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._dataloader import (
    load_fine_steering_mirror,
    load_parallel_wiener_hammerstein,
)
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
)
from best_linear_approximation._typing import (
    ExcitedBins,
    FrequencyDomainSignal,
    SamplingFrequencyHz,
    TimeDomainSignal,
)


KNOWN_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    "realization": SignalContract(
        ranks=SignalRanks(r=None, u=3, y=4),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2)),
        ),
    ),
    "experiment": SignalContract(
        ranks=SignalRanks(r=None, u=4, y=5),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
}


NOISY_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    "realization": SignalContract(
        ranks=SignalRanks(r=None, u=4, y=4),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
    "experiment": SignalContract(
        ranks=SignalRanks(r=None, u=5, y=5),
        matching_axes=(
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3, 4)),
        ),
    ),
}


def known_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> None:
    u, y, fs, excited_bins = _prepare_arguments_known_input(u, y, fs, excited_bins)
    
    n_samples, ny, nu, n_experiments, n_periods = y.shape
    
    if n_periods > 1:
        warn_for_possible_transients(y, max_bin=excited_bins[-1])


def noisy_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> None:

    u, y, fs, excited_bins = _prepare_arguments_noisy_input(u, y, fs, excited_bins)
    
    n_samples, ny, nu, n_experiments, n_periods = y.shape
    
    if n_periods > 1:
        warn_for_possible_transients(y, max_bin=excited_bins[-1])
        
        
def _prepare_arguments_known_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve all arguments according to :func:`prepare_arguments`."""
    return prepare_arguments(None, u, y, fs, excited_bins, KNOWN_INPUT_CONTRACTS)[1:]


def _prepare_arguments_noisy_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[TimeDomainSignal, TimeDomainSignal, SamplingFrequencyHz, ExcitedBins]:
    """Validate and resolve all arguments according to :func:`prepare_arguments`."""
    return prepare_arguments(None, u, y, fs, excited_bins, NOISY_INPUT_CONTRACTS)[1:]


if __name__ == "__main__":

    data = load_fine_steering_mirror()["train 200mV"]
    noisy_input(data.u, data.y, data.fs)

    data = load_parallel_wiener_hammerstein()["ParWH-amp-0"]
    known_input(np.mean(data.u, axis=-1), data.y, data.fs, data.excited_bins)




