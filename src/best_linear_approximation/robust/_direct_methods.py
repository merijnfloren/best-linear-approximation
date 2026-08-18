from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._config import TimeDomainSignal
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
    check_zero_sized_axes,
    validate_estimation_requirements,
    validate_signal_contract,
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
    *,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    fs: float | int,
    excited_bins: NDArray[np.int_] | float | None = None,
) -> ContractType:
    return _validate_arguments_known_input(u, y, fs, excited_bins)


def noisy_input(
    *,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    fs: float,
    excited_bins: NDArray[np.int_] | float | None = None,
) -> ContractType:
    return _validate_arguments_noisy_input(u, y, fs, excited_bins)


def _validate_arguments_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    fs: float,
    excited_bins: NDArray[np.int_] | float | None,
) -> ContractType:
    return _validate_arguments(u, y, fs, excited_bins, KNOWN_INPUT_CONTRACTS)


def _validate_arguments_noisy_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    fs: float,
    excited_bins: NDArray[np.int_] | float | None,
) -> ContractType:
    return _validate_arguments(u, y, fs, excited_bins, NOISY_INPUT_CONTRACTS)



from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingFrequency:
    value: float

    def __post_init__(self):
        if not math.isfinite(self.value):
            msg = f"Sampling frequency must be finite, got {self.value}."
            raise ValueError(msg)
        if self.value <= 0:
            msg = f"Sampling frequency must be strictly positive, got {self.value}."
            raise ValueError(msg)


def _validate_arguments(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    fs: float,
    excited_bins: NDArray[np.int_] | float | None,
    direct_contracts: Mapping[ContractType, SignalContract],
) -> ContractType:
    
    # Structural validation
    check_zero_sized_axes([u, y])
    contract_type = validate_signal_contract(None, u, y, direct_contracts)
    validate_estimation_requirements(u, n_periods=y.shape[-1], contract_type=contract_type)
    
    fs_obj = SamplingFrequency(fs)  # Validate sampling frequency
    
    if isinstance(excited_bins, np.ndarray):
        # check if it is a 1D array of strictly possible integers whose size is smaller or equal to u.shape[0]//2+1
        if excited_bins.ndim != 1 or not np.issubdtype(excited_bins.dtype, np.integer):
            msg = "excited_bins must be a 1D array of integers."
            raise ValueError(msg)
        if np.any(excited_bins <= 0):
            msg = "excited_bins must contain strictly positive integers."
            raise ValueError(msg)
        if excited_bins.size > u.shape[0] // 2 + 1:
            msg = (
                f"excited_bins size must be smaller or equal to u.shape[0]//2+1, "
                f"got {excited_bins.size} > {u.shape[0]//2+1}."
            )
            raise ValueError(msg)
    elif isinstance(excited_bins, float):
        if not (0 < excited_bins < 1):
            msg = "excited_bins as a float must be in the range (0, 1)."
            raise ValueError(msg)
    elif excited_bins is not None:
        msg = (
            f"excited_bins must be a 1D array of strictly positive integers, a float in (0, 1), or None, "
            f"got {type(excited_bins).__name__} = {excited_bins}."
        )
        raise TypeError(msg)


    return contract_type




if __name__ == "__main__":
    # Example usage6
    u = np.empty((10, 3, 8, 2))
    y = np.empty((10, 9, 9, 2))
    fs = 100
    
    # fsss = SamplingFrequency(np.nan)
    # print(f"Sampling frequency: {fsss.value}")

    contract_type = noisy_input(u=u, y=y, fs=fs)
    print(f"Contract type for noisy input: {contract_type}")

    # u = np.empty((10, 3, 9))
    # y = np.empty((10, 19, 9, 2))
    # fs = 100.0
    # excited_bins = np.array([1, 2, 3])
    # contract_type = known_input(u=u, y=y, fs=fs, excited_bins=excited_bins)
    # print(f"Contract type for known input: {contract_type}")

    
    
