from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._array_shapes import move_matrix_axes_to_end
from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    propagate_covariance,
)
from best_linear_approximation._frequency_response import compute_frequency_response
from best_linear_approximation._linear_algebra import vec
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
)
from best_linear_approximation._typing import (
    ExcitedBins,
    SamplingFrequencyHz,
    TimeDomainSignal,
)

KNOWN_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=None, u=3, y=4),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2)),),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=None, u=4, y=5),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),),
    ),
}


NOISY_INPUT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=None, u=4, y=4),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=None, u=5, y=5),
        matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3, 4)),),
    ),
}


def known_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> None:
    u, y, fs, excited_bins = _prepare_arguments_known_input(u, y, fs, excited_bins)
    
    return _compute_best_linear_approximation_known_input(u, y, excited_bins)
    

def noisy_input(
    u: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> None:
    u, y, fs, excited_bins = _prepare_arguments_noisy_input(u, y, fs, excited_bins)
    
    return _compute_best_linear_approximation_noisy_input(u, y, excited_bins)


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



def _compute_best_linear_approximation_known_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:  
    """Compute the best linear approximation and its covariances from known input data."""
    
    n_experiments, n_periods = u.shape[-2:]
    
    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_matrix_last = move_matrix_axes_to_end(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_matrix_last = move_matrix_axes_to_end(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U_excited = np.fft.rfft(u_matrix_last, axis=0)[excited_bins]
    Y_excited = np.fft.rfft(y_matrix_last, axis=0)[excited_bins]

    # Frequency response: (n_excited_bins, n_experiments, n_periods, ny, nu)
    G_per_experiment_period = compute_frequency_response(U_excited, Y_excited)
    
    # Average over periods: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = np.mean(G_per_experiment_period, axis=2) 
    
    # Best linear approximation: (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)
    
    # Total covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(
            vec(G_per_experiment)  # (n_excited_bins, n_experiments, ny * nu)
        ) / n_experiments
    else:
        G_cov_total = None

    # Noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_periods > 1:
        G_cov_noise = np.mean(
            compute_sample_covariance(
                vec(G_per_experiment_period)  # (n_excited_bins, n_experiments, n_periods, ny * nu)
            )
            / n_periods,
            axis=1,
        ) / n_experiments
    else:
        G_cov_noise = None

    return G, G_cov_total, G_cov_noise


def _compute_best_linear_approximation_noisy_input(
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    excited_bins: ExcitedBins,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]] | None,
    NDArray[np.complexfloating[Any, Any]] | None,
]:
    """Compute the best linear approximation and its covariances from noisy input data."""
    
    ny, nu, n_experiments, n_periods = y.shape[-4:]
    
    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    u_matrix_last = move_matrix_axes_to_end(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_matrix_last = move_matrix_axes_to_end(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    U_excited = np.fft.rfft(u_matrix_last, axis=0)[excited_bins]
    Y_excited = np.fft.rfft(y_matrix_last, axis=0)[excited_bins]
    
    if n_periods > 1:
        # Stack input-output spectra: (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
        Z_excited = np.concatenate((Y_excited, U_excited), axis=-2)
        
        # Data covariance matrix: (n_excited_bins, n_experiments, (ny + nu) * nu, (ny + nu) * nu)
        cov_Z_noise = compute_sample_covariance(
            vec(Z_excited)  # (n_excited_bins, n_experiments, n_periods, (ny + nu) * nu)
        )
    else:
        cov_Z_noise = None
    
    # Proceed with the period sample means
    U_excited = np.mean(U_excited, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y_excited = np.mean(Y_excited, axis=2)  # (n_excited_bins, n_experiments, ny, nu)
    
    # Frequency response: (n_excited_bins, n_experiments, ny, nu)
    G_per_experiment = compute_frequency_response(U_excited, Y_excited)
    
    # Best linear approximation: (n_excited_bins, ny, nu)
    G = np.mean(G_per_experiment, axis=1)
    
    # Total covariance: (n_excited_bins, ny * nu, ny * nu)
    if n_experiments > 1:
        G_cov_total = compute_sample_covariance(
            vec(G_per_experiment)  # (n_excited_bins, n_experiments, ny * nu)
        ) / n_experiments
    else:
        G_cov_total = None

    # Noise covariance: (n_excited_bins, ny * nu, ny * nu)
    if cov_Z_noise is not None:
        n_excited_bins = G.shape[0]
               
        # Batched U^{-T}
        U_inv_transpose = np.linalg.solve(U_excited, np.eye(nu)).mT if nu > 1 else 1 / U_excited

        # Batched V = [I_ny, -G]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, n_experiments, ny, ny))
        V = np.concatenate((I_ny, -G_per_experiment), axis=-1)

        # Batched Kronecker product between U^{-T} and V
        jacobian = (
            U_inv_transpose[..., :, None, :, None] * V[..., None, :, None, :]
        ).reshape(n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)

        G_cov_noise = np.mean(
            propagate_covariance(  # (n_excited_bins, n_experiments, ny * nu, ny * nu)   
                cov_Z_noise,
                jacobian,
            )
            / n_periods,
            axis=1,
        ) / n_experiments
    else:
        G_cov_noise = None

    return G, G_cov_total, G_cov_noise
    
    
if __name__ == "__main__":
    from best_linear_approximation._dataloader import (
        load_f16,
        load_fine_steering_mirror,
        load_parallel_wiener_hammerstein,
        load_silverbox,
    )
    
    data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]
    # data = load_fine_steering_mirror()["train 300mV"]
    
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    
    import matplotlib.pyplot as plt
    

    def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return 20 * np.log10(np.abs(magnitude))
    
    plt.figure()
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.show()
    
    data = load_silverbox()["train SB multisine"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs)
        
    
    plt.figure()
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.legend()
    plt.show()
    
    data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    
    
    # create 3x1 subplots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 1, 1])), label="cov_total[1, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 2, 2])), label="cov_total[2, 0]")
    plt.plot(to_db(np.sqrt(cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
    plt.legend()
    plt.show()
    
    
    data = load_fine_steering_mirror()["train 300mV"]
    G, cov_total, cov_noise = noisy_input(data.u, data.y, data.fs, data.excited_bins)
    
    # create 3x3 subplots
    plt.figure()
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1)
        plt.plot(to_db(G[:, i, 0]), label=f"G[{i}, 0]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 0]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 0]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 2)
        plt.plot(to_db(G[:, i, 1]), label=f"G[{i}, 1]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 1]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 1]")
        plt.legend()
        plt.subplot(3, 3, i * 3 + 3)
        plt.plot(to_db(G[:, i, 2]), label=f"G[{i}, 2]")
        plt.plot(to_db(np.sqrt(cov_total[:, i, i])), label=f"cov_total[{i}, 2]")
        plt.plot(to_db(np.sqrt(cov_noise[:, i, i])), label=f"cov_noise[{i}, 2]")
        plt.legend()
    plt.show()
    
    
    
    # import matplotlib.pyplot as plt
    
    # # data = load_silverbox()["train SB multisine"]
    # data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]

    # u = np.mean(data.u, axis=-1)
    # G, cov_noise = known_input(u, data.y, data.fs, data.excited_bins)
    
    
    # def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    #     return 20 * np.log10(np.abs(magnitude))
    
    # # plt.figure()
    # # plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    # # plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    # # plt.legend()
    # # plt.show()
    
    # data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    # u = np.mean(data.u, axis=-1)
    # G2, cov2_noise = known_input(u[:,:,[0,1,2]], data.y[:,:,[0,1,2],:], data.fs, data.excited_bins)
    
    # nois
    
    # cond = 0
    # for i in range(G2.shape[0]):
    #     condi = np.linalg.cond(cov2_noise[i, :, :])
    #     # print(condi)
    #     cond += condi
        
    # print("Average condition number:", cond / G2.shape[0])
    
    # # create 3x1 subplots
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(to_db(G2[:, 0, 0]), label="G2[0, 0]")
    # plt.plot(to_db(np.sqrt(cov2_noise[:, 0, 0])), label="cov2_noise[0, 0]")
    # plt.legend()
    # plt.subplot(3, 1, 2)
    # plt.plot(to_db(G2[:, 1, 0]), label="G2[1, 0]")
    # plt.plot(to_db(np.sqrt(cov2_noise[:, 1, 1])), label="cov2_noise[1, 0]")
    # plt.legend()
    # plt.subplot(3, 1, 3)
    # plt.plot(to_db(G2[:, 2, 0]), label="G2[2, 0]")
    # plt.plot(to_db(np.sqrt(cov2_noise[:, 2, 2])), label="cov2_noise[2, 0]")
    # plt.legend()
    # plt.show()
