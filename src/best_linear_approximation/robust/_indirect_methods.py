from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from best_linear_approximation._argument_preparation import prepare_arguments
from best_linear_approximation._array_shapes import as_batched_matrices
from best_linear_approximation._bla import (
    FrequencyInfo,
    NonparametricBLA,
    create_bla,
    create_frequency_info,
)
from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._covariance import (
    compute_sample_covariance,
    propagate_covariance,
)
from best_linear_approximation._frequency_response import compute_frequency_response
from best_linear_approximation._linear_algebra import (
    kronecker_product,
    vec,
)
from best_linear_approximation._signal_validation import (
    ContractType,
    MatchingAxes,
    SignalContract,
    SignalRanks,
)
from best_linear_approximation._typing import RealArray, TimeDomainSignal

INDIRECT_CONTRACTS: Mapping[ContractType, SignalContract] = {
    ContractType.REALIZATION: SignalContract(
        ranks=SignalRanks(r=3, u=4, y=4),
        matching_axes=(
            MatchingAxes(signals=("r", "u"), axes=(0, 1, 2)),
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),
        ),
    ),
    ContractType.EXPERIMENT: SignalContract(
        ranks=SignalRanks(r=4, u=5, y=5),
        matching_axes=(
            MatchingAxes(signals=("r", "u"), axes=(0, 1, 2, 3)),
            MatchingAxes(signals=("u", "y"), axes=(0, 2, 3, 4)),
        ),
    ),
}


def known_reference(
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> NonparametricBLA:
    """Estimate a BLA using a known reference as an instrumental variable.

    Parameters
    ----------
    r : RealArray
        Periodic reference measurements in realization layout with shape
        ``(n_samples, nu, n_realizations)``, or experiment layout with shape
        ``(n_samples, nu, nu, n_experiments)``.
    u : RealArray
        Periodic input measurements in realization layout with shape
        ``(n_samples, nu, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, nu, nu, n_experiments, n_periods)``.
    y : RealArray
        Periodic output measurements in realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        Strictly increasing indices of the excited non-DC, non-Nyquist ``rfft``
        bins. A float in ``(0, 1)`` instead selects them automatically from
        the clean reference ``r``: bins whose channel-averaged spectral
        magnitude exceeds this fraction of the maximum magnitude are selected.

    Returns
    -------
    NonparametricBLA
        Frequency response and available total, noise, and nonlinear covariances.

    """
    r, u, y, fs, excited_bins = prepare_arguments(r, u, y, fs, excited_bins, INDIRECT_CONTRACTS)
    freq = create_frequency_info(u.shape[0], fs, excited_bins)
    return _compute_robust_indirect(r, u, y, freq)


def closed_loop(
    r: RealArray,
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> NonparametricBLA:
    """Estimate a closed-loop BLA using a known reference.

    Parameters
    ----------
    r : RealArray
        Periodic reference measurements in realization layout with shape
        ``(n_samples, nu, n_realizations)``, or experiment layout with shape
        ``(n_samples, nu, nu, n_experiments)``.
    u : RealArray
        Periodic input measurements in realization layout with shape
        ``(n_samples, nu, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, nu, nu, n_experiments, n_periods)``.
    y : RealArray
        Periodic output measurements in realization layout with shape
        ``(n_samples, ny, n_realizations, n_periods)``, or experiment layout
        with shape ``(n_samples, ny, nu, n_experiments, n_periods)``.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        Strictly increasing indices of the excited non-DC, non-Nyquist ``rfft``
        bins. A float in ``(0, 1)`` instead selects them automatically from
        the clean reference ``r``: bins whose channel-averaged spectral
        magnitude exceeds this fraction of the maximum magnitude are selected.

    Returns
    -------
    NonparametricBLA
        Frequency response and available total, noise, and nonlinear covariances.

    """
    return known_reference(r, u, y, fs, excited_bins)


def _compute_robust_indirect(
    r: TimeDomainSignal,
    u: TimeDomainSignal,
    y: TimeDomainSignal,
    freq: FrequencyInfo,
) -> NonparametricBLA:
    """Compute the best linear approximation and its covariances from noisy input data."""
    ny, nu, n_experiments, n_periods = y.shape[-4:]

    # Arrange the arrays for NumPy's batched linear algebra broadcasting
    r_batched_matrices = as_batched_matrices(r)  # (n_samples, n_experiments, 1, nu, nu)
    u_batched_matrices = as_batched_matrices(u)  # (n_samples, n_experiments, n_periods, nu, nu)
    y_batched_matrices = as_batched_matrices(y)  # (n_samples, n_experiments, n_periods, ny, nu)

    # To excited frequencies
    R = np.fft.rfft(r_batched_matrices, axis=0)[freq.excited_bins]
    U = np.fft.rfft(u_batched_matrices, axis=0)[freq.excited_bins]
    Y = np.fft.rfft(y_batched_matrices, axis=0)[freq.excited_bins]

    # Data noise covariance: (n_excited_bins, n_experiments, (ny + nu) * nu, (ny + nu) * nu)
    if n_periods > 1:
        Z = np.concatenate(  # (n_excited_bins, n_experiments, n_periods, ny + nu, nu)
            (Y, U),
            axis=-2,
        )
        cov_Z_noise = compute_sample_covariance(vec(Z)) / n_periods
    else:
        cov_Z_noise = None

    # Proceed with the period sample means
    U = np.mean(U, axis=2)  # (n_excited_bins, n_experiments, nu, nu)
    Y = np.mean(Y, axis=2)  # (n_excited_bins, n_experiments, ny, nu)

    # Remove singleton reference dimension
    R = np.squeeze(R, axis=2)  # (n_excited_bins, n_experiments, nu, nu)

    # Project the input-output spectra onto the known reference
    reference_projection = R.conj().mT
    U_R = U @ reference_projection  # (n_excited_bins, n_experiments, nu, nu)
    Y_R = Y @ reference_projection  # (n_excited_bins, n_experiments, ny, nu)

    # Project the per-experiment noise covariance onto the known reference
    n_excited_bins = freq.excited_bins.size
    if cov_Z_noise is not None:
        n_channels = ny + nu
        I_channels = np.broadcast_to(
            np.eye(n_channels), (n_excited_bins, n_experiments, n_channels, n_channels),
        )
        reference_transform = kronecker_product(R.conj(), I_channels)
        cov_Z_R_noise = np.mean(
            propagate_covariance(cov_Z_noise, reference_transform),
            axis=1,
        ) / n_experiments
    else:
        cov_Z_R_noise = None

    # Data total covariance: (n_excited_bins, (ny + nu) * nu, (ny + nu) * nu)
    if n_experiments > 1:
        Z_R = np.concatenate(  # (n_excited_bins, n_experiments, ny + nu, nu)
            (Y_R, U_R),
            axis=-2,
        )
        cov_Z_R_total = compute_sample_covariance(vec(Z_R)) / n_experiments
    else:
        cov_Z_R_total = None

    # Proceed with the experiment sample means
    U_R = np.mean(U_R, axis=1)  # (n_excited_bins, nu, nu)
    Y_R = np.mean(Y_R, axis=1)  # (n_excited_bins, ny, nu)

    # Best linear approximation (BLA): (n_excited_bins, ny, nu)
    G = compute_frequency_response(U_R, Y_R)

    # BLA covariances
    G_cov_total = None
    G_cov_noise = None
    if cov_Z_R_total is not None or cov_Z_R_noise is not None:
        # Batched U^(-T)
        U_R_inv_transpose = np.linalg.solve(U_R, np.eye(nu)).mT if nu > 1 else 1 / U_R

        # Batched V = [I_ny, -G]
        I_ny = np.broadcast_to(np.eye(ny), (n_excited_bins, ny, ny))
        V = np.concatenate((I_ny, -G), axis=-1)

        # Batched Jacobian: (n_excited_bins, n_experiments, ny * nu, (ny + nu) * nu)
        jacobian = kronecker_product(U_R_inv_transpose, V)

        # Total covariance: (n_excited_bins, ny * nu, ny * nu)
        if cov_Z_R_total is not None:
            G_cov_total = propagate_covariance(cov_Z_R_total, jacobian)

        # Noise covariance: (n_excited_bins, ny * nu, ny * nu)
        if cov_Z_R_noise is not None:
            G_cov_noise = propagate_covariance(cov_Z_R_noise, jacobian)

    return create_bla(freq, G, G_cov_total, G_cov_noise)




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from best_linear_approximation._dataloader import load_f16


    def to_db(magnitude: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return 20 * np.log10(np.abs(magnitude))


    data = load_f16()["F16Data_SpecialOddMSine_Level3.mat"]
    r = data.r.mean(axis=-1)
    bla = closed_loop(r, data.u, data.y, data.fs, data.excited_bins)
    G, cov_total, cov_noise = bla.G.value, bla.G.total.cov, bla.G.noise.cov
    
    print(data.u.shape)
    print(bla.G.total.std.shape)

    # create 3x1 subplots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 0, 0])), label="cov_total[0, 0]")
    plt.plot(to_db(bla.G.nonlinear.std[:, 0]), label="std_nonlinear[0, 0]", linestyle="--")
    plt.plot(to_db(np.sqrt(cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 1, 1])), label="cov_total[1, 0]")
    plt.plot(to_db(bla.G.nonlinear.std[:, 1]), label="std_nonlinear[1, 0]", linestyle="--")
    plt.plot(to_db(np.sqrt(cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    plt.plot(to_db(np.sqrt(cov_total[:, 2, 2])), label="cov_total[2, 0]")
    plt.plot(to_db(bla.G.nonlinear.std[:, 2]), label="std_nonlinear[2, 0]", linestyle="--")
    plt.plot(to_db(np.sqrt(cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
    plt.legend()
    plt.show()

    data = load_f16()["F16Data_FullMSine_Level3.mat"]
    r = data.r.mean(axis=-1)
    bla = closed_loop(r, data.u, data.y, data.fs, data.excited_bins)
    G, cov_total, cov_noise = bla.G.value, bla.G.total.cov, bla.G.noise.cov

    # create 3x1 subplots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(to_db(G[:, 0, 0]), label="G[0, 0]")

    plt.plot(to_db(np.sqrt(8*cov_noise[:, 0, 0])), label="cov_noise[0, 0]")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(to_db(G[:, 1, 0]), label="G[1, 0]")
    plt.plot(to_db(np.sqrt(8*cov_noise[:, 1, 1])), label="cov_noise[1, 0]")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(to_db(G[:, 2, 0]), label="G[2, 0]")
    plt.plot(to_db(np.sqrt(8*cov_noise[:, 2, 2])), label="cov_noise[2, 0]")
    plt.legend()
    plt.show()
    
        # create 3x1 subplots
    print(data.y.shape)
    plt.figure()
    plt.plot(np.squeeze(data.y[:, 0, 0, :]), label="G[0, 0]")
    plt.show()
