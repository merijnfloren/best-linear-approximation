from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, cast

import numpy as np

from best_linear_approximation._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS
from best_linear_approximation._spectral_validation import detect_excited_bins
from best_linear_approximation._typing import RealArray

if TYPE_CHECKING:
    from best_linear_approximation._typing import ExcitedBins, SamplingFrequencyHz


class DataBLA[ReferenceSignal: RealArray | None](NamedTuple):
    r: ReferenceSignal
    u: RealArray
    y: RealArray
    fs: SamplingFrequencyHz
    excited_bins: ExcitedBins


def load_f16(*, return_transients: bool = False) -> dict[str, DataBLA[RealArray]]:
    """Load selected F16 training datasets.

    This function downloads the F16 benchmark data if it is not already cached
    locally. The downloaded data is stored in a working directory named
    ``nonlinear_benchmarks``. The location depends on the operating system:

    - Windows: ``%LOCALAPPDATA%/nonlinear_benchmarks/``
    - Unix-like systems: ``~/.nonlinear_benchmarks/``
    - macOS: ``~/Library/Application Support/nonlinear_benchmarks/``

    Signals are reshaped to ``(n_samples, n_channels, n_realizations, n_periods)``.
    Returns the sampling frequency and detected excited frequency bins for each
    dataset. Reference signals are available.

    Note that this is a custom variant of the ``nonlinear_benchmarks`` F16 loader that
    returns the full multisine and special-odd multisine datasets while excluding the
    sine-sweep and all validation datasets. Unlike the original loader, it also includes
    the reference signal ``r`` for each dataset.

    Parameters
    ----------
    return_transients : bool, optional
        Whether to return the transient periods of each dataset. Default is ``False``.

    Returns
    -------
    dict[str, DataBLA]
        Datasets keyed by name. Each value contains signals, sampling frequency,
        and excited frequency bins.

        The specific keys are:
        - ``F16Data_FullMSine_Level1.mat``
        - ``F16Data_FullMSine_Level3.mat``
        - ``F16Data_FullMSine_Level5.mat``
        - ``F16Data_FullMSine_Level7.mat``
        - ``F16Data_SpecialOddMSine_Level1.mat``
        - ``F16Data_SpecialOddMSine_Level2.mat``
        - ``F16Data_SpecialOddMSine_Level3.mat``

    """
    try:
        from nonlinear_benchmarks.utilities import cashed_download
        from scipy.io import loadmat
    except ImportError as error:
        _raise_missing_benchmark_dependency(error)

    url = "https://data.4tu.nl/file/b6dc643b-ecc6-437c-8a8a-1681650ec3fe/5414dfdc-6e8d-4208-be6e-fa553de9866f"
    download_size = 148455295

    save_dir = cashed_download(
        url,
        "F16",
        zip_name="F16GVT_Files.zip",
        dir_placement=None,
        download_size=download_size,
        force_download=False,
    )
    save_dir = Path(save_dir) / "F16GVT_Files" / "BenchmarkData"

    matfiles = list(Path(save_dir).glob("*.mat"))
    bla_data: dict[str, DataBLA[RealArray]] = {}
    for file in sorted(matfiles):
        name = file.name
        if ("FullMSine" in name or "SpecialOddMSine" in name) and "Validation" not in name:
            out = loadmat(file)
            r = np.array(out["Voltage"])
            u = np.array(out["Force"])
            y = np.array(out["Acceleration"])
            fs = cast("SamplingFrequencyHz", out["Fs"][0, 0])
            nu, ny = 1, 3
            if "FullMSine" in name:
                n_samples, n_realizations, n_periods = 8192, 1, 9

                shape_u = (n_samples, nu, n_realizations, n_periods)
                r = r.reshape(shape_u, order="F")
                u = u.reshape(shape_u, order="F")
            elif "SpecialOddMSine" in name:
                n_samples, n_realizations, n_periods = 16384, 9, 3

                shape_u = (n_samples, n_periods, n_realizations, nu)
                r = r.T.reshape(shape_u, order="F").transpose(0, 3, 2, 1)
                u = u.T.reshape(shape_u, order="F").transpose(0, 3, 2, 1)
            else:
                msg = f"Unexpected dataset name: {name}"
                raise ValueError(msg)

            shape_y = (n_samples, n_periods, n_realizations, ny)
            y = y.T.reshape(shape_y, order="F").transpose(0, 3, 2, 1)

            if not return_transients:
                r = r[:, :, :, 1:]
                u = u[:, :, :, 1:]
                y = y[:, :, :, 1:]

            excited_bins = detect_excited_bins(
                r,  # detect from the cleaner reference signal
                fs,
                DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
                print_summary=False,
            )

            bla_data[name] = DataBLA(r=r, u=u, y=y, fs=fs, excited_bins=excited_bins)

    return bla_data


def load_fine_steering_mirror() -> dict[str, DataBLA[None]]:
    """Load Fine Steering Mirror training datasets.

    This function downloads the Fine Steering Mirror benchmark data if it is not
    already cached locally. The downloaded data is stored in a working directory
    named ``nonlinear_benchmarks``. The location depends on the operating system:

    - Windows: ``%LOCALAPPDATA%/nonlinear_benchmarks/``
    - Unix-like systems: ``~/.nonlinear_benchmarks/``
    - macOS: ``~/Library/Application Support/nonlinear_benchmarks/``

    Signals already have shape ``(n_samples, n_channels, n_realizations, n_periods)``.
    Returns the sampling frequency and excited frequency bins for each dataset.
    Reference signals are unavailable.

    Returns
    -------
    dict[str, DataBLA]
        Datasets keyed by name. Each value contains signals, sampling frequency,
        and excited frequency bins.

        The specific keys are:
        - ``train 100mV``
        - ``train 200mV``
        - ``train 300mV``

    """
    nlb = _load_benchmark_module()

    # Quantities taken from the Fine Steering Mirror paper
    f_max = 3000  # [Hz]
    fs = 6400  # [Hz]
    n_samples = 8192

    excited_bins = np.arange(1, math.ceil(f_max / (fs / n_samples)))

    nlb_data = cast("list[Any]", nlb.FineSteeringMirror()[0])
    bla_data: dict[str, DataBLA[None]] = {}
    for data in nlb_data:
        u, y = data.u, data.y

        bla_data[data.name] = DataBLA(
            r=None,
            u=u,
            y=y,
            fs=cast("SamplingFrequencyHz", fs),
            excited_bins=cast("ExcitedBins", excited_bins),
        )

    return bla_data


def load_parallel_wiener_hammerstein() -> dict[str, DataBLA[None]]:
    """Load Parallel Wiener-Hammerstein training datasets.

    This function downloads the Parallel Wiener-Hammerstein benchmark data if it
    is not already cached locally. The downloaded data is stored in a working directory
    named ``nonlinear_benchmarks``. The location depends on the operating system:

    - Windows: ``%LOCALAPPDATA%/nonlinear_benchmarks/``
    - Unix-like systems: ``~/.nonlinear_benchmarks/``
    - macOS: ``~/Library/Application Support/nonlinear_benchmarks/``

    Signals are grouped by amplitude level and reshaped to
    ``(n_samples, n_channels, n_realizations, n_periods)``. Returns the
    sampling frequency and detected excited frequency bins for each dataset.
    Reference signals are unavailable.

    Returns
    -------
    dict[str, DataBLA]
        Datasets keyed by name. Each value contains signals, sampling frequency,
        and excited frequency bins.

        The specific keys are:
        - ``ParWH-amp-0``
        - ``ParWH-amp-1``
        - ``ParWH-amp-2``
        - ``ParWH-amp-3``
        - ``ParWH-amp-4``

    """
    nlb = _load_benchmark_module()

    nu, ny = 1, 1
    n_samples, n_realizations, n_periods = 16384, 20, 2

    fs = cast("SamplingFrequencyHz", 78_000)
    amplitudes = [0, 1, 2, 3, 4]

    nlb_data = cast("list[Any]", nlb.ParWH()[0])
    bla_data: dict[str, DataBLA[None]] = {}
    for amplitude in amplitudes:
        nlb_data_per_amplitude = [
            data
            for data in nlb_data
            for phase in range(n_realizations)
            if data.name == f"Est-phase-{phase}-amp-{amplitude}"
        ]
        u = (
            np.array(
                [data.u for data in nlb_data_per_amplitude],
            )
            .reshape(n_realizations, nu, n_samples, n_periods)
            .transpose(2, 1, 0, 3)
        )
        y = (
            np.array(
                [data.y for data in nlb_data_per_amplitude],
            )
            .reshape(n_realizations, ny, n_samples, n_periods)
            .transpose(2, 1, 0, 3)
        )

        excited_bins = detect_excited_bins(
            u,
            fs,
            DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
            print_summary=False,
        )

        name = f"ParWH-amp-{amplitude}"
        bla_data[name] = DataBLA(
            r=None,
            u=u,
            y=y,
            fs=fs,
            excited_bins=excited_bins,
        )

    return bla_data


def load_silverbox() -> dict[str, DataBLA[None]]:
    """Load Silverbox training data.

    This function downloads the Silverbox benchmark data if it is not already
    cached locally. The downloaded data is stored in a working directory named
    ``nonlinear_benchmarks``. The location depends on the operating system:

    - Windows: ``%LOCALAPPDATA%/nonlinear_benchmarks/``
    - Unix-like systems: ``~/.nonlinear_benchmarks/``
    - macOS: ``~/Library/Application Support/nonlinear_benchmarks/``

    Signals are split and reshaped to ``(n_samples, n_channels, n_realizations, n_periods)``.
    Returns the sampling frequency and known excited frequency bins.
    Reference signals are unavailable.

    Returns
    -------
    dict[str, DataBLA]
        Datasets keyed by name. Each value contains signals, sampling frequency,
        and excited frequency bins.

        The specific keys are:
        - ``train SB multisine``

    """
    nlb = _load_benchmark_module()
    nlb_data = cast("Any", nlb.Silverbox()[0])
    u, y = nlb_data.u, nlb_data.y

    nu, ny = 1, 1
    n_samples, n_realizations, n_periods = 8192, 6, 1

    # Quantities taken from the Silverbox paper
    n_zero = 100  # number of zero-valued samples separating the blocks
    fs = cast("SamplingFrequencyHz", 1e7 / 2**14)
    excited_bins = cast("ExcitedBins", np.arange(1, 2 * 1342, 2))

    # Discard transient samples
    n_transient_init = 164  # first realization
    n_transient = 400  # subsequent realizations

    u_matrix = np.zeros((n_samples, n_realizations))
    y_matrix = np.zeros((n_samples, n_realizations))
    for k in range(n_realizations):
        if k == 0:
            u = u[n_transient_init:]
            y = y[n_transient_init:]
        else:
            idx = n_zero + n_transient
            u = u[idx:]
            y = y[idx:]

        u_matrix[:, k] = u[:n_samples]
        y_matrix[:, k] = y[:n_samples]

        u = u[n_samples:]
        y = y[n_samples:]

    u = u_matrix.reshape(n_samples, nu, n_realizations, n_periods)
    y = y_matrix.reshape(n_samples, ny, n_realizations, n_periods)

    return {nlb_data.name: DataBLA(r=None, u=u, y=y, fs=fs, excited_bins=excited_bins)}


def _load_benchmark_module() -> Any:
    """Import the optional nonlinear-benchmarks package."""
    try:
        import nonlinear_benchmarks as nlb
    except ImportError as error:
        _raise_missing_benchmark_dependency(error)

    return nlb


def _raise_missing_benchmark_dependency(error: ImportError) -> NoReturn:
    msg = (
        "Benchmark loaders require the optional dependencies. "
        "Install best-linear-approximation[benchmarks]."
    )
    raise ImportError(msg) from error


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    data = load_parallel_wiener_hammerstein()["ParWH-amp-4"]

    y = data.y

    Y = np.fft.rfft(y, axis=0)

    Y_mean_over_periods = np.mean(Y, axis=-1)

    # noise variance
    Y_centered = Y - Y_mean_over_periods[..., None]
    Y_var_noise = np.sum(np.abs(Y_centered) ** 2, axis=-1) / (y.shape[-1] - 1)
    Y_var_noise = np.mean(Y_var_noise, axis=-1)

    # total variance
    Y_abs_mean_over_periods = np.abs(Y_mean_over_periods)
    Y_abs_mean_over_experiments = np.mean(Y_abs_mean_over_periods, axis=-1)
    Y_centered_total = Y_abs_mean_over_periods - Y_abs_mean_over_experiments[..., None]
    Y_var_total = np.sum(np.abs(Y_centered_total) ** 2, axis=-1) / (y.shape[-2] - 1)

    # mean output spectrum
    Y_mean = np.mean(Y_abs_mean_over_experiments, axis=-1)


    # plot everything
    plt.figure()
    plt.semilogy(Y_mean, label="mean output spectrum")
    plt.semilogy(np.sqrt(Y_var_noise), label="noise variance")
    plt.semilogy(np.sqrt(Y_var_total), label="total variance")
    plt.xlabel("frequency bin")
    plt.ylabel("magnitude")
    plt.legend()
    plt.show()
