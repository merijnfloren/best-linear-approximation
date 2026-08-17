from pathlib import Path

import numpy as np
from nonlinear_benchmarks.utilities import cashed_download
from scipy.io import loadmat


def load_f16_data() -> dict[str, dict[str, np.ndarray | float]]:
    """Load selected F16 benchmark datasets and reshape their signals.

    This is a custom variant of the ``nonlinear_benchmarks`` F16 loader that returns the full
    multisine and special-odd multisine datasets while excluding the sine-sweep and all validation
    datasets. Unlike the original loader, it also includes the reference signal for each dataset.
    All signals are reshaped to the convention used by ``best_linear_approximation``:

    ``(n_samples, n_channels, n_realizations, n_periods)``.

    Returns a dictionary containing the reshaped reference, input, and output signals together
    with the sampling frequency for each dataset.
    """
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
    data = {}
    for file in sorted(matfiles):
        name = file.name
        if ("FullMSine" in name or "SpecialOddMSine" in name) and "Validation" not in name:
            out = loadmat(file)
            r = out["Voltage"]
            u = out["Force"]
            y = out["Acceleration"]
            fs = out["Fs"][0, 0]
            nu, ny = 1, 3
            if "FullMSine" in name:
                n_samples, n_realizations, n_periods = 8192, 1, 9

                u = np.reshape(u, (n_samples, nu, n_realizations, n_periods), order="F")
                r = np.reshape(r, (n_samples, nu, n_realizations, n_periods), order="F")
            elif "SpecialOddMSine" in name:
                n_samples, n_realizations, n_periods = 16384, 9, 3

                shape_u = (n_samples, n_periods, n_realizations, nu)
                u = np.reshape(u.T, shape_u, order="F").transpose(0, 3, 2, 1)
                r = np.reshape(r.T, shape_u, order="F").transpose(0, 3, 2, 1)
            else:
                msg = f"Unsupported F16 dataset: {name}"
                raise ValueError(msg)

            shape_y = (n_samples, n_periods, n_realizations, ny)
            y = np.reshape(y.T, shape_y, order="F").transpose(0, 3, 2, 1)

            data[name] = {"r": r, "u": u, "y": y, "fs": fs}

    return data
