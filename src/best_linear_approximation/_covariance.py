from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_sample_covariance(array: NDArray[Any]) -> NDArray[Any]:
    """Compute sample covariance matrices.

    ``array`` has shape ``(..., n_samples, n_channels)``. Leading axes are treated
    as batch axes. Returns an array of shape ``(..., n_channels, n_channels)``.
    """
    array = array.mT  # change to shape (..., n_channels, n_samples)
    centered = array - np.mean(array, axis=-1, keepdims=True)
    return (centered @ centered.conj().mT) / (centered.shape[-1] - 1)


def propagate_covariance(covariance: NDArray[Any], jacobian: NDArray[Any]) -> NDArray[Any]:
    """Propagate covariance through a linear transformation.

    ``covariance`` has shape ``(..., n_channels, n_channels)`` and ``jacobian`` has shape
    ``(..., n_transformed_channels, n_channels)``. Leading axes are treated as batch axes.
    For each batch, the covariance is transformed by the Jacobian according to
    ``jacobian @ covariance @ jacobian.conj().T``. Returns an array of shape
    ``(..., n_transformed_channels, n_transformed_channels)``.
    """
    return jacobian @ covariance @ jacobian.conj().mT


def project_onto_positive_semidefinite(covariance: NDArray[Any]) -> NDArray[Any]:
    """Project covariance matrices onto the positive-semidefinite cone.

    The final two axes of ``covariance`` are treated as square matrix axes; leading
    axes are treated as batch axes. The matrices are first Hermitian symmetrized,
    then negative eigenvalues are set to zero.
    """
    hermitian_covariance = (covariance + covariance.conj().mT) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_covariance)
    nonnegative_eigenvalues = np.maximum(eigenvalues, 0)
    return (eigenvectors * nonnegative_eigenvalues[..., None, :]) @ eigenvectors.conj().mT

