from __future__ import annotations

from typing import TYPE_CHECKING

from best_linear_approximation._linear_algebra import right_solve

if TYPE_CHECKING:
    from best_linear_approximation._typing import ComplexArray


def compute_frequency_response(
    input_spectrum: ComplexArray,
    output_spectrum: ComplexArray,
) -> ComplexArray:
    """Compute the frequency response at the excited frequencies.

    The input and output spectra have shapes ``(n_excited_bins, ..., nu, nu)``
    and ``(n_excited_bins, ..., ny, nu)``, respectively. The intermediate axes
    must have the same rank or be explicitly padded with singleton axes so that
    they are broadcast-compatible. Returns the frequency response with shape
    ``(n_excited_bins, ..., ny, nu)``.
    """
    return right_solve(output_spectrum, input_spectrum)
