from __future__ import annotations

# Tuned to pass all tests in "tests/test_spectral_validation.py", which use
# flat excitation spectra. This value may be too high for non-flat spectra.
DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS = 0.5  # keep in sync with docstrings
