# Exceptions


class ZeroSizedAxisError(ValueError):
    """Raised when a signal array contains an axis with size zero."""


class InvalidSignalRanksError(ValueError):
    """Raised when signal arrays do not have a supported number of dimensions."""


class InvalidSignalAxesError(ValueError):
    """Raised when signal arrays violate required axis-size relationships."""


class InsufficientExperimentsError(ValueError):
    """Raised when the data cannot form an experiment for frequency-response estimation."""


class NonSquareExperimentError(ValueError):
    """Raised when an experiment-layout input is not square in its input axes."""


# Warnings


class RealizationsTruncatedWarning(UserWarning):
    """Warn that trailing realizations will be discarded before estimation."""


class TotalCovarianceUnavailableWarning(UserWarning):
    """Warn that total covariance cannot be estimated from one experiment."""


class NoiseCovarianceUnavailableWarning(UserWarning):
    """Warn that noise covariance cannot be estimated from one period."""
