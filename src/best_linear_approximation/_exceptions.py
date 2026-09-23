# Exceptions


class InsufficientExperimentsError(ValueError):
    """Raised when the data cannot form an experiment for frequency-response estimation."""


class InvalidSignalAxesError(ValueError):
    """Raised when signal arrays violate required axis-size relationships."""


class InvalidSignalRanksError(ValueError):
    """Raised when signal arrays do not have a supported number of dimensions."""


class NoExcitedBinsError(ValueError):
    """Raised when no excited frequency bins are found in a signal array."""


# Warnings


class NoiseCovarianceUnavailableWarning(UserWarning):
    """Warn that noise covariance cannot be estimated from one period."""


class PossibleExcitationAmplitudeMismatchWarning(UserWarning):
    """Warn that excitation amplitude may change between adjacent realizations."""


class PossibleTransientWarning(UserWarning):
    """Warn that the output data may not have reached steady state."""


class RealizationsTruncatedWarning(UserWarning):
    """Warn that trailing realizations will be discarded before estimation."""


class TotalCovarianceUnavailableWarning(UserWarning):
    """Warn that total covariance cannot be estimated from one experiment."""
