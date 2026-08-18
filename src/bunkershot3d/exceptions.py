"""
Custom exceptions for BunkerShot3D.
"""


class BackendNotImplementedError(NotImplementedError):
    """Raised when a requested simulation backend is not available or not implemented."""

    def __init__(self, backend: str, feature: str = "") -> None:
        msg = f"Backend '{backend}' is not implemented."
        if feature:
            msg += f" Feature: {feature}"
        super().__init__(msg)
        self.backend = backend
        self.feature = feature
