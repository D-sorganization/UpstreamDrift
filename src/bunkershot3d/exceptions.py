"""
Custom exceptions for BunkerShot3D.
"""


class BackendNotImplementedError(NotImplementedError):
    """Raised when a simulation backend is not available or not installed.

    This is a subclass of NotImplementedError so that legacy code catching
    NotImplementedError continues to work.
    """
