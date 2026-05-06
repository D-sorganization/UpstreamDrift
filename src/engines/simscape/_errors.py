"""Error hierarchy for the Simscape adapter skeleton.

All errors subclass :class:`src.shared.python.core.error_utils.SimulationError`
so that callers using the platform-wide simulation-error handling continue to
function uniformly across engines.

The MATLAB-specific subclasses defined in
``option4_python_bridge/INTERFACES.md`` (``EngineStartupError``,
``LicenseError``, ``ModelLoadError``) live closer to the MATLAB Engine call
site and will land alongside issue #4006. The skeleton needs only the
three errors below, which are all that the lifecycle layer can raise.
"""

from __future__ import annotations

from src.shared.python.core.error_utils import SimulationError

__all__ = [
    "SimscapeModelNotFoundError",
    "SimscapeNotInstalledError",
    "SimscapeStateError",
]


class SimscapeNotInstalledError(SimulationError):
    """Raised when ``matlab.engine`` is required but not importable.

    The skeleton never tries to import ``matlab.engine``; this error is
    reserved for code paths that genuinely require the MATLAB Engine API
    (i.e. :meth:`SimscapeAdapter.step` and
    :meth:`SimscapeAdapter.simulate_with_coefficients`, both of which are
    deferred to #4006).
    """

    def __init__(self, hint: str | None = None) -> None:
        message = (
            "MATLAB Engine for Python is not installed. "
            "Install with `python -m pip install matlabengine` matching the "
            "host MATLAB release; see option4_python_bridge/INSTALLATION.md."
        )
        if hint:
            message = f"{message} ({hint})"
        super().__init__(message)


class SimscapeModelNotFoundError(SimulationError):
    """Raised when the requested .slx model or its metadata cannot be found."""

    def __init__(self, path: str, *, reason: str | None = None) -> None:
        self.path = path
        message = f"Simscape model not found: {path}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)


class SimscapeStateError(SimulationError):
    """Raised when the adapter is asked to do something illegal for its state.

    Examples include calling :meth:`SimscapeAdapter.step` before the model has
    been loaded or after :meth:`SimscapeAdapter.close` has shut the engine
    down.
    """

    def __init__(
        self,
        operation: str,
        *,
        current_state: str,
        required_state: str,
    ) -> None:
        self.operation = operation
        self.current_state = current_state
        self.required_state = required_state
        message = (
            f"Cannot perform '{operation}' from state '{current_state}'; "
            f"required state is '{required_state}'."
        )
        super().__init__(message)
