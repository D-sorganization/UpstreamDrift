import json
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any


class WorkflowDiagnosticContext:
    """Context manager for heavy workflow tests that records state and dumps diagnostics on failure."""

    def __init__(self, dump_dir: str, workflow_name: str) -> None:
        self.dump_dir = Path(dump_dir)
        self.workflow_name = workflow_name
        self.states: dict[str, Any] = {}
        self.start_time = datetime.now()

    def record_state(self, step_name: str, state: Any) -> None:
        """Record the intermediate state for a workflow step."""
        self.states[step_name] = state

    def __enter__(self) -> "WorkflowDiagnosticContext":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            # An exception occurred, dump the diagnostics
            self._dump_diagnostics(exc_type, exc_val, exc_tb)
            # Do not swallow the exception
            return False
        return True

    def _dump_diagnostics(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Dump the recorded states to a file for diagnostics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = self.dump_dir / f"{self.workflow_name}_{timestamp}"
        dump_path.mkdir(parents=True, exist_ok=True)

        diagnostic_data = {
            "workflow_name": self.workflow_name,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "exception_type": exc_type.__name__,
            "exception_msg": str(exc_val),
            "traceback": "".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
            "states": self.states,
        }

        # Write to JSON (handles basic types, should serialize safely)
        with open(dump_path / "diagnostics.json", "w", encoding="utf-8") as f:
            try:
                json.dump(diagnostic_data, f, indent=4, default=str)
            except (TypeError, ValueError, OverflowError) as e:
                f.write(f"Failed to serialize state: {e}\n\nStates stringified: {str(self.states)}")
