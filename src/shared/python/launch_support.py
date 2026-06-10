"""Shared launch-error handling for the top-level launch entry points.

The project's front-door scripts (``launch_upstream_drift.py``,
``launch_golf_suite.py``) historically called their dispatch bare, so any
failure (missing optional dependency, port already in use, corrupt env) greeted
the user — often on first run — with a raw traceback and a crash exit. This
helper wraps the dispatch so expected failures produce an actionable one-line
message and a clean ``SystemExit(1)``, with the full traceback available only at
DEBUG/verbose level (issue #7168).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger("upstream_drift_launcher")

# Exit codes: 1 = launch failure (2 is reserved by argparse for bad arguments).
EXIT_LAUNCH_FAILURE = 1


def run_launch(dispatch: Callable[[], None], *, hint: str | None = None) -> None:
    """Run ``dispatch`` and convert expected failures into a clean exit.

    Args:
        dispatch: Zero-arg callable that performs the actual launch.
        hint: Optional remediation hint appended to the error message
            (e.g. "try --help or see docs/...").

    Raises:
        SystemExit: With code :data:`EXIT_LAUNCH_FAILURE` if ``dispatch`` raises
            an expected runtime failure. ``KeyboardInterrupt`` and
            ``SystemExit`` propagate unchanged.
    """
    suffix = f" {hint}" if hint else ""
    try:
        dispatch()
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - front door: report cleanly, don't crash
        # Full traceback only when the user asked for verbose/DEBUG output.
        logger.debug("Launch failed (full traceback):", exc_info=True)
        logger.error("Launch failed: %s: %s.%s", type(exc).__name__, exc, suffix)
        raise SystemExit(EXIT_LAUNCH_FAILURE) from exc


__all__ = ["EXIT_LAUNCH_FAILURE", "run_launch"]
