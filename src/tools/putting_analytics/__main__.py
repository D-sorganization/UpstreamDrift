"""Open the putting analytics window: ``python -m src.tools.putting_analytics``."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.tools.standalone_window import run_standalone_widget  # noqa: E402

WINDOW_SIZE = (1100, 760)


def main() -> int:
    """Launch the standalone window; returns a process exit code."""
    return run_standalone_widget(
        "src.tools.putting_analytics.gui:MainWidget", WINDOW_SIZE
    )


if __name__ == "__main__":
    raise SystemExit(main())
