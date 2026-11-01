"""Open the putting analytics window: ``python -m src.tools.putting_analytics``."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main() -> int:
    """Launch the standalone window; returns a process exit code."""
    try:
        from PyQt6.QtWidgets import QApplication

        from src.tools.putting_analytics.gui import MainWidget
    except (ImportError, OSError) as exc:
        sys.stderr.write(
            "The desktop window requires the GUI extra. Install with:\n"
            "  pip install -e '.[gui-tools]'\n"
            f"Import error: {exc}\n"
        )
        return 1
    app = QApplication(sys.argv)
    widget = MainWidget()
    widget.resize(1100, 760)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
