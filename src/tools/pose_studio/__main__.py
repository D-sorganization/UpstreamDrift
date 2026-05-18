"""Run the Pose Studio GUI.

Usage::

    python -m src.tools.pose_studio

Requires the ``gui-tools`` extra (PyQt6 + matplotlib QtAgg backend).
"""

from __future__ import annotations

import sys


def get_dockable_ui():
    """Return the main window instance for docking in the unified launcher."""
    from src.tools.pose_studio.gui import get_dockable_ui as _get_dock
    return _get_dock()


def main() -> int:
    try:
        from src.tools.pose_studio.gui import main as _gui_main
    except ImportError as exc:
        sys.stderr.write(
            "Could not import the Pose Studio GUI dependencies "
            f"(PyQt6 / matplotlib): {exc}\n\n"
            "Install with:\n  pip install upstream-drift[gui-tools]\n"
        )
        return 1
    return _gui_main()


if __name__ == "__main__":
    sys.exit(main())
