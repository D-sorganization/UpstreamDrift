"""Run the Starting-Pose Matcher GUI.

Usage::

    python -m src.tools.starting_pose_matcher

Requires the ``gui-tools`` extra (see pyproject.toml)::

    pip install upstream-drift[gui-tools]
"""

from __future__ import annotations

import sys


def get_dockable_ui():
    """Return the main window instance for docking in the unified launcher."""
    from src.tools.starting_pose_matcher.gui import get_dockable_ui as _get_dock
    return _get_dock()


def main() -> int:
    try:
        from src.tools.starting_pose_matcher.gui import main as _gui_main
    except ImportError as exc:
        sys.stderr.write(
            "Could not import the matcher's GUI dependencies "
            f"(PyQt6 / matplotlib): {exc}\n\n"
            "Install with:\n  pip install upstream-drift[gui-tools]\n"
        )
        return 1
    return _gui_main()


if __name__ == "__main__":
    sys.exit(main())
