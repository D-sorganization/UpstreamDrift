"""Run the Motion Matching tool.

Usage::

    python -m src.tools.motion_matching

Requires PyQt6 (the ``gui-tools`` extra). The tool drives the full-body
matching pipeline for the tour-average driver and 7-iron captures; see
``docs/development/full_body_models/HANDOFF.md``.
"""

from __future__ import annotations

import sys


def get_dockable_ui():  # noqa: ANN201 - launcher protocol
    """Return the main window instance for docking in the unified launcher."""
    from src.tools.motion_matching.gui import get_dockable_ui as _get_dock

    return _get_dock()


def main() -> int:
    try:
        from src.tools.motion_matching.gui import main as _gui_main
    except ImportError as exc:
        sys.stderr.write(
            f"Could not import the Motion Matching GUI dependencies (PyQt6): {exc}\n"
            "Install with:\n  pip install upstream-drift[gui-tools]\n"
        )
        return 1
    return _gui_main()


if __name__ == "__main__":
    sys.exit(main())
