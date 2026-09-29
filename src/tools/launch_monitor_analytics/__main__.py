"""Run the standalone Launch Monitor Analytics workbench."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from src.tools.launch_monitor_analytics.gui import main as gui_main
    except ImportError as exc:
        sys.stderr.write(
            "Launch Monitor Analytics requires GUI dependencies. Install with:\n"
            "  pip install upstream-drift[gui-tools]\n"
            f"Import error: {exc}\n"
        )
        return 1
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
