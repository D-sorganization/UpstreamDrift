"""Run the Simulation Backends GUI.

Usage::

    python -m src.tools.simulation_backends_launcher

Requires the ``gui-tools`` extra (PyQt6 + the matplotlib QtAgg backend).
If those dependencies are missing the module prints a friendly install hint
to ``stderr`` (``print`` is forbidden under ``src/``) and exits non-zero.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Entry point for ``python -m src.tools.simulation_backends_launcher``."""
    try:
        from src.tools.simulation_backends_launcher.gui import main as _gui_main
    except ImportError as exc:
        sys.stderr.write(
            "Could not import the Simulation Backends GUI dependencies "
            f"(PyQt6 / matplotlib): {exc}\n\n"
            "Install with:\n  pip install upstream-drift[gui-tools]\n"
        )
        return 1
    return _gui_main()


if __name__ == "__main__":
    sys.exit(main())
