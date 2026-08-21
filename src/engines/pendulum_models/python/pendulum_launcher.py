"""Direct-launch entry point for the pendulum engine (issue #8967).

``src.shared.python.launcher_factory.ENGINE_MODULES`` routes
``--engine pendulum`` here. The actual GUI lives in the standalone
``double_pendulum_model`` package (tkinter + matplotlib), which uses
absolute ``double_pendulum_model.*`` imports and therefore needs this
directory on ``sys.path`` before it can be imported.

Keep module import light: the GUI (and its tkinter/matplotlib imports)
is only loaded when :func:`main` runs, so path resolution stays cheap
for the launch-mode QA gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Directory containing the ``double_pendulum_model`` package.
_PACKAGE_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Launch the double pendulum GUI.

    Precondition: ``double_pendulum_model`` is importable once this
    module's directory is on ``sys.path`` (it ships with the repo).
    Blocks in the tkinter main loop until the window closes.
    """
    package_dir = str(_PACKAGE_DIR)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    from double_pendulum_model.ui.double_pendulum_gui import run_app

    run_app()


if __name__ == "__main__":
    main()
