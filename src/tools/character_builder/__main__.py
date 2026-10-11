"""Standalone entry point: ``python -m src.tools.character_builder``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the Character Builder window."""
    from PyQt6 import QtWidgets

    from .gui import CharacterBuilderWidget

    args = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(args)
    widget = CharacterBuilderWidget()
    widget.setWindowTitle("Character Builder")
    widget.resize(820, 600)
    app.aboutToQuit.connect(widget.cleanup)
    widget.show()
    return app.exec()


def get_dockable_ui() -> object:
    """Return a window for docking in the unified launcher."""
    from .gui import CharacterBuilderWidget

    widget = CharacterBuilderWidget()
    widget.setWindowTitle("Character Builder")
    return widget


if __name__ == "__main__":
    sys.exit(main())
