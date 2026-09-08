"""``python -m src.tools.capture_rig``: the Capture Rig in its own window."""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from .gui import CaptureRigWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = CaptureRigWindow(autostart_preview=True)  # applies the theme itself
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
