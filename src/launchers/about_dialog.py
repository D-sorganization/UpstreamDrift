"""About dialog for the launcher.

Builds an "About UpstreamDrift" dialog that shows live runtime version
information for Python, Qt (via PyQt6), NumPy, ezc3d, and the application
itself. The version of the app is resolved from the ``VERSION`` file at
the repository root when present, otherwise from package metadata, and
finally a hardcoded fallback.

This is intentionally a thin module so it can be imported lazily from the
help menu without pulling heavy dependencies at launcher startup.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QT_VERSION_STR, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox

from src.shared.python.version_info import (
    ISSUES_URL as ISSUES_URL,
)
from src.shared.python.version_info import (
    REPO_URL as REPO_URL,
)
from src.shared.python.version_info import (
    resolve_app_version as _resolve_app_version,
)
from src.shared.python.version_info import (
    safe_module_version as _safe_version,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


def gather_version_info() -> dict[str, str]:
    """Collect version strings for the About dialog.

    Returns:
        Mapping with keys ``app``, ``python``, ``qt``, ``numpy``,
        ``ezc3d``, and ``platform``. All values are strings; missing
        dependencies are reported as ``"not installed"``.
    """
    return {
        "app": _resolve_app_version(),
        "python": platform.python_version(),
        "qt": QT_VERSION_STR,
        "numpy": _safe_version("numpy"),
        "ezc3d": _safe_version("ezc3d"),
        "platform": f"{platform.system()} {platform.release()}",
    }


def build_about_html(info: dict[str, str] | None = None) -> str:
    """Build the HTML body shown in the About dialog.

    Args:
        info: Pre-collected version info (e.g. from
            :func:`gather_version_info`). When ``None`` it is collected
            now.

    Returns:
        HTML string suitable for :class:`QMessageBox.about`.
    """
    if info is None:
        info = gather_version_info()
    return (
        "<h2>UpstreamDrift</h2>"
        "<h3>Biomechanical motion analysis platform</h3>"
        f"<p><b>Version:</b> {info['app']}</p>"
        "<hr>"
        "<p>A unified desktop platform for biomechanical motion analysis "
        "across multiple physics engines including MuJoCo, Drake, "
        "Pinocchio, OpenSim, and MyoSuite.</p>"
        "<p><b>Runtime:</b></p>"
        "<ul>"
        f"<li>Python {info['python']}</li>"
        f"<li>Qt {info['qt']}</li>"
        f"<li>NumPy {info['numpy']}</li>"
        f"<li>ezc3d {info['ezc3d']}</li>"
        f"<li>Platform: {info['platform']}</li>"
        "</ul>"
        f'<p><a href="{REPO_URL}">GitHub repository</a> &middot; '
        f'<a href="{ISSUES_URL}">Report a bug</a></p>'
        "<p>Copyright 2024-2026 UpstreamDrift Contributors.</p>"
    )


def show_about_dialog(parent: QWidget | None = None) -> None:
    """Display the About dialog as a modal :class:`QMessageBox`.

    Args:
        parent: Parent widget for the dialog. Pass ``None`` for an
            application-modal dialog with no parent.

    Postcondition:
        Returns after the user dismisses the dialog. Side effect: a
        modal dialog is shown and then closed.
    """
    QMessageBox.about(parent, "About UpstreamDrift", build_about_html())


def open_issues_page() -> None:
    """Open the public issue tracker in the user's default browser."""
    QDesktopServices.openUrl(QUrl(ISSUES_URL))


def user_guide_candidates() -> list[Path]:
    """Return the bundled user-guide candidates, most readable first.

    All three previous candidates were missing on disk — two named documents
    that have never existed and one ``parents[3]`` variant that resolved
    *outside* the repository entirely — so ``open_user_guide`` silently opened
    the GitHub repo instead of the guide its tooltip promises (issue #8014).
    ``docs/user_guide/user_manual.md`` is explicitly written to be read in the
    built-in document reader, so it is tried first.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "docs" / "user_guide" / "user_manual.md",
        repo_root / "docs" / "user_guide" / "upstream_drift_user_manual.md",
        repo_root / "docs" / "index.md",
    ]


def open_user_guide() -> None:
    """Open the bundled user guide in the in-app document reader."""
    for path in user_guide_candidates():
        if path.exists():
            from src.shared.python.ui.qt.widgets.document_reader import show_document

            show_document(path)
            return
    QDesktopServices.openUrl(QUrl(REPO_URL))


def open_motion_match_loaders_doc() -> None:
    """Open the motion-matching loader reference doc.

    Falls back to the bundled user guide and finally the public repo URL
    when no local copy is present.
    """
    candidates = [
        Path(__file__).resolve().parents[2]
        / "docs"
        / "user_guide"
        / "motion_matching"
        / "loading_targets.md",
        Path(__file__).resolve().parents[2] / "docs" / "motion_matching" / "README.md",
    ]
    for path in candidates:
        if path.exists():
            from src.shared.python.ui.qt.widgets.document_reader import show_document

            show_document(path)
            return
    open_user_guide()


if __name__ == "__main__":
    # Quick manual smoke test:  python -m src.launchers.about_dialog
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    show_about_dialog()
