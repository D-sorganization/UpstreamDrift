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

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


REPO_URL = "https://github.com/D-sorganization/UpstreamDrift"
ISSUES_URL = f"{REPO_URL}/issues"


def _read_version_file() -> str | None:
    """Return the contents of the repo-root ``VERSION`` file, if present.

    Returns:
        The stripped first line of ``VERSION``, or ``None`` when the file
        does not exist or cannot be read. Never raises.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "VERSION",
        Path(__file__).resolve().parents[3] / "VERSION",
    ]
    for path in candidates:
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text.splitlines()[0].strip()
        except OSError:
            continue
    return None


def _resolve_app_version() -> str:
    """Resolve the application version string.

    Resolution order:
        1. ``VERSION`` file at repo root.
        2. ``importlib.metadata`` for ``upstream-drift``.
        3. Hardcoded fallback.

    Returns:
        Version string (never empty).
    """
    v = _read_version_file()
    if v:
        return v
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("upstream-drift")
        except PackageNotFoundError:
            pass
        try:
            return version("golf-modeling-suite")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    return "1.0.0-beta"


def _safe_version(import_name: str) -> str:
    """Import a module and return ``__version__`` if available.

    Args:
        import_name: Module to import (e.g. ``"numpy"``).

    Returns:
        The module's ``__version__`` string, or ``"not installed"`` if the
        module cannot be imported, or ``"unknown"`` if the module is
        importable but does not expose ``__version__``.
    """
    # Issue #5911: ``ImportError`` was previously a bare ``except Exception``.
    # Narrowed to the real exception types that ``__import__`` raises.
    # ``ModuleNotFoundError`` is a subclass of ``ImportError`` (caught).
    # ``ValueError`` covers the ``__import__("")`` empty-name path.
    try:
        mod = __import__(import_name)
    except (ImportError, ValueError):
        return "not installed"
    return str(getattr(mod, "__version__", "unknown"))


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


def open_user_guide() -> None:
    """Open the bundled user guide in the system browser.

    Looks for ``docs/user_guide/index.md`` relative to the repository
    root. Falls back to opening the repository URL if no local copy is
    found.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "docs" / "USER_MANUAL.md",
        Path(__file__).resolve().parents[2] / "docs" / "user_guide" / "index.md",
        Path(__file__).resolve().parents[3] / "docs" / "user_guide" / "index.md",
    ]
    for path in candidates:
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
