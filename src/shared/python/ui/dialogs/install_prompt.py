"""Install-prompt dialog for missing optional dependencies.

When a user requests a feature that requires an optional package that is
not currently installed, this dialog offers to install it in the
background, skip this time, or permanently suppress the prompt.

Architecture
------------
* :class:`InstallPromptDialog` — PyQt6 modal dialog with three actions.
* :class:`_InstallWorker` — ``QThread`` subclass that runs the pip
  install in a background thread so the UI stays responsive.
* :data:`PREFS_FILE` — ``~/.upstreamdrift/prefs.json`` stores
  ``dont_ask_again`` feature flags across sessions.

Design by Contract
------------------
Preconditions:
    * ``feature_name`` must be a non-empty string registered in the
      feature registry.
    * ``package_name`` must be a non-empty string (the human-readable
      package label shown in the dialog body).
Postconditions:
    * If the user clicks **Yes**, the install runs to completion (or
      fails) in a background thread; ``registry.refresh()`` is called
      regardless of success so the registry reflects the post-install
      state.
    * If the user clicks **Don't ask again**, the suppression preference
      is written to :data:`PREFS_FILE` before the dialog closes.

Law of Demeter
--------------
The dialog does not reach into the registry internals.  It calls
:func:`src.shared.python.feature_registry.install_feature` and
:func:`src.shared.python.feature_registry.refresh` via the public API
only.

No ``print()`` — all diagnostic output goes through ``logging``.
"""

from __future__ import annotations

import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.feature_registry import install_feature, refresh  # noqa: F401

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User prefs file
# ---------------------------------------------------------------------------

PREFS_DIR: Path = Path.home() / ".upstreamdrift"
PREFS_FILE: Path = PREFS_DIR / "prefs.json"

_DONT_ASK_KEY = "dont_ask_again_features"


def _load_prefs() -> dict:
    """Load prefs from :data:`PREFS_FILE`, returning an empty dict on failure."""
    if PREFS_FILE.exists():
        try:
            return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("Failed to read prefs file %s: %s", PREFS_FILE, exc)
    return {}


def _save_prefs(data: dict) -> None:
    """Persist *data* to :data:`PREFS_FILE`.

    Preconditions:
        * ``data`` is a JSON-serialisable dict.
    """
    try:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        PREFS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Failed to save prefs to %s: %s", PREFS_FILE, exc)


def is_suppressed(feature_name: str) -> bool:
    """Return ``True`` if the user has chosen *Don't ask again* for this feature.

    Preconditions:
        * ``feature_name`` is a non-empty string.
    """
    if not isinstance(feature_name, str) or not feature_name:
        raise ValueError("feature_name must be a non-empty string")
    prefs = _load_prefs()
    return feature_name in prefs.get(_DONT_ASK_KEY, [])


def suppress_feature(feature_name: str) -> None:
    """Persist a *Don't ask again* entry for ``feature_name``.

    Preconditions:
        * ``feature_name`` is a non-empty string.
    """
    if not isinstance(feature_name, str) or not feature_name:
        raise ValueError("feature_name must be a non-empty string")
    prefs = _load_prefs()
    suppressed: list[str] = prefs.setdefault(_DONT_ASK_KEY, [])
    if feature_name not in suppressed:
        suppressed.append(feature_name)
        _save_prefs(prefs)
        logger.info("Install prompt suppressed for feature %r", feature_name)


# ---------------------------------------------------------------------------
# Result enum
# ---------------------------------------------------------------------------


class InstallPromptResult(Enum):
    """Outcome of :class:`InstallPromptDialog`.

    Attributes:
        YES: User clicked **Yes**; install was attempted.
        NO: User clicked **No**; install was skipped this session.
        DONT_ASK: User clicked **Don't ask again**; suppression persisted.
        SUPPRESSED: Dialog was not shown because the user had previously
            chosen *Don't ask again*.
    """

    YES = auto()
    NO = auto()
    DONT_ASK = auto()
    SUPPRESSED = auto()


# ---------------------------------------------------------------------------
# Background install worker
# ---------------------------------------------------------------------------


class _InstallWorker(QThread):
    """Run :func:`install_feature` in a background thread.

    Signals:
        finished(success: bool, reason: str): Emitted when the install
            subprocess exits (or is rejected by a safety rail).
        progress(message: str): Emitted periodically while the install
            is running (currently only at start/end; a richer
            implementation could tail the subprocess stdout).
    """

    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, feature_name: str, parent: QWidget | None = None) -> None:
        """Initialise the worker.

        Preconditions:
            * ``feature_name`` is a non-empty string.
        """
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("feature_name must be a non-empty string")
        super().__init__(parent)
        self._feature_name = feature_name

    def run(self) -> None:
        """Execute the install and emit :attr:`finished`."""
        from src.shared.python.ui.dialogs import install_prompt as _mod

        self.progress.emit(f"Installing {self._feature_name}…")
        logger.info("Background install started for feature %r", self._feature_name)
        try:
            result = _mod.install_feature(self._feature_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error during install of %r", self._feature_name
            )
            self.finished.emit(False, str(exc))
            return
        logger.info(
            "Background install finished for %r: success=%s reason=%s",
            self._feature_name,
            result.success,
            result.reason,
        )
        self.finished.emit(result.success, result.reason)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class InstallPromptDialog(QDialog):
    """Modal dialog prompting the user to install a missing optional package.

    The dialog presents three choices:

    * **Yes** — trigger an async pip install and show a progress bar.
    * **No** — close without installing (ask again next time).
    * **Don't ask again** — close and persist a suppression flag.

    On completion of the background install (success *or* failure) the
    dialog calls :func:`src.shared.python.feature_registry.refresh` so
    the registry reflects the post-install state.

    Usage::

        dialog = InstallPromptDialog(
            feature_name="drake",
            package_name="Drake",
            parent=main_window,
        )
        result = dialog.prompt()
        if result == InstallPromptResult.YES:
            ...

    Args:
        feature_name: Registry key (e.g. ``"drake"``).
        package_name: Human-readable label shown in the dialog body
            (e.g. ``"Drake"``).
        parent: Optional parent widget.
    """

    #: Emitted after the background install finishes (success, reason).
    install_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        feature_name: str,
        package_name: str,
        parent: QWidget | None = None,
    ) -> None:
        """Create the dialog.

        Preconditions:
            * ``feature_name`` is a non-empty string.
            * ``package_name`` is a non-empty string.
        """
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("feature_name must be a non-empty string")
        if not isinstance(package_name, str) or not package_name:
            raise ValueError("package_name must be a non-empty string")
        super().__init__(parent)

        self._feature_name = feature_name
        self._package_name = package_name
        self._result: InstallPromptResult | None = None
        self._worker: _InstallWorker | None = None

        self.setWindowTitle("Missing Optional Dependency")
        self.setMinimumWidth(420)
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prompt(self) -> InstallPromptResult:
        """Show the dialog if not suppressed; return the user's choice.

        If the feature is already suppressed via *Don't ask again*, this
        method returns :attr:`InstallPromptResult.SUPPRESSED` immediately
        without showing a dialog.

        Postconditions:
            * Returns a member of :class:`InstallPromptResult`.
        """
        if is_suppressed(self._feature_name):
            logger.debug(
                "Install prompt suppressed for %r; skipping display",
                self._feature_name,
            )
            return InstallPromptResult.SUPPRESSED
        self.exec()
        return self._result if self._result is not None else InstallPromptResult.NO

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the dialog widgets."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Body text
        body_text = (
            f"Feature <b>{self._feature_name}</b> requires installing "
            f"<b>{self._package_name}</b>.<br/><br/>"
            "Install now?"
        )
        self._label = QLabel(body_text)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        # Progress bar (hidden until install starts)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        layout.addWidget(self._progress)

        # Status label (hidden until install starts)
        self._status_label = QLabel("")
        self._status_label.hide()
        layout.addWidget(self._status_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self._yes_btn = QPushButton("Yes")
        self._yes_btn.setDefault(True)
        self._yes_btn.clicked.connect(self._on_yes)
        btn_layout.addWidget(self._yes_btn)

        self._no_btn = QPushButton("No")
        self._no_btn.clicked.connect(self._on_no)
        btn_layout.addWidget(self._no_btn)

        self._dont_ask_btn = QPushButton("Don't ask again")
        self._dont_ask_btn.clicked.connect(self._on_dont_ask)
        btn_layout.addWidget(self._dont_ask_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_yes(self) -> None:
        """Start background install and show progress bar."""
        self._result = InstallPromptResult.YES
        self._yes_btn.setEnabled(False)
        self._no_btn.setEnabled(False)
        self._dont_ask_btn.setEnabled(False)
        self._progress.show()
        self._status_label.show()
        self._status_label.setText(f"Installing {self._package_name}…")

        self._worker = _InstallWorker(self._feature_name, parent=self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_no(self) -> None:
        """Close without installing."""
        self._result = InstallPromptResult.NO
        self.reject()

    def _on_dont_ask(self) -> None:
        """Persist suppression and close."""
        suppress_feature(self._feature_name)
        self._result = InstallPromptResult.DONT_ASK
        self.reject()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_worker_progress(self, message: str) -> None:
        """Update status label with progress text."""
        self._status_label.setText(message)

    def _on_worker_finished(self, success: bool, reason: str) -> None:
        """Handle install completion: refresh registry and close.

        Postconditions:
            * :func:`src.shared.python.feature_registry.refresh` has been
              called before this method returns.
            * :attr:`install_finished` signal has been emitted.
        """
        self._progress.hide()
        status = "installed successfully" if success else f"install failed: {reason}"
        self._status_label.setText(f"{self._package_name} {status}.")
        logger.info(
            "InstallPromptDialog: install of %r finished — success=%s",
            self._feature_name,
            success,
        )

        # Hot-refresh the capability registry regardless of success/failure
        # so the registry reflects the true post-install state.
        try:
            from src.shared.python.ui.dialogs import install_prompt as _mod

            _mod.refresh()
            logger.debug(
                "Capability registry refreshed after install of %r",
                self._feature_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registry refresh failed after install: %s", exc)

        self.install_finished.emit(success, reason)
        self.accept()


__all__ = [
    "PREFS_DIR",
    "PREFS_FILE",
    "InstallPromptDialog",
    "InstallPromptResult",
    "is_suppressed",
    "suppress_feature",
]
