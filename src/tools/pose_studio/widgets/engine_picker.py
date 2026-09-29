"""Engine picker widget with status pill.

A :class:`QComboBox` listing every engine in :data:`SUPPORTED_ENGINES`
plus a coloured status pill (mock/yellow, live/green, error/red) so the
user can see at a glance whether the selected engine is running on a
real wheel or the headless mock fallback.

The widget is purely presentational; the wiring to
:class:`EngineController` lives in :mod:`src.tools.pose_studio.gui`.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from src.shared.python.theme.style_constants import Styles
from src.tools.pose_studio.core import SUPPORTED_ENGINES, EngineStatus


class EnginePicker(QtWidgets.QWidget):
    """Combo box + colour pill for the active live-kinematics engine.

    Signals
    -------
    engine_selected(str)
        Emitted with the new engine name when the user changes the
        combo selection.
    """

    engine_selected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        initial_engine: str = "drake",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if initial_engine not in SUPPORTED_ENGINES:
            raise ValueError(
                f"initial_engine {initial_engine!r} not in SUPPORTED_ENGINES "
                f"{SUPPORTED_ENGINES!r}"
            )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QtWidgets.QLabel("Engine:"))

        self.combo = QtWidgets.QComboBox()
        self.combo.setToolTip(
            "Select the active physics engine. Switching swaps the live "
            "kinematics service and the pose-convention adapter without a "
            "process restart."
        )
        for engine in SUPPORTED_ENGINES:
            self.combo.addItem(engine)
        self.combo.setCurrentText(initial_engine)
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addWidget(self.combo)

        self.status_pill = QtWidgets.QLabel(EngineStatus.MOCK.value)
        self.status_pill.setToolTip(
            "Engine status: 'live' (real engine wheel), 'mock' (headless "
            "fallback when the wheel is not installed), or 'error' (engine "
            "failed to activate; see log)."
        )
        self.status_pill.setMinimumWidth(48)
        self.status_pill.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.set_status(EngineStatus.MOCK)
        layout.addWidget(self.status_pill)

        layout.addStretch(1)

    # ---- public surface ------------------------------------------------

    def current_engine(self) -> str:
        """Return the currently selected engine name."""
        return self.combo.currentText()

    def set_status(self, status: EngineStatus) -> None:
        """Re-paint the status pill for *status*."""
        if not isinstance(status, EngineStatus):
            raise TypeError(
                f"status must be an EngineStatus, got {type(status).__name__}"
            )
        self.status_pill.setText(status.value)
        if status is EngineStatus.LIVE:
            self.status_pill.setStyleSheet(Styles.STATUS_SUCCESS_BOLD)
        elif status is EngineStatus.MOCK:
            self.status_pill.setStyleSheet(Styles.STATUS_WARNING)
        else:
            self.status_pill.setStyleSheet(Styles.STATUS_ERROR_BOLD)

    def set_engine(self, engine_name: str) -> None:
        """Programmatically set the combo to *engine_name* (no signal)."""
        if engine_name not in SUPPORTED_ENGINES:
            raise ValueError(
                f"engine_name {engine_name!r} not in SUPPORTED_ENGINES "
                f"{SUPPORTED_ENGINES!r}"
            )
        blocker = QtCore.QSignalBlocker(self.combo)
        self.combo.setCurrentText(engine_name)
        del blocker

    # ---- internals -----------------------------------------------------

    def _on_combo_changed(self, engine: str) -> None:
        self.engine_selected.emit(engine)


__all__ = ["EnginePicker"]
