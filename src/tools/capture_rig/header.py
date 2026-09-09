"""The tile's header: session line, status strip and layout bar (#9816).

:class:`TileStatus` is the Qt-free model of what the strip shows (cameras
bound, recorder phase, last take outcome); :class:`StatusStrip` renders it
as themed chips and :class:`HeaderBar` frames session line, strip and the
pane :class:`~.layout.LayoutBar` as one toolbar. Styles come from
:mod:`.styling`; :meth:`HeaderBar.restyle` re-reads them on a theme change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.typography import Sizes, Weights, get_qfont

from . import styling
from .record_bar import Phase

IDLE_TEXT = "recorder idle"


def cameras_text(bound: int) -> str:
    """Precondition: ``bound >= 0``."""
    require(bound >= 0, "camera count must be non-negative", bound)
    return "cameras: none bound" if bound == 0 else f"cameras: {bound} bound"


def recording_kind(phase: Phase) -> str:
    """Chip kind for the recorder phase (see :data:`styling.CHIP_KINDS`)."""
    if phase is Phase.RECORDING:
        return "record"
    return "warning" if phase is Phase.COUNTDOWN else "neutral"


@dataclass(frozen=True)
class TileStatus:
    """What the status strip shows. Invariant: ``cameras_bound >= 0``."""

    cameras_bound: int = 0
    phase: Phase = Phase.IDLE
    readout: str = ""
    last_take_code: int | None = None

    def __post_init__(self) -> None:
        require(self.cameras_bound >= 0, "camera count must be non-negative")

    def with_last_take(self, exit_code: int) -> TileStatus:
        return replace(self, last_take_code=exit_code)

    def last_take_text(self) -> str:
        if self.last_take_code is None:
            return "last take: none yet"
        if self.last_take_code == 0:
            return "last take: written"
        return f"last take: failed (exit {self.last_take_code})"

    def last_take_kind(self) -> str:
        if self.last_take_code is None:
            return "neutral"
        return "ok" if self.last_take_code == 0 else "warning"

    def recording_text(self) -> str:
        return self.readout or IDLE_TEXT


class StatusStrip(QWidget):
    """Three chips: cameras bound, recorder state, last take outcome."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = TileStatus()
        self.cameras = QLabel()
        self.recording = QLabel()
        self.last_take = QLabel()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(LayoutMetrics.SPACING_SM)
        for chip in (self.cameras, self.recording, self.last_take):
            chip.setFont(get_qfont(Sizes.SM, Weights.MEDIUM))
            row.addWidget(chip)
        self.refresh()

    @property
    def status(self) -> TileStatus:
        return self._status

    def set_cameras(self, bound: int) -> None:
        self._status = replace(self._status, cameras_bound=bound)
        self.refresh()

    def set_recording(self, phase: Phase, readout: str) -> None:
        self._status = replace(self._status, phase=phase, readout=readout)
        self.refresh()

    def set_last_take(self, exit_code: int) -> None:
        self._status = self._status.with_last_take(exit_code)
        self.refresh()

    def refresh(self) -> None:
        """Texts and chip styles from the model (also the theme-change hook)."""
        status = self._status
        self.cameras.setText(cameras_text(status.cameras_bound))
        self.cameras.setStyleSheet(
            styling.chip_style("ok" if status.cameras_bound else "neutral")
        )
        self.recording.setText(status.recording_text())
        self.recording.setStyleSheet(styling.chip_style(recording_kind(status.phase)))
        self.last_take.setText(status.last_take_text())
        self.last_take.setStyleSheet(styling.chip_style(status.last_take_kind()))

    restyle = refresh


class HeaderBar(QFrame):
    """Session line on the left, status chips and the layout bar on the right."""

    def __init__(self, layout_bar: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        require(layout_bar is not None, "a layout bar is required")
        self.setObjectName(styling.HEADER_OBJECT_NAME)
        self.session = QLabel("no session loaded")
        self.session.setFont(get_qfont(Sizes.MD, Weights.MEDIUM))
        self.session.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status = StatusStrip()
        self.layout_bar = layout_bar
        row = QHBoxLayout(self)
        row.setContentsMargins(
            LayoutMetrics.SPACING_MD,
            LayoutMetrics.SPACING_SM,
            LayoutMetrics.SPACING_MD,
            LayoutMetrics.SPACING_SM,
        )
        row.setSpacing(LayoutMetrics.SPACING_LG)
        row.addWidget(self.session, 1)
        row.addWidget(self.status)
        row.addWidget(layout_bar)
        self.restyle()

    def set_session(self, text: str) -> None:
        self.session.setText(text)

    def restyle(self) -> None:
        self.setStyleSheet(styling.header_style())
        self.status.restyle()
