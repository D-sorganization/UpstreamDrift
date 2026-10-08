"""Impact Parameters widget: launch-monitor card, D-plane and path views.

Pure presentation over :mod:`impact_parameters.panel_model`; the maths lives in
the shared core.  Unavailable rows render the word "unavailable" with the
reason as a tooltip, never a number.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlencode

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    UNITS,
    ImpactCard,
    build_impact_card,
    target_frame_from_heading,
)

UNAVAILABLE_TEXT = "unavailable"
_ARROW_LEN = 70.0


def format_row(row: Any) -> str:
    """Text for a card row; unavailable rows never show a number."""
    if row.value is None:
        return UNAVAILABLE_TEXT
    text = f"{row.value:.1f} {row.unit}".strip()
    return f"{text} ({row.note})" if row.note else text


class DPlaneView(QWidget):
    """Top view (path/face vs target line) and side view (AoA / dynamic loft)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 170)
        self._angles: dict[str, float | None] = {}

    def set_angles(self, angles: dict[str, float | None]) -> None:
        self._angles = dict(angles)
        self.update()

    def _arrow(
        self, p: QPainter, origin: QPointF, deg: float | None, color: str
    ) -> None:
        if deg is None:
            return
        pen = QPen(QColor(color), 3)
        p.setPen(pen)
        rad = math.radians(deg)
        tip = QPointF(
            origin.x() + _ARROW_LEN * math.cos(rad),
            origin.y() - _ARROW_LEN * math.sin(rad),
        )
        p.drawLine(origin, tip)

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        w, h = self.width() / 2.0, float(self.height())
        for i, title in enumerate(("Top (path, face)", "Side (AoA, loft)")):
            left = i * w
            p.setPen(QPen(QColor("#888888"), 1))
            p.drawRect(QRectF(left + 4, 4, w - 8, h - 8))
            p.drawText(QPointF(left + 10, 20), title)
            origin = QPointF(left + w / 2 - _ARROW_LEN / 2, h / 2 + 10)
            p.setPen(QPen(QColor("#aaaaaa"), 1, Qt.PenStyle.DashLine))
            p.drawLine(origin, QPointF(origin.x() + _ARROW_LEN, origin.y()))
            if i == 0:
                self._arrow(
                    p, origin, _neg(self._angles.get("club_path_deg")), "#2a7de1"
                )
                self._arrow(
                    p, origin, _neg(self._angles.get("face_angle_deg")), "#e1862a"
                )
            else:
                self._arrow(p, origin, self._angles.get("attack_angle_deg"), "#2a7de1")
                self._arrow(p, origin, self._angles.get("dynamic_loft_deg"), "#e1862a")
        p.end()


def _neg(value: float | None) -> float | None:
    return None if value is None else -value


class ImpactParametersWidget(QWidget):
    """Card with units toggle, target-line input and Impact Explorer link."""

    open_impact_explorer_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: ClubheadSeries | None = None
        self._card: ImpactCard | None = None
        self._value_labels: dict[str, QLabel] = {}
        self._build_ui()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.units_combo = QComboBox()
        self.units_combo.addItems(list(UNITS))
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-180.0, 180.0)
        self.target_spin.setSuffix(" deg")
        self.target_spin.setToolTip("Target line heading, counter-clockwise from -Y")
        self.left_handed = QCheckBox("Left-handed")
        for label, widget in (
            ("Units", self.units_combo),
            ("Target", self.target_spin),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(self.left_handed)
        root.addLayout(controls)
        self.status_label = QLabel(UNAVAILABLE_TEXT)
        root.addWidget(self.status_label)
        self.form = QFormLayout()
        root.addLayout(self.form)
        self.diagram = DPlaneView()
        root.addWidget(self.diagram)
        self.explorer_button = QPushButton("Open in Impact Explorer")
        root.addWidget(self.explorer_button)
        self.units_combo.currentTextChanged.connect(self.refresh)
        self.target_spin.valueChanged.connect(self.refresh)
        self.left_handed.toggled.connect(self.refresh)
        self.explorer_button.clicked.connect(self._emit_explorer)

    # ----------------------------------------------------------------- data --
    def set_series(
        self, series: ClubheadSeries | None, impact_index: int | None = None
    ) -> None:
        """Provide the clubhead series (``None`` clears to unavailable)."""
        self._series = series
        self._impact_index = impact_index
        self.refresh()

    def set_card(self, card: ImpactCard) -> None:
        """Show a precomputed card (e.g. from the API)."""
        self._card = card
        self._render()

    def refresh(self, *_: Any) -> None:
        if self._series is None:
            self._card = None
            self._render()
            return
        hand = "left" if self.left_handed.isChecked() else "right"
        frame = target_frame_from_heading(self.target_spin.value(), hand)
        try:
            params = extract_impact_parameters(
                self._series, frame, impact_index=getattr(self, "_impact_index", None)
            )
        except ValueError as exc:
            self._card = None
            self._render(str(exc))
            return
        self._card = build_impact_card(params, units=self.units_combo.currentText())
        self._render()

    def _render(self, reason: str | None = None) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._value_labels.clear()
        card = self._card
        if card is None or not card.available:
            why = reason or (card.reason if card else "no clubhead series")
            self.status_label.setText(f"{UNAVAILABLE_TEXT}: {why}")
            self.diagram.set_angles({})
            self.explorer_button.setEnabled(False)
            return
        self.status_label.setText(
            f"Impact at t={card.impact_time_s:.4f} s ({card.impact_time_source})"
        )
        for row in card.rows:
            label = QLabel(format_row(row))
            if row.value is None and row.reason:
                label.setToolTip(row.reason)
            self._value_labels[row.key] = label
            self.form.addRow(row.label, label)
        self.diagram.set_angles(card.d_plane)
        self.explorer_button.setEnabled(True)

    def delivery(self) -> dict[str, Any]:
        """Delivery dict for Impact Explorer, or empty when unavailable."""
        if self._card is None or not self._card.available:
            return {}
        return {r.key: r.value for r in self._card.rows if r.value is not None}

    def _emit_explorer(self) -> None:
        self.open_impact_explorer_requested.emit(self.delivery())

    def explorer_query(self) -> str:
        """Query string carrying the delivery for the Impact Explorer link."""
        return urlencode(self.delivery())

    def cleanup(self) -> None:
        """Release data references; safe to call repeatedly."""
        self._series = None
        self._card = None
