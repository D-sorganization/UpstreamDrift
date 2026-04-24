"""
Torque preview widget for polynomial joint torques.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class TorquePreviewWidget(QWidget):
    """Plot polynomial torque profiles for one or more joints."""

    COLOR_BG = QColor(26, 26, 36)
    COLOR_GRID = QColor(50, 50, 65)
    COLOR_TEXT = QColor(180, 180, 200)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._profiles: list[tuple[str, list[float], QColor]] = []
        self._t_end = 2.0
        # Per-joint clamp limits (None = no clamping, float = symmetric ±limit)
        self._clamp_limits: list[float | None] = []

    def set_duration(self, t_end: float) -> None:
        self._t_end = max(0.1, float(t_end))
        self.update()

    def set_profiles(
        self,
        profiles: Iterable[tuple[str, list[float], QColor]],
        clamp_limits: list[float | None] | None = None,
    ) -> None:
        """Set the polynomial profiles and optional per-joint clamp limits.

        Parameters
        ----------
        profiles: list of (name, coefficients, color)
        clamp_limits: list parallel to profiles with clamp magnitudes or None
        """
        if profiles is None:
            raise ValueError("profiles must be provided")
        self._profiles = [
            (name, list(coeffs), color) for name, coeffs, color in profiles
        ]
        self._clamp_limits = list(clamp_limits) if clamp_limits else []
        self.update()

    def paintEvent(self, event: object) -> None:
        if event is None:
            raise ValueError("event must be provided")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        rect = self.rect().adjusted(10, 10, -10, -24)
        self._draw_grid(painter, QRectF(rect))

        if not self._profiles:
            painter.setPen(self.COLOR_TEXT)
            painter.setFont(QFont("Sans", 9))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Torque preview"
            )
            painter.end()
            return

        t = np.linspace(0.0, self._t_end, 120)
        series = []
        for name, coeffs, _ in self._profiles:
            if len(coeffs) == 0:
                series.append((name, np.zeros_like(t)))
                continue
            poly = np.array(coeffs[::-1])
            series.append((name, np.polyval(poly, t)))

        # Build clamped versions when limits are active
        clamped_series: list[np.ndarray | None] = []
        for i, (_, vals) in enumerate(series):
            limit = self._clamp_limits[i] if i < len(self._clamp_limits) else None
            if limit is not None and np.isfinite(limit):
                clamped_series.append(np.clip(vals, -limit, limit))
            else:
                clamped_series.append(None)

        # Compute Y scale from all values (including unclamped for context)
        all_vals = np.concatenate([s[1] for s in series]) if series else np.array([0.0])
        v_min = float(np.min(all_vals))
        v_max = float(np.max(all_vals))
        if abs(v_max - v_min) < 1e-6:
            v_min -= 1.0
            v_max += 1.0

        qrect = QRectF(rect)

        # Draw clamp limit lines (horizontal dashed)
        for i, (__, ___, color) in enumerate(self._profiles):
            limit = self._clamp_limits[i] if i < len(self._clamp_limits) else None
            if limit is not None and np.isfinite(limit):
                clamp_pen = QPen(color.darker(150), 1, Qt.PenStyle.DashLine)
                painter.setPen(clamp_pen)
                for lv in [limit, -limit]:
                    y = qrect.bottom() - (lv - v_min) / (v_max - v_min) * qrect.height()
                    if qrect.top() <= y <= qrect.bottom():
                        painter.drawLine(
                            QPointF(qrect.left(), y), QPointF(qrect.right(), y)
                        )

        for idx, ((_, values), (__, ___, color)) in enumerate(
            zip(series, self._profiles, strict=True)
        ):
            clamped = clamped_series[idx]

            # If clamped, draw unclamped as thin dashed (demand)
            if clamped is not None:
                demand_pen = QPen(color, 1, Qt.PenStyle.DotLine)
                painter.setPen(demand_pen)
                points: list[QPointF] = []
                for i, val in enumerate(values):
                    x = qrect.left() + (t[i] / self._t_end) * qrect.width()
                    y = (
                        qrect.bottom()
                        - (val - v_min) / (v_max - v_min) * qrect.height()
                    )
                    points.append(QPointF(x, y))
                for i in range(1, len(points)):
                    painter.drawLine(points[i - 1], points[i])

                # Draw clamped as thick solid (effective)
                eff_pen = QPen(color, 2)
                painter.setPen(eff_pen)
                points = []
                for i, val in enumerate(clamped):
                    x = qrect.left() + (t[i] / self._t_end) * qrect.width()
                    y = (
                        qrect.bottom()
                        - (val - v_min) / (v_max - v_min) * qrect.height()
                    )
                    points.append(QPointF(x, y))
                for i in range(1, len(points)):
                    painter.drawLine(points[i - 1], points[i])
            else:
                # No clamping — draw as normal
                pen = QPen(color, 2)
                painter.setPen(pen)
                points = []
                for i, val in enumerate(values):
                    x = qrect.left() + (t[i] / self._t_end) * qrect.width()
                    y = (
                        qrect.bottom()
                        - (val - v_min) / (v_max - v_min) * qrect.height()
                    )
                    points.append(QPointF(x, y))
                for i in range(1, len(points)):
                    painter.drawLine(points[i - 1], points[i])

        self._draw_legend(painter, qrect)
        painter.end()

    def _draw_grid(self, painter: QPainter, rect: QRectF) -> None:
        if painter is None:
            raise ValueError("painter must be provided")
        painter.setPen(QPen(self.COLOR_GRID, 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = rect.top() + i * rect.height() / 4
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for i in range(5):
            x = rect.left() + i * rect.width() / 4
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

    def _draw_legend(self, painter: QPainter, rect: QRectF) -> None:
        if painter is None:
            raise ValueError("painter must be provided")
        painter.setFont(QFont("Sans", 8))
        x = rect.left()
        y = rect.bottom() + 14
        for name, _, color in self._profiles:
            painter.setPen(color)
            painter.drawText(int(x), int(y), name)
            x += 70
