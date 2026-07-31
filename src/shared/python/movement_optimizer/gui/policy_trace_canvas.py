# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Trace plot canvas for swingset policy searches."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from movement_optimizer.models.swingset import CyclicPolicyTraceSample
from movement_optimizer.rendering import Palette, get_chart_color


def _build_trace_colors() -> dict[str, QColor]:
    return {
        "ARM": QColor(get_chart_color(2)),
        "BODY": QColor(get_chart_color(0)),
        "CHAIN": QColor(Palette.FG_DIM),
        "GRID": QColor(Palette.BG_INPUT),
        "LEG": QColor(get_chart_color(1)),
        "SURFACE": QColor(Palette.BG),
        "TRACE_BEST": QColor(Palette.GREEN),
        "TRACE_PARAM": QColor(get_chart_color(0)),
        "TRACE_SCORE": QColor(get_chart_color(1)),
    }


_TRACE_COLORS = _build_trace_colors()
ARM = _TRACE_COLORS["ARM"]
BODY = _TRACE_COLORS["BODY"]
CHAIN = _TRACE_COLORS["CHAIN"]
GRID = _TRACE_COLORS["GRID"]
LEG = _TRACE_COLORS["LEG"]
SURFACE = _TRACE_COLORS["SURFACE"]
TRACE_BEST = _TRACE_COLORS["TRACE_BEST"]
TRACE_PARAM = _TRACE_COLORS["TRACE_PARAM"]
TRACE_SCORE = _TRACE_COLORS["TRACE_SCORE"]


def refresh_policy_trace_palette() -> None:
    """Rebind trace-canvas colours from the active theme palette."""
    global ARM, BODY, CHAIN, GRID, LEG, SURFACE, TRACE_BEST, TRACE_PARAM, TRACE_SCORE
    colors = _build_trace_colors()
    ARM = colors["ARM"]
    BODY = colors["BODY"]
    CHAIN = colors["CHAIN"]
    GRID = colors["GRID"]
    LEG = colors["LEG"]
    SURFACE = colors["SURFACE"]
    TRACE_BEST = colors["TRACE_BEST"]
    TRACE_PARAM = colors["TRACE_PARAM"]
    TRACE_SCORE = colors["TRACE_SCORE"]


class PolicyTraceCanvas(QWidget):
    """Compact plot of policy-search score and parameter traces."""

    #: Height of the strip reserved at the top for the legend so it never
    #: paints over the plotted series.
    _LEGEND_BAND_PX = 22

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(160)
        self._samples: tuple[CyclicPolicyTraceSample, ...] = ()
        self._series: dict[str, np.ndarray] = {}
        self._legend_visible = True

    def set_trace(self, samples: tuple[CyclicPolicyTraceSample, ...]) -> None:
        self._samples = samples
        self._series = self._build_series(samples)
        self.update()

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the inline legend and repaint."""
        self._legend_visible = bool(visible)
        self.update()

    def legend_visible(self) -> bool:
        """Return whether the inline legend is currently drawn."""
        return self._legend_visible

    def _top_margin(self) -> float:
        """Top inset for the plotted series, reserving room for the legend."""
        return float(self._LEGEND_BAND_PX if self._legend_visible else 8)

    def sample_count(self) -> int:
        return len(self._samples)

    def has_parameter_series(self, name: str) -> bool:
        return name in self._series and self._series[name].size > 0

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), SURFACE)
        painter.setPen(QPen(GRID, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if len(self._samples) < 2:
            return
        self._draw_normalized_series(painter, "best_score_m", TRACE_BEST, 3)
        self._draw_normalized_series(painter, "score_m", TRACE_SCORE, 2)
        self._draw_normalized_series(painter, "frequency_hz", TRACE_PARAM, 1)
        self._draw_normalized_series(painter, "hip_rate_amplitude_rad_s", ARM, 1)
        self._draw_normalized_series(painter, "torso_rate_amplitude_rad_s", BODY, 1)
        self._draw_normalized_series(painter, "knee_rate_ratio", LEG, 1)
        if self._legend_visible:
            self._draw_legend(painter)

    def _draw_normalized_series(
        self,
        painter: QPainter,
        key: str,
        color: QColor,
        width: int,
    ) -> None:
        values = self._series.get(key)
        if values is None or values.size < 2:
            return
        lower = float(np.min(values))
        upper = float(np.max(values))
        if np.isclose(lower, upper):
            normalized = np.full(values.shape, 0.5, dtype=np.float64)
        else:
            normalized = (values - lower) / (upper - lower)
        top = self._top_margin()
        bottom = self.height() - 8.0
        span = max(bottom - top, 1.0)
        points = [
            QPointF(
                8.0 + index * (self.width() - 16.0) / (values.size - 1),
                bottom - value * span,
            )
            for index, value in enumerate(normalized)
        ]
        painter.setPen(QPen(color, width))
        for start, end in pairwise(points):
            painter.drawLine(start, end)

    def _draw_legend(self, painter: QPainter) -> None:
        legend = (
            ("best", TRACE_BEST),
            ("score", TRACE_SCORE),
            ("freq", TRACE_PARAM),
            ("hip", ARM),
            ("torso", BODY),
            ("knee", LEG),
        )
        x = 8
        y = 16
        for label, color in legend:
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, y - 4, x + 12, y - 4)
            painter.setPen(QPen(color, 1))
            painter.drawText(x + 16, y, label)
            x += 54
        painter.setPen(QPen(CHAIN, 1))
        painter.drawText(max(8, self.width() - 64), self.height() - 8, "iteration")

    def _build_series(
        self,
        samples: tuple[CyclicPolicyTraceSample, ...],
    ) -> dict[str, np.ndarray]:
        return {
            "score_m": _trace_series(samples, lambda sample: sample.score_m),
            "best_score_m": _trace_series(samples, lambda sample: sample.best_score_m),
            "frequency_hz": _trace_series(
                samples, lambda sample: sample.parameters.frequency_hz
            ),
            "hip_rate_amplitude_rad_s": _trace_series(
                samples, lambda sample: sample.parameters.hip_rate_amplitude_rad_s
            ),
            "torso_rate_amplitude_rad_s": _trace_series(
                samples, lambda sample: sample.parameters.torso_rate_amplitude_rad_s
            ),
            "knee_rate_ratio": _trace_series(
                samples, lambda sample: sample.parameters.knee_rate_ratio
            ),
            "phase_rad": _trace_series(
                samples, lambda sample: sample.parameters.phase_rad
            ),
        }


def _trace_series(
    samples: tuple[CyclicPolicyTraceSample, ...],
    read_value: Callable[[CyclicPolicyTraceSample], float],
) -> np.ndarray:
    return np.asarray([read_value(sample) for sample in samples], dtype=np.float64)
