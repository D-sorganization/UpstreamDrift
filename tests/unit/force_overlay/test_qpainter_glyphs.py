"""Unit tests for QPainter 2D force/torque glyph renderer (ADR-0052, #11292)."""

from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QImage, QPainter

pytestmark = [pytest.mark.unit]

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.qpainter_glyphs import (
    draw_glyphs_2d,
)


@pytest.fixture
def sample_glyphs() -> GlyphSet:
    frame = ForceTorqueFrame(
        time_s=0.5,
        engine="mujoco",
        wrenches=(
            OverlayWrench(
                label="joint:lead_wrist",
                kind=WrenchKind.JOINT_REACTION,
                body="wrist",
                torque_nm=(0.0, 0.0, 10.0),
                force_n=(100.0, 50.0, 0.0),
                point_m=(0.0, 0.0, 0.0),
                source="sim",
            ),
        ),
    )
    return build_glyphs(frame, style=ForceGlyphStyle())


def _project_2d(pt: tuple[float, ...]) -> QPointF:
    # Maps meters to screen pixels (center at 100, 100, 200 px/m)
    x, y = pt[0], pt[1]
    return QPointF(100.0 + x * 200.0, 100.0 - y * 200.0)


def test_draw_glyphs_2d_renders_shaft_and_head_pixels(sample_glyphs: GlyphSet) -> None:
    img = QImage(200, 200, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))  # transparent

    painter = QPainter(img)
    try:
        draw_glyphs_2d(painter, _project_2d, sample_glyphs, px_width=3.0, halo=True)
    finally:
        painter.end()

    # Verify non-transparent pixels exist
    colored = []
    for y in range(img.height()):
        for x in range(img.width()):
            if img.pixelColor(x, y).alpha() > 0:
                colored.append((x, y))

    assert len(colored) > 20


def test_draw_glyphs_2d_empty_glyphs_renders_nothing() -> None:
    img = QImage(100, 100, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))

    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=())
    glyphs = build_glyphs(frame, style=ForceGlyphStyle())

    painter = QPainter(img)
    try:
        draw_glyphs_2d(painter, _project_2d, glyphs)
    finally:
        painter.end()

    for y in range(img.height()):
        for x in range(img.width()):
            assert img.pixelColor(x, y).alpha() == 0


def test_draw_glyphs_2d_projector_signatures(sample_glyphs: GlyphSet) -> None:
    img = QImage(200, 200, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))

    # Projector returning tuple instead of QPointF
    def tuple_projector(pt: tuple[float, ...]) -> tuple[float, float]:
        return (100.0 + pt[0] * 50.0, 100.0 - pt[1] * 50.0)

    painter = QPainter(img)
    try:
        draw_glyphs_2d(painter, tuple_projector, sample_glyphs)
    finally:
        painter.end()

    has_colored = any(
        img.pixelColor(x, y).alpha() > 0
        for y in range(img.height())
        for x in range(img.width())
    )
    assert has_colored


class _CountingPainterProxy:
    """Forwards every call to a real QPainter, counting ``drawPolygon`` calls.

    Used to assert draw-call structure (ADR-0052's double chevron adds exactly
    one extra arrowhead polygon) without relying on pixel goldens.
    """

    def __init__(self, painter: QPainter) -> None:
        self._painter = painter
        self.polygon_calls = 0

    def drawPolygon(self, *args: object, **kwargs: object) -> None:
        self.polygon_calls += 1
        self._painter.drawPolygon(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._painter, name)


def _make_arrow_glyph(*, clamped: bool) -> ArrowGlyph:
    from src.shared.python.force_overlay.contracts import WrenchKind

    return ArrowGlyph(
        label="joint:lead_wrist",
        kind=WrenchKind.JOINT_REACTION,
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(1.0, 0.0, 0.0),
        head_base_m=(0.8, 0.0, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(1.0, 0.0, 0.0, 1.0),
        magnitude=500.0,
        units="N",
        clamped=clamped,
    )


def _polygon_call_count(glyphs: GlyphSet) -> int:
    img = QImage(200, 200, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    proxy = _CountingPainterProxy(painter)
    try:
        draw_glyphs_2d(proxy, _project_2d, glyphs, px_width=3.0, halo=False)
    finally:
        painter.end()
    return proxy.polygon_calls


def test_draw_glyphs_2d_clamped_arrow_draws_one_extra_chevron_polygon() -> None:
    """ADR-0052: a clamped arrow draws exactly one extra arrowhead polygon."""
    unclamped = GlyphSet(
        time_s=0.0,
        arrows=(_make_arrow_glyph(clamped=False),),
        torque_arcs=(),
        legend=LegendSpec(),
    )
    clamped = GlyphSet(
        time_s=0.0,
        arrows=(_make_arrow_glyph(clamped=True),),
        torque_arcs=(),
        legend=LegendSpec(clamped_labels=("joint:lead_wrist",)),
    )

    unclamped_calls = _polygon_call_count(unclamped)
    clamped_calls = _polygon_call_count(clamped)

    assert clamped_calls == unclamped_calls + 1
