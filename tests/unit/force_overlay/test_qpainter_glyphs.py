"""Unit tests for QPainter 2D force/torque glyph renderer (ADR-0052, #11292)."""

from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QImage, QPainter

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    GlyphSet,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.qpainter_glyphs import (
    draw_glyphs_2d,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


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
