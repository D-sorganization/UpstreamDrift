# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the force/torque vector overlay (gui/vector_overlay.py)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QImage, QPainter

from movement_optimizer.gui.vector_overlay import (
    ComMarker,
    ForceArrow,
    OverlayScene,
    TorqueArc,
    VectorStyle,
    auto_scale_factor,
    draw_com_markers,
    draw_force_arrows,
    draw_overlay_scene,
    draw_torque_arcs,
)

_SIZE = 120
_MID = _SIZE // 2


def _flipping_projector(scale: float = 20.0) -> Callable[[tuple[float, float]], QPointF]:
    # Mimics the canvas projector's Y handling: larger world-y -> smaller screen-y.
    def _project(point: tuple[float, float]) -> QPointF:
        x, y = point
        return QPointF(_MID + scale * x, _MID - scale * y)

    return _project


def _render(draw: Callable[[QPainter], None]) -> QImage:
    image = QImage(_SIZE, _SIZE, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))  # fully transparent (fill(0) would be opaque black)
    painter = QPainter(image)
    try:
        draw(painter)
    finally:
        # Always end the painter -- even when ``draw`` raises (the negative tests
        # expect a ValueError). A QPainter left active on an image segfaults Qt
        # when garbage-collected on Linux/CI.
        painter.end()
    return image


def _colored_pixels(image: QImage) -> list[tuple[int, int]]:
    # Use pixelColor (not QColor(pixel)) -- the QColor(QRgb) constructor forces
    # alpha to 255, which would report every pixel as opaque.
    found = []
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() > 0:
                found.append((x, y))
    return found


@pytest.fixture()
def style() -> VectorStyle:
    return VectorStyle(color=QColor("#ff0000"), width=2, head_px=8.0)


def test_auto_scale_factor_empty_returns_one() -> None:
    assert auto_scale_factor([], 5.0) == 1.0


def test_auto_scale_factor_zero_vectors_returns_one(style: VectorStyle) -> None:
    arrows = [ForceArrow((0.0, 0.0), (0.0, 0.0), style)]
    assert auto_scale_factor(arrows, 5.0) == 1.0


def test_auto_scale_factor_scales_largest_to_target(style: VectorStyle) -> None:
    arrows = [
        ForceArrow((0.0, 0.0), (3.0, 4.0), style),  # magnitude 5
        ForceArrow((0.0, 0.0), (1.0, 0.0), style),
    ]
    assert auto_scale_factor(arrows, 10.0) == pytest.approx(2.0)


def test_auto_scale_factor_rejects_nonpositive_target(style: VectorStyle) -> None:
    with pytest.raises(ValueError, match="target_world_len"):
        auto_scale_factor([], 0.0)


def test_draw_force_arrows_renders_pixels(qapp, style: VectorStyle) -> None:
    arrows = [ForceArrow((0.0, 0.0), (1.0, 0.0), style)]
    image = _render(lambda p: draw_force_arrows(p, _flipping_projector(), arrows, scale=1.0))
    assert _colored_pixels(image)


def test_draw_force_arrows_respects_projector_y_flip(qapp, style: VectorStyle) -> None:
    # A +y world vector must render ABOVE the origin (smaller screen-y).
    up = [ForceArrow((0.0, 0.0), (0.0, 1.0), style)]
    image = _render(lambda p: draw_force_arrows(p, _flipping_projector(), up, scale=1.0))
    ys = [y for _x, y in _colored_pixels(image)]
    assert min(ys) < _MID  # reached above the origin row


def test_draw_force_arrows_rejects_nonpositive_scale(qapp, style: VectorStyle) -> None:
    arrows = [ForceArrow((0.0, 0.0), (1.0, 0.0), style)]
    with pytest.raises(ValueError, match="scale"):
        _render(lambda p: draw_force_arrows(p, _flipping_projector(), arrows, scale=0.0))


def test_draw_force_arrows_rejects_nonfinite_scale(qapp, style: VectorStyle) -> None:
    arrows = [ForceArrow((0.0, 0.0), (1.0, 0.0), style)]
    with pytest.raises(ValueError, match="scale"):
        _render(lambda p: draw_force_arrows(p, _flipping_projector(), arrows, scale=float("inf")))


def test_draw_force_arrows_skips_zero_length(qapp, style: VectorStyle) -> None:
    arrows = [ForceArrow((0.0, 0.0), (0.0, 0.0), style)]
    image = _render(lambda p: draw_force_arrows(p, _flipping_projector(), arrows, scale=1.0))
    assert not _colored_pixels(image)  # no shaft, no head


def test_draw_torque_arcs_renders(qapp, style: VectorStyle) -> None:
    arcs = [TorqueArc((0.0, 0.0), 12.0, style), TorqueArc((0.5, 0.0), -8.0, style)]
    image = _render(lambda p: draw_torque_arcs(p, _flipping_projector(), arcs, reference_nm=12.0))
    assert _colored_pixels(image)


def test_draw_torque_arcs_rejects_nonpositive_reference(qapp, style: VectorStyle) -> None:
    arcs = [TorqueArc((0.0, 0.0), 1.0, style)]
    with pytest.raises(ValueError, match="reference_nm"):
        _render(lambda p: draw_torque_arcs(p, _flipping_projector(), arcs, reference_nm=0.0))


def test_draw_com_markers_renders(qapp, style: VectorStyle) -> None:
    markers = [ComMarker((0.0, 0.0), style)]
    image = _render(lambda p: draw_com_markers(p, _flipping_projector(), markers))
    assert _colored_pixels(image)


def test_draw_overlay_scene_combines_primitives(qapp, style: VectorStyle) -> None:
    scene = OverlayScene(
        arrows=(ForceArrow((0.0, 0.0), (1.0, 0.0), style),),
        torque_arcs=(TorqueArc((0.0, 0.0), 5.0, style),),
        com_markers=(ComMarker((0.0, 0.0), style),),
    )
    image = _render(
        lambda p: draw_overlay_scene(
            p, _flipping_projector(), scene, arrow_scale=1.0, torque_reference_nm=5.0
        )
    )
    assert _colored_pixels(image)


def test_motion_canvas_set_overlays_paints(qapp, style: VectorStyle) -> None:
    from movement_optimizer.gui.motion_tabs import MotionCanvas

    canvas = MotionCanvas()
    canvas.resize(200, 200)
    canvas.set_scene([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
    scene = OverlayScene(arrows=(ForceArrow((0.0, 1.0), (0.0, 1.0), style),))
    canvas.set_overlays(scene)
    pixmap = canvas.grab()
    assert not pixmap.isNull()
    assert pixmap.width() == 200


def test_draw_arrowhead_skips_zero_length(qapp, style: VectorStyle) -> None:
    from movement_optimizer.gui.vector_overlay import _draw_arrowhead

    image = QImage(_SIZE, _SIZE, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    same = QPointF(10.0, 10.0)
    _draw_arrowhead(painter, same, same, style.head_px)  # must not raise (early return)
    painter.end()
    assert not _colored_pixels(image)
