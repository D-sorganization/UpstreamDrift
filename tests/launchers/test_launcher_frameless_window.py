"""Frameless launcher window resize helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytestmark = pytest.mark.unit

from PyQt6.QtCore import QPoint, QRect, Qt  # noqa: E402

from src.launchers.launcher_frameless_window import (  # noqa: E402
    ResizeEdge,
    cursor_shape_for_edge,
    resize_edge_at,
    resized_geometry,
)


def test_resize_edge_at_detects_edges_and_corners() -> None:
    assert resize_edge_at(2, 2, width=200, height=100) is ResizeEdge.TOP_LEFT
    assert resize_edge_at(198, 2, width=200, height=100) is ResizeEdge.TOP_RIGHT
    assert resize_edge_at(2, 98, width=200, height=100) is ResizeEdge.BOTTOM_LEFT
    assert resize_edge_at(198, 98, width=200, height=100) is ResizeEdge.BOTTOM_RIGHT
    assert resize_edge_at(2, 50, width=200, height=100) is ResizeEdge.LEFT
    assert resize_edge_at(198, 50, width=200, height=100) is ResizeEdge.RIGHT
    assert resize_edge_at(100, 2, width=200, height=100) is ResizeEdge.TOP
    assert resize_edge_at(100, 98, width=200, height=100) is ResizeEdge.BOTTOM
    assert resize_edge_at(100, 50, width=200, height=100) is None


def test_resize_edge_at_ignores_positions_outside_window() -> None:
    assert resize_edge_at(-1, 4, width=200, height=100) is None
    assert resize_edge_at(201, 4, width=200, height=100) is None
    assert resize_edge_at(4, -1, width=200, height=100) is None
    assert resize_edge_at(4, 101, width=200, height=100) is None


def test_cursor_shape_for_edge_matches_existing_resize_cursors() -> None:
    assert cursor_shape_for_edge(ResizeEdge.TOP_LEFT) is Qt.CursorShape.SizeFDiagCursor
    assert (
        cursor_shape_for_edge(ResizeEdge.BOTTOM_RIGHT) is Qt.CursorShape.SizeFDiagCursor
    )
    assert cursor_shape_for_edge(ResizeEdge.TOP_RIGHT) is Qt.CursorShape.SizeBDiagCursor
    assert (
        cursor_shape_for_edge(ResizeEdge.BOTTOM_LEFT) is Qt.CursorShape.SizeBDiagCursor
    )
    assert cursor_shape_for_edge(ResizeEdge.LEFT) is Qt.CursorShape.SizeHorCursor
    assert cursor_shape_for_edge(ResizeEdge.RIGHT) is Qt.CursorShape.SizeHorCursor
    assert cursor_shape_for_edge(ResizeEdge.TOP) is Qt.CursorShape.SizeVerCursor
    assert cursor_shape_for_edge(ResizeEdge.BOTTOM) is Qt.CursorShape.SizeVerCursor


def test_resized_geometry_applies_deltas_for_corner_drag() -> None:
    resized = resized_geometry(
        QRect(100, 100, 300, 200),
        ResizeEdge.BOTTOM_RIGHT,
        QPoint(20, 30),
        minimum_width=200,
        minimum_height=150,
    )

    assert resized == QRect(100, 100, 320, 230)


def test_resized_geometry_rejects_minimum_size_violation() -> None:
    resized = resized_geometry(
        QRect(100, 100, 300, 200),
        ResizeEdge.LEFT,
        QPoint(250, 0),
        minimum_width=200,
        minimum_height=150,
    )

    assert resized is None
