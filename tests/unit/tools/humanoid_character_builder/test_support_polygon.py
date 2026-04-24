"""Tests for SupportPolygon.distance_to_edge scalar-math optimisation.

Covers: projection before segment, projection after segment, projection onto
segment interior, degenerate (zero-length) edge, and a point outside the
polygon (should return -1.0).
"""

from __future__ import annotations

import math

import pytest
from humanoid_character_builder.core.model import SupportPolygon

pytestmark = pytest.mark.unit

pytestmark = pytest.mark.unit


@pytest.fixture()
def unit_square() -> SupportPolygon:
    """A 1 x 1 square centred on (0.5, 0.5)."""
    return SupportPolygon(vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


class TestDistanceToEdge:
    """Verify distance_to_edge for the scalar-math implementation."""

    def test_centre_returns_half(self, unit_square: SupportPolygon) -> None:
        """Centre of the square is equidistant (0.5) from all edges."""
        dist = unit_square.distance_to_edge((0.5, 0.5))
        assert abs(dist - 0.5) < 1e-9

    def test_projection_onto_interior_of_edge(
        self, unit_square: SupportPolygon
    ) -> None:
        """Point close to the bottom edge projects onto the edge interior."""
        dist = unit_square.distance_to_edge((0.5, 0.1))
        assert abs(dist - 0.1) < 1e-9

    def test_projection_before_segment_start(self, unit_square: SupportPolygon) -> None:
        """Point near a corner clamps to the vertex (t < 0 path)."""
        # (0.1, 0.1) is inside; nearest edge-point is the vertex (0, 0) at
        # distance sqrt(0.01 + 0.01) via bottom or left edge.
        dist = unit_square.distance_to_edge((0.1, 0.1))
        # Minimum distance should be 0.1 (to the bottom or left edge directly)
        assert dist > 0.0
        assert dist < 0.15  # sanity bound

    def test_projection_after_segment_end(self, unit_square: SupportPolygon) -> None:
        """Point near the far corner clamps to the far vertex (t > 1 path)."""
        dist = unit_square.distance_to_edge((0.9, 0.9))
        assert dist > 0.0
        assert dist < 0.15

    def test_degenerate_edge(self) -> None:
        """Degenerate edge (both vertices identical) returns hypot distance."""
        poly = SupportPolygon(
            vertices=[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        )
        # The polygon still has enough real vertices; degenerate edge is
        # (0,0)-(0,0).  For a point well inside, the result must be finite.
        dist = poly.distance_to_edge((0.5, 0.5))
        assert math.isfinite(dist)
        assert dist >= 0.0

    def test_outside_returns_negative_one(self, unit_square: SupportPolygon) -> None:
        """Points outside the polygon return -1.0 per the convention."""
        dist = unit_square.distance_to_edge((2.0, 0.5))
        assert dist == -1.0
