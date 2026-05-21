"""Comprehensive tests for programmatic_pid.types module.

Tests Point, BBox, ValidationIssue, SpecValidationError, LayoutRegions, TextConfig.
"""

from __future__ import annotations

import pytest
from programmatic_pid.types import (
    BBox,
    LayoutRegions,
    Point,
    SpecValidationError,
    TextConfig,
    ValidationIssue,
)


class TestPoint:
    def test_creation(self):
        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0

    def test_named_tuple_indexing(self):
        p = Point(3.0, 4.0)
        assert p[0] == 3.0
        assert p[1] == 4.0

    def test_equality(self):
        assert Point(1.0, 2.0) == Point(1.0, 2.0)
        assert Point(1.0, 2.0) != Point(2.0, 1.0)


class TestBBox:
    def test_creation(self):
        b = BBox(0.0, 0.0, 10.0, 5.0)
        assert b.x_min == 0.0
        assert b.y_min == 0.0
        assert b.x_max == 10.0
        assert b.y_max == 5.0

    def test_width_and_height(self):
        b = BBox(1.0, 2.0, 5.0, 8.0)
        assert b.width == pytest.approx(4.0)
        assert b.height == pytest.approx(6.0)

    def test_center(self):
        b = BBox(0.0, 0.0, 4.0, 6.0)
        c = b.center
        assert c.x == pytest.approx(2.0)
        assert c.y == pytest.approx(3.0)

    def test_contains_point_inside(self):
        b = BBox(0.0, 0.0, 10.0, 10.0)
        assert b.contains_point(Point(5.0, 5.0))

    def test_contains_point_boundary(self):
        b = BBox(0.0, 0.0, 10.0, 10.0)
        assert b.contains_point(Point(0.0, 0.0))
        assert b.contains_point(Point(10.0, 10.0))

    def test_contains_point_outside(self):
        b = BBox(0.0, 0.0, 10.0, 10.0)
        assert not b.contains_point(Point(11.0, 5.0))

    def test_overlaps_true(self):
        a = BBox(0.0, 0.0, 5.0, 5.0)
        b = BBox(3.0, 3.0, 8.0, 8.0)
        assert a.overlaps(b)

    def test_overlaps_false(self):
        a = BBox(0.0, 0.0, 3.0, 3.0)
        b = BBox(4.0, 4.0, 8.0, 8.0)
        assert not a.overlaps(b)

    def test_overlaps_with_pad(self):
        a = BBox(0.0, 0.0, 3.0, 3.0)
        b = BBox(4.0, 4.0, 8.0, 8.0)
        # Without pad they don't overlap, but with pad=2 they should
        assert a.overlaps(b, pad=2.0)

    def test_union(self):
        a = BBox(0.0, 0.0, 3.0, 3.0)
        b = BBox(2.0, 2.0, 6.0, 6.0)
        u = a.union(b)
        assert u.x_min == 0.0
        assert u.y_min == 0.0
        assert u.x_max == 6.0
        assert u.y_max == 6.0

    def test_expanded(self):
        b = BBox(2.0, 2.0, 8.0, 8.0)
        e = b.expanded(1.0)
        assert e.x_min == pytest.approx(1.0)
        assert e.y_min == pytest.approx(1.0)
        assert e.x_max == pytest.approx(9.0)
        assert e.y_max == pytest.approx(9.0)


class TestValidationIssue:
    def test_creation_defaults(self):
        vi = ValidationIssue("some.path", "A message")
        assert vi.path == "some.path"
        assert vi.message == "A message"
        assert vi.severity == "error"

    def test_warning_severity(self):
        vi = ValidationIssue("field", "Something", severity="warning")
        assert vi.severity == "warning"

    def test_to_dict(self):
        vi = ValidationIssue("a.b", "msg", severity="error")
        d = vi.to_dict()
        assert d == {"path": "a.b", "message": "msg", "severity": "error"}


class TestSpecValidationError:
    def test_is_value_error(self):
        err = SpecValidationError("bad spec")
        assert isinstance(err, ValueError)
        assert "bad spec" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(SpecValidationError, match="invalid"):
            raise SpecValidationError("invalid spec")


class TestLayoutRegions:
    def test_creation(self):
        bbox = BBox(0.0, 0.0, 100.0, 100.0)
        lr = LayoutRegions(
            layout_cfg={"key": "val"},
            equipment_bbox=bbox,
            canvas_bbox=bbox,
            panels={"left": (0.0, 0.0, 10.0, 10.0)},
        )
        assert lr.layout_cfg == {"key": "val"}
        assert lr.equipment_bbox == bbox


class TestTextConfig:
    def test_creation(self):
        tc = TextConfig(
            title_height=14.0,
            subtitle_height=12.0,
            body_height=10.0,
            small_height=8.0,
        )
        assert tc.title_height == 14.0
        assert tc.small_height == 8.0
