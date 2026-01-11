"""Tests for URDF builder handedness support.

Task 3.4: Handedness Integration tests.
"""

import pytest

from tools.urdf_generator.urdf_builder import Handedness, URDFBuilder


class TestHandednessEnum:
    """Tests for Handedness enumeration."""

    def test_right_value(self) -> None:
        """Right handedness should have correct value."""
        assert Handedness.RIGHT.value == "right"

    def test_left_value(self) -> None:
        """Left handedness should have correct value."""
        assert Handedness.LEFT.value == "left"


class TestHandednessConfiguration:
    """Tests for handedness configuration methods."""

    @pytest.fixture
    def builder(self) -> URDFBuilder:
        """Create a fresh URDFBuilder."""
        return URDFBuilder()

    def test_default_handedness_is_right(self, builder: URDFBuilder) -> None:
        """Default handedness should be RIGHT."""
        assert builder.get_handedness() == Handedness.RIGHT

    def test_set_handedness_left(self, builder: URDFBuilder) -> None:
        """set_handedness should update handedness to LEFT."""
        builder.set_handedness(Handedness.LEFT)

        assert builder.get_handedness() == Handedness.LEFT

    def test_set_handedness_right(self, builder: URDFBuilder) -> None:
        """set_handedness should update handedness to RIGHT."""
        builder.set_handedness(Handedness.LEFT)  # Start with left
        builder.set_handedness(Handedness.RIGHT)  # Change to right

        assert builder.get_handedness() == Handedness.RIGHT


class TestMirrorForHandedness:
    """Tests for mirror_for_handedness method."""

    @pytest.fixture
    def builder_with_segment(self) -> URDFBuilder:
        """Create URDFBuilder with a test segment."""
        builder = URDFBuilder()
        builder.add_segment(
            {
                "name": "right_arm",
                "geometry": {
                    "shape": "cylinder",
                    "dimensions": {"length": 0.5, "width": 0.1},
                    "position": {"x": 0.0, "y": 0.3, "z": 0.0},
                    "orientation": {"roll": 0, "pitch": 0, "yaw": 0},
                },
                "physics": {"mass": 2.0},
                "joint": {"type": "revolute", "axis": {"x": 0, "y": 1, "z": 0}},
            }
        )
        return builder

    def test_mirror_toggles_handedness(self, builder_with_segment: URDFBuilder) -> None:
        """mirror_for_handedness should toggle handedness from RIGHT to LEFT."""
        assert builder_with_segment.get_handedness() == Handedness.RIGHT

        builder_with_segment.mirror_for_handedness()

        assert builder_with_segment.get_handedness() == Handedness.LEFT

    def test_mirror_negates_y_position(self, builder_with_segment: URDFBuilder) -> None:
        """mirror_for_handedness should negate Y position."""
        original_y = builder_with_segment.segments[0]["geometry"]["position"]["y"]

        builder_with_segment.mirror_for_handedness()

        new_y = builder_with_segment.segments[0]["geometry"]["position"]["y"]
        assert new_y == -original_y

    def test_mirror_renames_right_to_left(
        self, builder_with_segment: URDFBuilder
    ) -> None:
        """mirror_for_handedness should rename right_ prefix to left_."""
        builder_with_segment.mirror_for_handedness()

        assert builder_with_segment.segments[0]["name"] == "left_arm"

    def test_mirror_renames_left_to_right(self) -> None:
        """mirror_for_handedness should rename left_ prefix to right_."""
        builder = URDFBuilder()
        builder.add_segment(
            {
                "name": "left_leg",
                "geometry": {
                    "shape": "box",
                    "dimensions": {"length": 0.5, "width": 0.1, "height": 0.1},
                    "position": {"x": 0, "y": -0.2, "z": 0},
                    "orientation": {"roll": 0, "pitch": 0, "yaw": 0},
                },
                "physics": {"mass": 3.0},
            }
        )

        builder.mirror_for_handedness()

        assert builder.segments[0]["name"] == "right_leg"

    def test_double_mirror_restores_original(
        self, builder_with_segment: URDFBuilder
    ) -> None:
        """Mirroring twice should restore original values."""
        original_y = builder_with_segment.segments[0]["geometry"]["position"]["y"]
        original_name = builder_with_segment.segments[0]["name"]

        builder_with_segment.mirror_for_handedness()
        builder_with_segment.mirror_for_handedness()

        assert (
            builder_with_segment.segments[0]["geometry"]["position"]["y"] == original_y
        )
        assert builder_with_segment.segments[0]["name"] == original_name
        assert builder_with_segment.get_handedness() == Handedness.RIGHT


class TestGetMirroredUrdf:
    """Tests for get_mirrored_urdf method."""

    @pytest.fixture
    def builder_with_segment(self) -> URDFBuilder:
        """Create URDFBuilder with a test segment."""
        builder = URDFBuilder()
        builder.add_segment(
            {
                "name": "club_head",
                "geometry": {
                    "shape": "box",
                    "dimensions": {"length": 0.1, "width": 0.05, "height": 0.05},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"roll": 0, "pitch": 0, "yaw": 0},
                },
                "physics": {"mass": 0.3},
            }
        )
        return builder

    def test_same_handedness_returns_normal_urdf(
        self, builder_with_segment: URDFBuilder
    ) -> None:
        """get_mirrored_urdf with same handedness should return normal URDF."""
        normal_urdf = builder_with_segment.get_urdf()
        mirrored_urdf = builder_with_segment.get_mirrored_urdf(Handedness.RIGHT)

        assert mirrored_urdf == normal_urdf

    def test_different_handedness_does_not_modify_state(
        self, builder_with_segment: URDFBuilder
    ) -> None:
        """get_mirrored_urdf should not modify internal state."""
        original_handedness = builder_with_segment.get_handedness()
        original_name = builder_with_segment.segments[0]["name"]

        _mirrored = builder_with_segment.get_mirrored_urdf(Handedness.LEFT)

        # State should be unchanged
        assert builder_with_segment.get_handedness() == original_handedness
        assert builder_with_segment.segments[0]["name"] == original_name

    def test_mirrored_urdf_produces_valid_xml(
        self, builder_with_segment: URDFBuilder
    ) -> None:
        """get_mirrored_urdf should produce valid URDF XML."""
        mirrored = builder_with_segment.get_mirrored_urdf(Handedness.LEFT)

        # Should be parseable XML with robot element
        assert "<?xml" in mirrored
        assert "<robot" in mirrored
        assert "</robot>" in mirrored
