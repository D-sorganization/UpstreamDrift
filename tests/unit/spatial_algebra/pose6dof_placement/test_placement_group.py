"""
Unit tests for 6DOF positioning module.

Tests for Pose6DOF, Transform6DOF, and EntityPlacement classes
following TDD principles - tests written first.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.pose6dof import (
    EntityPlacement,
    PlacementGroup,
    Pose6DOF,
    Transform6DOF,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    quaternion_multiply,
    quaternion_to_euler,
    rotation_matrix_to_euler,
)


class TestPlacementGroup:
    """Tests for managing groups of entity placements."""

    def test_create_empty_group(self) -> None:
        """Test creating an empty placement group."""
        group = PlacementGroup()
        assert len(group) == 0

    def test_add_entities(self) -> None:
        """Test adding entities to group."""
        group = PlacementGroup()
        group.add(EntityPlacement(name="offense_1"))
        group.add(EntityPlacement(name="offense_2"))
        assert len(group) == 2

    def test_get_entity_by_name(self) -> None:
        """Test retrieving entity by name."""
        group = PlacementGroup()
        entity = EntityPlacement(name="offense_1")
        entity.move_to(5, 5, 0)
        group.add(entity)

        retrieved = group.get("offense_1")
        assert retrieved is not None
        np.testing.assert_allclose(retrieved.pose.position, [5, 5, 0], atol=1e-10)

    def test_remove_entity(self) -> None:
        """Test removing entity from group."""
        group = PlacementGroup()
        group.add(EntityPlacement(name="offense_1"))
        group.add(EntityPlacement(name="offense_2"))
        group.remove("offense_1")

        assert len(group) == 1
        assert group.get("offense_1") is None
        assert group.get("offense_2") is not None

    def test_iterate_entities(self) -> None:
        """Test iterating over entities."""
        group = PlacementGroup()
        group.add(EntityPlacement(name="a"))
        group.add(EntityPlacement(name="b"))

        names = [e.name for e in group]
        assert "a" in names
        assert "b" in names

    def test_move_all_entities(self) -> None:
        """Test moving all entities by offset."""
        group = PlacementGroup()
        e1 = EntityPlacement(name="a")
        e1.move_to(0, 0, 0)
        e2 = EntityPlacement(name="b")
        e2.move_to(1, 1, 0)
        group.add(e1)
        group.add(e2)

        group.translate_all([10, 0, 0])

        entity_a = group.get("a")
        entity_b = group.get("b")
        assert entity_a is not None
        assert entity_b is not None
        np.testing.assert_allclose(entity_a.pose.position, [10, 0, 0], atol=1e-10)
        np.testing.assert_allclose(entity_b.pose.position, [11, 1, 0], atol=1e-10)

    def test_rotate_group_around_point(self) -> None:
        """Test rotating entire group around a point."""
        group = PlacementGroup()
        e = EntityPlacement(name="a")
        e.move_to(1, 0, 0)
        group.add(e)

        # Rotate 90° around origin about z-axis
        group.rotate_around_point([0, 0, 0], axis=[0, 0, 1], angle=np.pi / 2)

        rotated_a = group.get("a")
        assert rotated_a is not None
        np.testing.assert_allclose(rotated_a.pose.position, [0, 1, 0], atol=1e-10)

    def test_get_centroid(self) -> None:
        """Test calculating group centroid."""
        group = PlacementGroup()
        e1 = EntityPlacement(name="a")
        e1.move_to(0, 0, 0)
        e2 = EntityPlacement(name="b")
        e2.move_to(2, 2, 0)
        group.add(e1)
        group.add(e2)

        centroid = group.centroid
        np.testing.assert_allclose(centroid, [1, 1, 0], atol=1e-10)

    def test_get_bounding_box(self) -> None:
        """Test calculating axis-aligned bounding box."""
        group = PlacementGroup()
        e1 = EntityPlacement(name="a")
        e1.move_to(0, 0, 0)
        e2 = EntityPlacement(name="b")
        e2.move_to(10, 5, 2)
        group.add(e1)
        group.add(e2)

        bbox = group.bounding_box
        np.testing.assert_allclose(bbox["min"], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(bbox["max"], [10, 5, 2], atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
