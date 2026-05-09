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


class TestEntityPlacement:
    """Tests for EntityPlacement - placing models/offenses in simulation space."""

    def test_create_entity_at_origin(self) -> None:
        """Test creating an entity at the origin."""
        entity = EntityPlacement(name="offense_1")
        assert entity.name == "offense_1"
        np.testing.assert_allclose(entity.pose.position, [0, 0, 0], atol=1e-10)

    def test_create_entity_with_pose(self) -> None:
        """Test creating an entity with initial pose."""
        pose = Pose6DOF(position=[5, 10, 0], euler_angles=[0, 0, np.pi / 4])
        entity = EntityPlacement(name="offense_2", pose=pose)
        np.testing.assert_allclose(entity.pose.position, [5, 10, 0], atol=1e-10)
        assert entity.pose.yaw == pytest.approx(np.pi / 4)

    def test_move_entity_to_position(self) -> None:
        """Test moving entity to absolute position."""
        entity = EntityPlacement(name="offense")
        entity.move_to(5, 10, 2)
        np.testing.assert_allclose(entity.pose.position, [5, 10, 2], atol=1e-10)

    def test_move_entity_by_offset(self) -> None:
        """Test moving entity by relative offset."""
        entity = EntityPlacement(name="offense")
        entity.move_to(1, 1, 1)
        entity.move_by(dx=1, dy=2, dz=3)
        np.testing.assert_allclose(entity.pose.position, [2, 3, 4], atol=1e-10)

    def test_rotate_entity_euler(self) -> None:
        """Test rotating entity using euler angles."""
        entity = EntityPlacement(name="offense")
        entity.rotate_euler(roll=0, pitch=0, yaw=np.pi / 2)
        assert entity.pose.yaw == pytest.approx(np.pi / 2)

    def test_set_yaw_directly(self) -> None:
        """Test setting yaw (heading) directly."""
        entity = EntityPlacement(name="offense")
        entity.set_yaw(np.pi)
        assert entity.pose.yaw == pytest.approx(np.pi)

    def test_rotate_entity_about_axis(self) -> None:
        """Test rotating entity about arbitrary axis."""
        entity = EntityPlacement(name="offense")
        entity.rotate_axis([0, 0, 1], np.pi / 2)
        assert entity.pose.yaw == pytest.approx(np.pi / 2)

    def test_look_at_point(self) -> None:
        """Test orienting entity to look at a point."""
        entity = EntityPlacement(name="offense")
        entity.move_to(0, 0, 0)
        entity.look_at([1, 0, 0])  # Look along +x

        # Entity's forward direction should point toward target
        forward = entity.forward_vector
        np.testing.assert_allclose(forward, [1, 0, 0], atol=1e-10)

    def test_forward_right_up_vectors(self) -> None:
        """Test getting local coordinate frame vectors."""
        entity = EntityPlacement(name="offense")
        # No rotation - default frame
        np.testing.assert_allclose(entity.forward_vector, [1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(entity.right_vector, [0, 1, 0], atol=1e-10)
        np.testing.assert_allclose(entity.up_vector, [0, 0, 1], atol=1e-10)

    def test_forward_vector_after_rotation(self) -> None:
        """Test forward vector after yaw rotation."""
        entity = EntityPlacement(name="offense")
        entity.set_yaw(np.pi / 2)  # 90° left turn
        # Forward should now point along +y
        np.testing.assert_allclose(entity.forward_vector, [0, 1, 0], atol=1e-10)

    def test_distance_to_point(self) -> None:
        """Test calculating distance to a point."""
        entity = EntityPlacement(name="offense")
        entity.move_to(0, 0, 0)
        dist = entity.distance_to([3, 4, 0])
        assert dist == pytest.approx(5.0)

    def test_distance_to_entity(self) -> None:
        """Test calculating distance to another entity."""
        e1 = EntityPlacement(name="offense_1")
        e2 = EntityPlacement(name="offense_2")
        e1.move_to(0, 0, 0)
        e2.move_to(3, 4, 0)
        assert e1.distance_to_entity(e2) == pytest.approx(5.0)

    def test_entity_metadata(self) -> None:
        """Test entity metadata storage."""
        entity = EntityPlacement(name="offense", metadata={"type": "offensive_unit"})
        assert entity.metadata["type"] == "offensive_unit"

    def test_entity_copy(self) -> None:
        """Test entity deep copy."""
        original = EntityPlacement(name="offense")
        original.move_to(5, 5, 5)
        copied = original.copy()

        assert copied.name == "offense"
        np.testing.assert_allclose(copied.pose.position, [5, 5, 5], atol=1e-10)

        # Modifying copy shouldn't affect original
        copied.move_to(0, 0, 0)
        np.testing.assert_allclose(original.pose.position, [5, 5, 5], atol=1e-10)

    def test_entity_to_transform(self) -> None:
        """Test converting entity placement to Transform6DOF."""
        entity = EntityPlacement(name="offense")
        entity.move_to(1, 2, 3)
        entity.set_yaw(np.pi / 4)

        transform = entity.to_transform()
        np.testing.assert_allclose(transform.translation, [1, 2, 3], atol=1e-10)

    def test_entity_from_transform(self) -> None:
        """Test creating entity from Transform6DOF."""
        T = Transform6DOF.from_translation([10, 20, 30])
        entity = EntityPlacement.from_transform("offense", T)
        np.testing.assert_allclose(entity.pose.position, [10, 20, 30], atol=1e-10)

    def test_entity_serialize_deserialize(self) -> None:
        """Test serialization to/from dict."""
        entity = EntityPlacement(name="offense", metadata={"score": 100})
        entity.move_to(1, 2, 3)
        entity.rotate_euler(roll=0.1, pitch=0.2, yaw=0.3)

        data = entity.to_dict()
        restored = EntityPlacement.from_dict(data)

        assert restored.name == entity.name
        np.testing.assert_allclose(restored.pose.position, entity.pose.position)
        np.testing.assert_allclose(restored.pose.euler_angles, entity.pose.euler_angles)
        assert restored.metadata["score"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
