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


class TestPose6DOF:
    """Tests for Pose6DOF class - intuitive 6DOF positioning."""

    def test_create_identity_pose(self) -> None:
        """Test creating an identity pose at origin with no rotation."""
        pose = Pose6DOF()
        np.testing.assert_allclose(pose.position, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(pose.euler_angles, [0, 0, 0], atol=1e-10)

    def test_create_pose_with_position(self) -> None:
        """Test creating a pose with specified position."""
        pose = Pose6DOF(position=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(pose.position, [1, 2, 3], atol=1e-10)
        np.testing.assert_allclose(pose.euler_angles, [0, 0, 0], atol=1e-10)

    def test_create_pose_with_euler_angles(self) -> None:
        """Test creating a pose with roll, pitch, yaw."""
        roll, pitch, yaw = np.pi / 6, np.pi / 4, np.pi / 3
        pose = Pose6DOF(euler_angles=[roll, pitch, yaw])
        np.testing.assert_allclose(pose.position, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(pose.euler_angles, [roll, pitch, yaw], atol=1e-10)

    def test_create_pose_with_quaternion(self) -> None:
        """Test creating a pose from quaternion."""
        # Quaternion for 90° rotation about z-axis: [w, x, y, z]
        quat = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        pose = Pose6DOF.from_quaternion([0, 0, 0], quat)
        # Should have yaw of 90°
        np.testing.assert_allclose(pose.euler_angles[2], np.pi / 2, atol=1e-6)

    def test_pose_to_quaternion_roundtrip(self) -> None:
        """Test euler -> quaternion -> euler roundtrip."""
        original = Pose6DOF(euler_angles=[0.1, 0.2, 0.3])
        quat = original.to_quaternion()
        reconstructed = Pose6DOF.from_quaternion([0, 0, 0], quat)
        np.testing.assert_allclose(
            original.euler_angles, reconstructed.euler_angles, atol=1e-10
        )

    def test_pose_rotation_matrix(self) -> None:
        """Test conversion to 3x3 rotation matrix."""
        # 90° rotation about z-axis
        pose = Pose6DOF(euler_angles=[0, 0, np.pi / 2])
        R = pose.rotation_matrix
        assert R.shape == (3, 3)

        # Check rotation is orthogonal
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

        # Check specific rotation effect
        x_axis = np.array([1, 0, 0])
        rotated = R @ x_axis
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-10)

    def test_pose_homogeneous_matrix(self) -> None:
        """Test conversion to 4x4 homogeneous transform."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0, 0, np.pi / 2])
        T = pose.homogeneous_matrix
        assert T.shape == (4, 4)

        # Check structure
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3], atol=1e-10)

    def test_pose_translate(self) -> None:
        """Test translation operation."""
        pose = Pose6DOF(position=[1, 2, 3])
        translated = pose.translate([1, 0, 0])
        np.testing.assert_allclose(translated.position, [2, 2, 3], atol=1e-10)
        # Original unchanged
        np.testing.assert_allclose(pose.position, [1, 2, 3], atol=1e-10)

    def test_pose_rotate_euler(self) -> None:
        """Test rotation by euler angles."""
        pose = Pose6DOF()
        rotated = pose.rotate_euler([0, 0, np.pi / 2])
        np.testing.assert_allclose(rotated.euler_angles[2], np.pi / 2, atol=1e-10)

    def test_pose_x_y_z_properties(self) -> None:
        """Test convenient x, y, z accessors."""
        pose = Pose6DOF(position=[1.5, 2.5, 3.5])
        assert pose.x == pytest.approx(1.5)
        assert pose.y == pytest.approx(2.5)
        assert pose.z == pytest.approx(3.5)

    def test_pose_roll_pitch_yaw_properties(self) -> None:
        """Test convenient roll, pitch, yaw accessors."""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        pose = Pose6DOF(euler_angles=[roll, pitch, yaw])
        assert pose.roll == pytest.approx(roll)
        assert pose.pitch == pytest.approx(pitch)
        assert pose.yaw == pytest.approx(yaw)

    def test_pose_set_position_components(self) -> None:
        """Test setting individual position components."""
        pose = Pose6DOF()
        pose.x = 5.0
        pose.y = 6.0
        pose.z = 7.0
        np.testing.assert_allclose(pose.position, [5, 6, 7], atol=1e-10)

    def test_pose_set_rotation_components(self) -> None:
        """Test setting individual rotation components."""
        pose = Pose6DOF()
        pose.roll = 0.1
        pose.pitch = 0.2
        pose.yaw = 0.3
        np.testing.assert_allclose(pose.euler_angles, [0.1, 0.2, 0.3], atol=1e-10)

    def test_pose_inverse(self) -> None:
        """Test pose inversion."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        inv = pose.inverse()

        # Composing with inverse should give identity
        composed = pose.compose(inv)
        np.testing.assert_allclose(composed.position, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(composed.euler_angles, [0, 0, 0], atol=1e-6)

    def test_pose_compose(self) -> None:
        """Test composing two poses."""
        pose1 = Pose6DOF(position=[1, 0, 0])
        pose2 = Pose6DOF(position=[0, 1, 0])
        composed = pose1.compose(pose2)
        np.testing.assert_allclose(composed.position, [1, 1, 0], atol=1e-10)

    def test_pose_compose_with_rotation(self) -> None:
        """Test composing poses with rotation."""
        # First rotate 90° about z, then translate [1, 0, 0] in local frame
        pose1 = Pose6DOF(euler_angles=[0, 0, np.pi / 2])
        pose2 = Pose6DOF(position=[1, 0, 0])
        composed = pose1.compose(pose2)

        # After 90° z rotation, local x becomes world y
        np.testing.assert_allclose(composed.position, [0, 1, 0], atol=1e-10)

    def test_pose_transform_point(self) -> None:
        """Test transforming a point by the pose."""
        pose = Pose6DOF(position=[1, 0, 0], euler_angles=[0, 0, np.pi / 2])
        point = np.array([1, 0, 0])
        transformed = pose.transform_point(point)
        # Rotate [1,0,0] by 90° about z -> [0,1,0], then translate by [1,0,0]
        np.testing.assert_allclose(transformed, [1, 1, 0], atol=1e-10)

    def test_pose_transform_vector(self) -> None:
        """Test transforming a direction vector (no translation)."""
        pose = Pose6DOF(position=[10, 20, 30], euler_angles=[0, 0, np.pi / 2])
        vector = np.array([1, 0, 0])
        transformed = pose.transform_vector(vector)
        # Only rotation, no translation
        np.testing.assert_allclose(transformed, [0, 1, 0], atol=1e-10)

    def test_pose_equality(self) -> None:
        """Test pose equality comparison."""
        pose1 = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        pose2 = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        pose3 = Pose6DOF(position=[1, 2, 4], euler_angles=[0.1, 0.2, 0.3])

        assert pose1 == pose2
        assert pose1 != pose3

    def test_pose_copy(self) -> None:
        """Test pose copying."""
        original = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        copied = original.copy()

        assert original == copied
        # Modifying copy shouldn't affect original
        copied.x = 999
        assert original.x == pytest.approx(1.0)

    def test_pose_to_spatial_transform(self) -> None:
        """Test conversion to 6x6 Plücker transform matrix."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        X = pose.to_spatial_transform()
        assert X.shape == (6, 6)

        # Verify structure: should be consistent with xtrans
        R = pose.rotation_matrix
        # Upper left 3x3 should be rotation
        np.testing.assert_allclose(X[:3, :3], R, atol=1e-10)
        # Lower right 3x3 should be rotation
        np.testing.assert_allclose(X[3:6, 3:6], R, atol=1e-10)

    def test_pose_repr(self) -> None:
        """Test string representation."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        repr_str = repr(pose)
        assert "Pose6DOF" in repr_str
        assert "position" in repr_str


class TestTransform6DOF:
    """Tests for Transform6DOF class - 6DOF rigid body transformations."""

    def test_identity_transform(self) -> None:
        """Test identity transformation."""
        T = Transform6DOF.identity()
        np.testing.assert_allclose(T.translation, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=1e-10)

    def test_translation_only(self) -> None:
        """Test pure translation transform."""
        T = Transform6DOF.from_translation([1, 2, 3])
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=1e-10)
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=1e-10)

    def test_rotation_about_x(self) -> None:
        """Test rotation about x-axis."""
        T = Transform6DOF.from_rotation_x(np.pi / 2)
        R = T.rotation_matrix

        # y -> z for 90° about x
        y_axis = np.array([0, 1, 0])
        np.testing.assert_allclose(R @ y_axis, [0, 0, 1], atol=1e-10)

    def test_rotation_about_y(self) -> None:
        """Test rotation about y-axis."""
        T = Transform6DOF.from_rotation_y(np.pi / 2)
        R = T.rotation_matrix

        # z -> x for 90° about y
        z_axis = np.array([0, 0, 1])
        np.testing.assert_allclose(R @ z_axis, [1, 0, 0], atol=1e-10)

    def test_rotation_about_z(self) -> None:
        """Test rotation about z-axis."""
        T = Transform6DOF.from_rotation_z(np.pi / 2)
        R = T.rotation_matrix

        # x -> y for 90° about z
        x_axis = np.array([1, 0, 0])
        np.testing.assert_allclose(R @ x_axis, [0, 1, 0], atol=1e-10)

    def test_rotation_about_arbitrary_axis(self) -> None:
        """Test rotation about arbitrary axis."""
        # 180° rotation about axis [1, 1, 0] (normalized)
        axis = np.array([1, 1, 0]) / np.sqrt(2)
        T = Transform6DOF.from_axis_angle(axis, np.pi)
        R = T.rotation_matrix

        # z should be flipped
        z_axis = np.array([0, 0, 1])
        np.testing.assert_allclose(R @ z_axis, [0, 0, -1], atol=1e-10)

    def test_from_rotation_matrix(self) -> None:
        """Test creating transform from rotation matrix."""
        # 90° about z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T = Transform6DOF.from_rotation_matrix(R, [1, 2, 3])
        np.testing.assert_allclose(T.rotation_matrix, R, atol=1e-10)
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=1e-10)

    def test_compose_transforms(self) -> None:
        """Test composing two transforms."""
        T1 = Transform6DOF.from_translation([1, 0, 0])
        T2 = Transform6DOF.from_translation([0, 1, 0])
        T3 = T1.compose(T2)
        np.testing.assert_allclose(T3.translation, [1, 1, 0], atol=1e-10)

    def test_inverse_transform(self) -> None:
        """Test transform inversion."""
        T = Transform6DOF.from_rotation_z(np.pi / 4)
        T = T.compose(Transform6DOF.from_translation([1, 2, 3]))
        T_inv = T.inverse()

        # T * T_inv should be identity
        identity = T.compose(T_inv)
        np.testing.assert_allclose(identity.translation, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(identity.rotation_matrix, np.eye(3), atol=1e-10)

    def test_transform_point(self) -> None:
        """Test transforming a point."""
        T = Transform6DOF.from_rotation_z(np.pi / 2)
        T = T.compose(Transform6DOF.from_translation([1, 0, 0]))

        point = np.array([1, 0, 0])
        transformed = T.transform_point(point)
        # Rotate [1,0,0] -> [0,1,0], then translate by [1,0,0]
        np.testing.assert_allclose(transformed, [1, 1, 0], atol=1e-10)

    def test_transform_points_batch(self) -> None:
        """Test transforming multiple points."""
        T = Transform6DOF.from_translation([1, 0, 0])
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        transformed = T.transform_points(points)
        expected = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]])
        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_to_homogeneous_matrix(self) -> None:
        """Test conversion to 4x4 homogeneous matrix."""
        T = Transform6DOF.from_translation([1, 2, 3])
        T = T.compose(Transform6DOF.from_rotation_z(np.pi / 2))
        H = T.homogeneous_matrix
        assert H.shape == (4, 4)
        np.testing.assert_allclose(H[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_from_homogeneous_matrix(self) -> None:
        """Test creating transform from 4x4 matrix."""
        H = np.eye(4)
        H[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° about z
        H[:3, 3] = [1, 2, 3]

        T = Transform6DOF.from_homogeneous_matrix(H)
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=1e-10)

        x_axis = np.array([1, 0, 0])
        np.testing.assert_allclose(T.rotation_matrix @ x_axis, [0, 1, 0], atol=1e-10)

    def test_to_spatial_transform_6x6(self) -> None:
        """Test conversion to 6x6 Plücker transform."""
        T = Transform6DOF.from_translation([1, 2, 3])
        X = T.to_spatial_transform()
        assert X.shape == (6, 6)

    def test_interpolate_transforms(self) -> None:
        """Test linear interpolation between transforms."""
        T1 = Transform6DOF.identity()
        T2 = Transform6DOF.from_translation([2, 0, 0])

        T_mid = Transform6DOF.interpolate(T1, T2, 0.5)
        np.testing.assert_allclose(T_mid.translation, [1, 0, 0], atol=1e-10)


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


class TestRotationConversions:
    """Tests for rotation representation conversions."""

    def test_euler_to_quaternion(self) -> None:
        """Test euler to quaternion conversion."""
        euler = [0, 0, np.pi / 2]  # 90° yaw
        quat = euler_to_quaternion(euler)

        # Quaternion should have unit norm
        assert np.linalg.norm(quat) == pytest.approx(1.0)

        # Roundtrip
        euler_back = quaternion_to_euler(quat)
        np.testing.assert_allclose(euler, euler_back, atol=1e-10)

    def test_euler_to_rotation_matrix(self) -> None:
        """Test euler to rotation matrix conversion."""
        # 90° about z
        euler = [0, 0, np.pi / 2]
        R = euler_to_rotation_matrix(euler)

        x = np.array([1, 0, 0])
        np.testing.assert_allclose(R @ x, [0, 1, 0], atol=1e-10)

    def test_rotation_matrix_to_euler(self) -> None:
        """Test rotation matrix to euler conversion."""
        euler_orig = [0.1, 0.2, 0.3]
        R = euler_to_rotation_matrix(euler_orig)
        euler_back = rotation_matrix_to_euler(R)
        np.testing.assert_allclose(euler_orig, euler_back, atol=1e-10)

    def test_axis_angle_to_rotation_matrix(self) -> None:
        """Test axis-angle to rotation matrix conversion."""
        # 90° about z
        R = axis_angle_to_rotation_matrix([0, 0, 1], np.pi / 2)
        x = np.array([1, 0, 0])
        np.testing.assert_allclose(R @ x, [0, 1, 0], atol=1e-10)

    def test_quaternion_multiply(self) -> None:
        """Test quaternion multiplication."""
        # Two 45° rotations about z should equal 90°
        q1 = euler_to_quaternion([0, 0, np.pi / 4])
        q2 = euler_to_quaternion([0, 0, np.pi / 4])
        q3 = quaternion_multiply(q1, q2)

        euler = quaternion_to_euler(q3)
        assert euler[2] == pytest.approx(np.pi / 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
