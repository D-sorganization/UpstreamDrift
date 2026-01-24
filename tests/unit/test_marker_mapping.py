"""Tests for marker-to-model mapping (Guideline A2 - Mandatory)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.engine_availability import (
    MUJOCO_AVAILABLE,
    skip_if_unavailable,
)

pytestmark = skip_if_unavailable("mujoco")

if MUJOCO_AVAILABLE:
    import mujoco

from src.shared.python.marker_mapping import (
    MarkerMapping,
    MarkerToModelMapper,
)


@pytest.fixture
def simple_model() -> mujoco.MjModel:
    """Simple model for testing."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="torso" pos="0 0 1">
                <geom type="box" size="0.2 0.1 0.3"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class TestMarkerMapping:
    """Test marker mapping functionality."""

    def test_add_mapping(self, simple_model: mujoco.MjModel) -> None:
        """Test adding marker mappings."""
        mapper = MarkerToModelMapper(simple_model)

        mapping = MarkerMapping(
            marker_name="M1",
            body_name="torso",
            body_offset=np.array([0.1, 0, 0]),
        )

        mapper.add_mapping(mapping)
        assert "torso" in mapper._mappings
        assert len(mapper._mappings["torso"]) == 1

    def test_perfect_fit_zero_residual(self, simple_model: mujoco.MjModel) -> None:
        """Test perfect fit gives zero residual."""
        mapper = MarkerToModelMapper(simple_model)

        # Create perfect correspondence
        offsets = [
            np.array([0.1, 0, 0]),
            np.array([0, 0.1, 0]),
            np.array([0, 0, 0.1]),
        ]

        for i, offset in enumerate(offsets):
            mapper.add_mapping(MarkerMapping(f"M{i}", "torso", offset))

        # Observed markers = offsets (perfect fit)
        marker_pos = np.array(offsets)

        result = mapper.fit_segment_pose("torso", marker_pos)

        assert result.success
        assert result.rms_error < 1e-6

    def test_outlier_detection(self, simple_model: mujoco.MjModel) -> None:
        """Test outlier detection removes bad markers."""
        mapper = MarkerToModelMapper(simple_model)

        # Add 4 markers
        for i in range(4):
            mapper.add_mapping(
                MarkerMapping(f"M{i}", "torso", np.array([i * 0.1, 0, 0]))
            )

        # 3 good markers + 1 outlier
        marker_pos = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
                [10.0, 10.0, 10.0],  # Outlier
            ]
        )

        result = mapper.fit_segment_pose("torso", marker_pos)

        assert result.success
        assert len(result.outlier_indices) > 0
        assert 3 in result.outlier_indices


@pytest.mark.integration
class TestKabschAlgorithm:
    """Test Kabsch rigid registration."""

    def test_identity_transform(self, simple_model: mujoco.MjModel) -> None:
        """Test Kabsch with identity transformation."""
        mapper = MarkerToModelMapper(simple_model)

        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        T = mapper._fit_rigid_transform(points, points)

        assert np.allclose(T, np.eye(4), atol=1e-6)

    def test_pure_rotation(self, simple_model: mujoco.MjModel) -> None:
        """Test Kabsch with pure rotation."""
        mapper = MarkerToModelMapper(simple_model)

        source = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # 90° rotation about Z
        angle = np.pi / 2
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        target = (R @ source.T).T

        T = mapper._fit_rigid_transform(source, target)

        assert np.allclose(T[:3, :3], R, atol=1e-6)
        assert np.allclose(T[:3, 3], 0, atol=1e-6)
