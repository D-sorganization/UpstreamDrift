"""Comprehensive tests for motion capture module."""

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML
from mujoco_humanoid_golf.motion_capture import (
    MarkerSet,
    MotionCaptureFrame,
    MotionCaptureLoader,
    MotionCaptureProcessor,
    MotionCaptureSequence,
    MotionCaptureValidator,
    MotionRetargeting,
)


class TestMotionCaptureFrame:
    """Tests for MotionCaptureFrame dataclass."""

    def test_initialization(self) -> None:
        """Test frame initialization."""
        markers = {"marker1": np.array([1.0, 2.0, 3.0])}
        frame = MotionCaptureFrame(time=1.0, marker_positions=markers)

        assert frame.time == 1.0
        assert "marker1" in frame.marker_positions
        np.testing.assert_array_equal(
            frame.marker_positions["marker1"], [1.0, 2.0, 3.0]
        )


class TestMotionCaptureSequence:
    """Tests for MotionCaptureSequence dataclass."""

    def test_initialization(self) -> None:
        """Test sequence initialization."""
        frames = [
            MotionCaptureFrame(time=0.0, marker_positions={"m1": np.array([0, 0, 0])}),
            MotionCaptureFrame(time=0.01, marker_positions={"m1": np.array([1, 1, 1])}),
        ]

        sequence = MotionCaptureSequence(
            frames=frames,
            frame_rate=100.0,
            marker_names=["m1"],
        )

        assert sequence.num_frames == 2
        assert sequence.frame_rate == 100.0
        assert sequence.duration == 0.01

    def test_get_marker_trajectory(self) -> None:
        """Test getting marker trajectory."""
        frames = [
            MotionCaptureFrame(time=0.0, marker_positions={"m1": np.array([0, 0, 0])}),
            MotionCaptureFrame(time=0.01, marker_positions={"m1": np.array([1, 1, 1])}),
        ]

        sequence = MotionCaptureSequence(
            frames=frames,
            frame_rate=100.0,
            marker_names=["m1"],
        )

        times, positions = sequence.get_marker_trajectory("m1")

        assert len(times) == 2
        assert positions.shape == (2, 3)
        np.testing.assert_array_equal(positions[0], [0, 0, 0])


class TestMarkerSet:
    """Tests for MarkerSet dataclass."""

    def test_initialization(self) -> None:
        """Test marker set initialization."""
        markers = {"m1": "body1", "m2": "body2"}
        offsets = {"m1": np.array([0, 0, 0]), "m2": np.array([0.1, 0, 0])}

        marker_set = MarkerSet(markers=markers, marker_offsets=offsets)

        assert marker_set.markers == markers
        assert len(marker_set.marker_offsets) == 2

    def test_golf_swing_marker_set(self) -> None:
        """Test golf swing marker set."""
        marker_set = MarkerSet.golf_swing_marker_set()

        assert len(marker_set.markers) > 0
        assert "CLUB_HEAD" in marker_set.markers
        assert "LSHO" in marker_set.markers


class TestMotionCaptureLoader:
    """Tests for MotionCaptureLoader class."""

    def test_load_csv(self) -> None:
        """Test loading CSV file."""
        # Create temporary CSV file
        csv_content = (
            "time,marker1_x,marker1_y,marker1_z\n0.0,1.0,2.0,3.0\n0.01,1.1,2.1,3.1\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
            f.write(csv_content)

        try:
            sequence = MotionCaptureLoader.load_csv(str(csv_path), frame_rate=100.0)

            assert sequence.num_frames == 2
            assert sequence.frame_rate == 100.0
            assert "marker1" in sequence.marker_names

        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_load_json(self) -> None:
        """Test loading JSON file."""
        json_content = {
            "frame_rate": 120.0,
            "marker_names": ["m1"],
            "frames": [
                {"time": 0.0, "markers": {"m1": [1.0, 2.0, 3.0]}},
                {"time": 0.01, "markers": {"m1": [1.1, 2.1, 3.1]}},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = Path(f.name)
            json.dump(json_content, f)

        try:
            sequence = MotionCaptureLoader.load_json(str(json_path))

            assert sequence.num_frames == 2
            assert sequence.frame_rate == 120.0
            assert "m1" in sequence.marker_names

        finally:
            if json_path.exists():
                json_path.unlink()

    def test_load_bvh(self) -> None:
        """Test loading BVH file (placeholder)."""
        # BVH loader returns None (placeholder)
        result = MotionCaptureLoader.load_bvh("nonexistent.bvh")
        assert result is None


class TestMotionRetargeting:
    """Tests for MotionRetargeting class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    @pytest.fixture()
    def marker_set(self) -> MarkerSet:
        """Create marker set for testing."""
        return MarkerSet(
            markers={"m1": "shoulder"},
            marker_offsets={"m1": np.array([0, 0, 0])},
        )

    def test_initialization(self, model_and_data, marker_set) -> None:
        """Test retargeting initialization."""
        model, data = model_and_data
        retargeting = MotionRetargeting(model, data, marker_set)

        assert retargeting.model == model
        assert retargeting.data == data
        assert retargeting.marker_set == marker_set

    def test_retarget_sequence(self, model_and_data, marker_set) -> None:
        """Test retargeting sequence."""
        model, data = model_and_data
        retargeting = MotionRetargeting(model, data, marker_set)

        frames = [
            MotionCaptureFrame(
                time=0.0,
                marker_positions={"m1": np.array([0.5, 0.0, 0.0])},
            ),
            MotionCaptureFrame(
                time=0.01,
                marker_positions={"m1": np.array([0.6, 0.0, 0.0])},
            ),
        ]

        sequence = MotionCaptureSequence(
            frames=frames,
            frame_rate=100.0,
            marker_names=["m1"],
        )

        times, trajectories, success_flags = retargeting.retarget_sequence(sequence)

        assert len(times) == 2
        assert trajectories.shape[0] == 2
        assert len(success_flags) == 2
        assert all(isinstance(s, bool) for s in success_flags)

    def test_compute_marker_errors(self, model_and_data, marker_set) -> None:
        """Test computing marker errors."""
        model, data = model_and_data
        retargeting = MotionRetargeting(model, data, marker_set)

        frame = MotionCaptureFrame(
            time=0.0,
            marker_positions={"m1": np.array([0.5, 0.0, 0.0])},
        )

        q = data.qpos.copy()
        errors = retargeting.compute_marker_errors(frame, q)

        assert isinstance(errors, dict)
        if "m1" in errors:
            assert errors["m1"] >= 0.0


class TestMotionCaptureProcessor:
    """Tests for MotionCaptureProcessor class."""

    def test_filter_trajectory(self) -> None:
        """Test filtering trajectory."""
        rng = np.random.default_rng(42)
        times = np.linspace(0, 1, 100)
        positions = rng.standard_normal((100, 3))

        filtered = MotionCaptureProcessor.filter_trajectory(
            times,
            positions,
            cutoff_frequency=6.0,
            sampling_rate=120.0,
        )

        assert filtered.shape == positions.shape
        assert np.all(np.isfinite(filtered))

    def test_compute_velocities_finite_difference(self) -> None:
        """Test computing velocities with finite difference."""
        times = np.array([0.0, 0.01, 0.02])
        positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        velocities = MotionCaptureProcessor.compute_velocities(
            times,
            positions,
            method="finite_difference",
        )

        assert velocities.shape == positions.shape
        assert np.all(np.isfinite(velocities))

    def test_compute_velocities_spline(self) -> None:
        """Test computing velocities with spline."""
        rng = np.random.default_rng(42)
        times = np.linspace(0, 1, 10)
        positions = rng.standard_normal((10, 3))

        velocities = MotionCaptureProcessor.compute_velocities(
            times,
            positions,
            method="spline",
        )

        assert velocities.shape == positions.shape
        assert np.all(np.isfinite(velocities))

    def test_compute_accelerations_finite_difference(self) -> None:
        """Test computing accelerations with finite difference."""
        times = np.array([0.0, 0.01, 0.02])
        velocities = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        accelerations = MotionCaptureProcessor.compute_accelerations(
            times,
            velocities,
            method="finite_difference",
        )

        assert accelerations.shape == velocities.shape
        assert np.all(np.isfinite(accelerations))

    def test_compute_accelerations_spline(self) -> None:
        """Test computing accelerations with spline."""
        rng = np.random.default_rng(42)
        times = np.linspace(0, 1, 10)
        velocities = rng.standard_normal((10, 3))

        accelerations = MotionCaptureProcessor.compute_accelerations(
            times,
            velocities,
            method="spline",
        )

        assert accelerations.shape == velocities.shape
        assert np.all(np.isfinite(accelerations))

    def test_resample_trajectory_linear(self) -> None:
        """Test resampling trajectory with linear interpolation."""
        times = np.array([0.0, 0.5, 1.0])
        trajectory = np.array([[0, 0], [1, 1], [2, 2]])
        new_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        resampled = MotionCaptureProcessor.resample_trajectory(
            times,
            trajectory,
            new_times,
            method="linear",
        )

        assert resampled.shape == (len(new_times), trajectory.shape[1])
        assert np.all(np.isfinite(resampled))

    def test_resample_trajectory_cubic(self) -> None:
        """Test resampling trajectory with cubic interpolation."""
        times = np.array([0.0, 0.5, 1.0])
        trajectory = np.array([[0, 0], [1, 1], [2, 2]])
        new_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        resampled = MotionCaptureProcessor.resample_trajectory(
            times,
            trajectory,
            new_times,
            method="cubic",
        )

        assert resampled.shape == (len(new_times), trajectory.shape[1])
        assert np.all(np.isfinite(resampled))

    def test_time_normalize(self) -> None:
        """Test time normalization."""
        rng = np.random.default_rng(42)
        times = np.linspace(0, 2, 20)
        trajectory = rng.standard_normal((20, 3))

        norm_times, norm_traj = MotionCaptureProcessor.time_normalize(
            times,
            trajectory,
            num_samples=101,
        )

        assert len(norm_times) == 101
        assert norm_traj.shape == (101, 3)
        assert norm_times[0] == 0.0
        assert norm_times[-1] == 1.0


class TestMotionCaptureValidator:
    """Tests for MotionCaptureValidator class."""

    def test_detect_gaps(self) -> None:
        """Test detecting gaps in marker trajectory."""
        frames = [
            MotionCaptureFrame(time=0.0, marker_positions={"m1": np.array([0, 0, 0])}),
            MotionCaptureFrame(time=0.01, marker_positions={"m1": np.array([1, 1, 1])}),
            MotionCaptureFrame(
                time=0.15, marker_positions={"m1": np.array([2, 2, 2])}
            ),  # Gap
        ]

        sequence = MotionCaptureSequence(
            frames=frames,
            frame_rate=100.0,
            marker_names=["m1"],
        )

        gaps = MotionCaptureValidator.detect_gaps(sequence, "m1", gap_threshold=0.05)

        assert isinstance(gaps, list)

    def test_compute_marker_velocity_stats(self) -> None:
        """Test computing marker velocity statistics."""
        frames = [
            MotionCaptureFrame(time=0.0, marker_positions={"m1": np.array([0, 0, 0])}),
            MotionCaptureFrame(time=0.01, marker_positions={"m1": np.array([1, 1, 1])}),
            MotionCaptureFrame(time=0.02, marker_positions={"m1": np.array([2, 2, 2])}),
        ]

        sequence = MotionCaptureSequence(
            frames=frames,
            frame_rate=100.0,
            marker_names=["m1"],
        )

        stats = MotionCaptureValidator.compute_marker_velocity_stats(sequence, "m1")

        assert "mean_speed" in stats or "error" in stats
        if "mean_speed" in stats:
            assert stats["mean_speed"] >= 0.0

    def test_check_marker_visibility(self) -> None:
        """Test checking marker visibility."""
        frames = [
            MotionCaptureFrame(time=0.0, marker_positions={"m1": np.array([0, 0, 0])}),
            MotionCaptureFrame(time=0.01, marker_positions={}),  # Missing marker
            MotionCaptureFrame(time=0.02, marker_positions={"m1": np.array([2, 2, 2])}),
        ]

        sequence = MotionCaptureSequence(
            frames=frames,
            frame_rate=100.0,
            marker_names=["m1"],
        )

        visibility = MotionCaptureValidator.check_marker_visibility(sequence, "m1")

        assert visibility["total_frames"] == 3
        assert visibility["visible_frames"] == 2
        assert visibility["visibility_percentage"] == pytest.approx(
            (2 / 3) * 100, abs=0.1
        )
