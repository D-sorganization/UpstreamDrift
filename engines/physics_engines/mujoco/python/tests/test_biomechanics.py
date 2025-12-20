"""Comprehensive tests for biomechanics module."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.biomechanics import (
    BiomechanicalAnalyzer,
    BiomechanicalData,
    SwingRecorder,
)
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML


class TestBiomechanicalData:
    """Tests for BiomechanicalData dataclass."""

    def test_default_initialization(self) -> None:
        """Test default initialization of BiomechanicalData."""
        data = BiomechanicalData()

        assert data.time == 0.0
        assert len(data.joint_positions) == 0
        assert len(data.joint_velocities) == 0
        assert data.club_head_position is None
        assert data.club_head_speed == 0.0
        assert data.kinetic_energy == 0.0

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        q = np.array([0.5, -0.3])
        v = np.array([1.0, -0.5])
        data = BiomechanicalData(
            time=1.5,
            joint_positions=q,
            joint_velocities=v,
            club_head_speed=10.5,
            kinetic_energy=5.2,
        )

        assert data.time == 1.5
        np.testing.assert_array_equal(data.joint_positions, q)
        np.testing.assert_array_equal(data.joint_velocities, v)
        assert data.club_head_speed == 10.5
        assert data.kinetic_energy == 5.2


class TestBiomechanicalAnalyzer:
    """Tests for BiomechanicalAnalyzer class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        return model, data

    def test_initialization(self, model_and_data) -> None:
        """Test analyzer initialization."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        assert analyzer.model == model
        assert analyzer.data == data
        assert analyzer._prev_club_vel is None
        assert analyzer.prev_qvel is None

    def test_find_body_id(self, model_and_data) -> None:
        """Test finding body ID by name pattern."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Should find shoulder body (case-insensitive)
        body_id = analyzer._find_body_id("shoulder")
        assert body_id is not None
        assert body_id > 0

        # Should not find nonexistent body
        body_id = analyzer._find_body_id("nonexistent_body_xyz")
        assert body_id is None

    def test_find_geom_id(self, model_and_data) -> None:
        """Test finding geom ID by name pattern."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Try to find a geom (may not exist in simple model)
        geom_id = analyzer._find_geom_id("floor")
        # Result depends on model, but should not crash

        # Should not find nonexistent geom
        geom_id = analyzer._find_geom_id("nonexistent_geom_xyz")
        assert geom_id is None

    def test_compute_joint_accelerations_first_call(self, model_and_data) -> None:
        """Test joint acceleration computation on first call."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # First call should return zeros and cache state
        qacc = analyzer.compute_joint_accelerations()

        assert qacc.shape == (model.nv,)
        assert np.allclose(qacc, 0.0)
        assert analyzer.prev_qvel is not None

    def test_compute_joint_accelerations_subsequent_calls(self, model_and_data) -> None:
        """Test joint acceleration computation on subsequent calls."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # First call
        analyzer.compute_joint_accelerations()

        # Set some velocities
        data.qvel[:] = [1.0, -0.5]
        data.time = 0.01

        # Second call should compute acceleration
        qacc = analyzer.compute_joint_accelerations()

        assert qacc.shape == (model.nv,)
        assert np.all(np.isfinite(qacc))

    def test_get_club_head_data_with_club(self, model_and_data) -> None:
        """Test getting club head data when club exists."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Step simulation to have valid state
        mujoco.mj_forward(model, data)

        pos, vel, speed = analyzer.get_club_head_data()

        # Club head may or may not exist in simple model
        if pos is not None:
            assert pos.shape == (3,)
            assert vel.shape == (3,)
            assert speed >= 0.0
            assert np.all(np.isfinite(pos))
            assert np.all(np.isfinite(vel))

    def test_get_club_head_data_without_club(self) -> None:
        """Test getting club head data when club doesn't exist."""
        # Create minimal model without club
        xml = """
        <mujoco model="no_club">
            <worldbody>
                <body name="base" pos="0 0 1">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        analyzer = BiomechanicalAnalyzer(model, data)

        pos, vel, speed = analyzer.get_club_head_data()

        assert pos is None
        assert vel is None
        assert speed == 0.0

    def test_get_ground_reaction_forces(self, model_and_data) -> None:
        """Test getting ground reaction forces."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Step simulation
        mujoco.mj_forward(model, data)

        left_grf, right_grf = analyzer.get_ground_reaction_forces()

        # Forces may be None if no contacts
        # If present, should be 3D vectors
        if left_grf is not None:
            assert left_grf.shape == (3,)
            assert np.all(np.isfinite(left_grf))
        if right_grf is not None:
            assert right_grf.shape == (3,)
            assert np.all(np.isfinite(right_grf))

    def test_get_center_of_mass(self, model_and_data) -> None:
        """Test getting center of mass position and velocity."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Step simulation
        mujoco.mj_forward(model, data)

        com_pos, com_vel = analyzer.get_center_of_mass()

        assert com_pos.shape == (3,)
        assert com_vel.shape == (3,)
        assert np.all(np.isfinite(com_pos))
        assert np.all(np.isfinite(com_vel))

    def test_compute_energies(self, model_and_data) -> None:
        """Test energy computation."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Step simulation
        mujoco.mj_forward(model, data)

        ke, pe, te = analyzer.compute_energies()

        assert isinstance(ke, float)
        assert isinstance(pe, float)
        assert isinstance(te, float)
        assert ke >= 0.0
        assert np.isfinite(ke)
        assert np.isfinite(pe)
        assert np.isfinite(te)
        assert abs(te - (ke + pe)) < 1e-6

    def test_get_actuator_powers(self, model_and_data) -> None:
        """Test actuator power computation."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Set some control inputs
        data.ctrl[:] = [1.0, -0.5]
        mujoco.mj_forward(model, data)

        powers = analyzer.get_actuator_powers()

        assert powers.shape == (model.nu,)
        assert np.all(np.isfinite(powers))

    def test_extract_full_state(self, model_and_data) -> None:
        """Test extracting full biomechanical state."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # Step simulation
        mujoco.mj_forward(model, data)

        state = analyzer.extract_full_state()

        assert isinstance(state, BiomechanicalData)
        assert state.time >= 0.0
        assert state.joint_positions.shape == (model.nq,)
        assert state.joint_velocities.shape == (model.nv,)
        assert state.joint_accelerations.shape == (model.nv,)
        assert state.actuator_forces.shape == (model.nu,)
        assert state.actuator_powers.shape == (model.nu,)
        assert np.isfinite(state.kinetic_energy)
        assert np.isfinite(state.potential_energy)
        assert np.isfinite(state.total_energy)

    def test_extract_full_state_multiple_calls(self, model_and_data) -> None:
        """Test extracting state multiple times (tests acceleration computation)."""
        model, data = model_and_data
        analyzer = BiomechanicalAnalyzer(model, data)

        # First call
        mujoco.mj_forward(model, data)
        state1 = analyzer.extract_full_state()

        # Advance time
        data.time = 0.01
        data.qvel[:] = [0.1, -0.05]
        mujoco.mj_forward(model, data)

        # Second call
        state2 = analyzer.extract_full_state()

        assert state2.time > state1.time
        assert state2.joint_velocities.shape == state1.joint_velocities.shape


class TestSwingRecorder:
    """Tests for SwingRecorder class."""

    def test_initialization(self) -> None:
        """Test recorder initialization."""
        recorder = SwingRecorder()

        assert len(recorder.frames) == 0
        assert not recorder.is_recording

    def test_reset(self) -> None:
        """Test resetting recorder."""
        recorder = SwingRecorder()
        recorder.start_recording()
        recorder.record_frame(BiomechanicalData(time=1.0))

        recorder.reset()

        assert len(recorder.frames) == 0
        assert not recorder.is_recording

    def test_start_stop_recording(self) -> None:
        """Test starting and stopping recording."""
        recorder = SwingRecorder()

        assert not recorder.is_recording

        recorder.start_recording()
        assert recorder.is_recording

        recorder.stop_recording()
        assert not recorder.is_recording

    def test_record_frame_when_recording(self) -> None:
        """Test recording frames when recording is active."""
        recorder = SwingRecorder()
        recorder.start_recording()

        data1 = BiomechanicalData(time=0.0)
        data2 = BiomechanicalData(time=0.01)

        recorder.record_frame(data1)
        recorder.record_frame(data2)

        assert len(recorder.frames) == 2
        assert recorder.frames[0].time == 0.0
        assert recorder.frames[1].time == 0.01

    def test_record_frame_when_not_recording(self) -> None:
        """Test that frames are not recorded when not recording."""
        recorder = SwingRecorder()

        data = BiomechanicalData(time=0.0)
        recorder.record_frame(data)

        assert len(recorder.frames) == 0

    def test_get_time_series_scalar(self) -> None:
        """Test getting time series for scalar field."""
        recorder = SwingRecorder()
        recorder.start_recording()

        for i in range(5):
            data = BiomechanicalData(time=i * 0.01, kinetic_energy=float(i))
            recorder.record_frame(data)

        times, values = recorder.get_time_series("kinetic_energy")

        assert len(times) == 5
        assert len(values) == 5
        np.testing.assert_array_equal(values, [0, 1, 2, 3, 4])

    def test_get_time_series_array(self) -> None:
        """Test getting time series for array field."""
        recorder = SwingRecorder()
        recorder.start_recording()

        for i in range(3):
            data = BiomechanicalData(
                time=i * 0.01,
                joint_positions=np.array([i, i + 1]),
            )
            recorder.record_frame(data)

        times, values = recorder.get_time_series("joint_positions")

        assert len(times) == 3
        assert values.shape == (3, 2)
        np.testing.assert_array_equal(values[0], [0, 1])
        np.testing.assert_array_equal(values[1], [1, 2])

    def test_get_time_series_empty(self) -> None:
        """Test getting time series from empty recorder."""
        recorder = SwingRecorder()

        times, values = recorder.get_time_series("kinetic_energy")

        assert len(times) == 0
        assert len(values) == 0

    def test_get_time_series_with_none_values(self) -> None:
        """Test getting time series with None values."""
        recorder = SwingRecorder()
        recorder.start_recording()

        # Record frames with None values
        for i in range(3):
            data = BiomechanicalData(time=i * 0.01, club_head_position=None)
            recorder.record_frame(data)

        times, values = recorder.get_time_series("club_head_position")

        # Should return empty arrays when all values are None
        assert len(times) == 0 or len(values) == 0

    def test_get_num_frames(self) -> None:
        """Test getting number of frames."""
        recorder = SwingRecorder()
        recorder.start_recording()

        assert recorder.get_num_frames() == 0

        for i in range(5):
            recorder.record_frame(BiomechanicalData(time=i * 0.01))

        assert recorder.get_num_frames() == 5

    def test_get_duration(self) -> None:
        """Test getting recording duration."""
        recorder = SwingRecorder()
        recorder.start_recording()

        assert recorder.get_duration() == 0.0

        recorder.record_frame(BiomechanicalData(time=0.0))
        recorder.record_frame(BiomechanicalData(time=0.5))

        assert recorder.get_duration() == 0.5

    def test_get_duration_single_frame(self) -> None:
        """Test duration with single frame."""
        recorder = SwingRecorder()
        recorder.start_recording()

        recorder.record_frame(BiomechanicalData(time=0.0))

        assert recorder.get_duration() == 0.0

    def test_export_to_dict(self) -> None:
        """Test exporting to dictionary."""
        recorder = SwingRecorder()
        recorder.start_recording()

        for i in range(3):
            data = BiomechanicalData(
                time=i * 0.01,
                kinetic_energy=float(i),
                joint_positions=np.array([i, i + 1]),
            )
            recorder.record_frame(data)

        export_dict = recorder.export_to_dict()

        assert "time" in export_dict
        assert "kinetic_energy" in export_dict
        assert len(export_dict["time"]) == 3
        assert len(export_dict["kinetic_energy"]) == 3

    def test_export_to_dict_empty(self) -> None:
        """Test exporting empty recorder."""
        recorder = SwingRecorder()

        export_dict = recorder.export_to_dict()

        assert export_dict == {}

    def test_export_to_dict_with_3d_fields(self) -> None:
        """Test exporting with 3D vector fields."""
        recorder = SwingRecorder()
        recorder.start_recording()

        for i in range(2):
            data = BiomechanicalData(
                time=i * 0.01,
                club_head_position=np.array([i, i + 1, i + 2]),
            )
            recorder.record_frame(data)

        export_dict = recorder.export_to_dict()

        assert "club_head_position_x" in export_dict
        assert "club_head_position_y" in export_dict
        assert "club_head_position_z" in export_dict
        assert len(export_dict["club_head_position_x"]) == 2
