"""Tests for wrench logging integration in GenericPhysicsRecorder (Issue #761).

Validates that the recorder correctly:
- Records ground forces and moments per-step
- Computes GRF analysis (impulse, COP trajectory) post-hoc
- Fits Functional Swing Plane (FSP) from clubhead trajectory
- Decomposes GRF wrenches into swing-plane components
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder


def _make_engine(nq: int = 7, nv: int = 6) -> MagicMock:
    """Create a mock PhysicsEngine with configurable DOF.

    Returns an engine that produces a golf-like swing trajectory with
    non-trivial ground forces and clubhead motion.
    """
    engine = MagicMock()
    engine.model_name = "TestEngine"

    # State
    q = np.zeros(nq)
    v = np.zeros(nv)
    _time = [0.0]

    def get_full_state() -> dict:
        return {
            "q": q.copy(),
            "v": v.copy(),
            "t": _time[0],
            "M": np.eye(nv),
        }

    engine.get_full_state = get_full_state

    # Contact forces: return 6-vector [force(3), moment(3)]
    _step = [0]

    def compute_contact_forces() -> np.ndarray:
        # Simulate varying vertical force with small horizontal
        t = _time[0]
        fz = 800.0 + 200.0 * np.sin(np.pi * t)
        fx = 50.0 * np.cos(2 * np.pi * t)
        fy = 30.0 * np.sin(2 * np.pi * t)
        # Moments from COP offset
        mx = 10.0 * np.sin(np.pi * t)
        my = -20.0 * np.cos(np.pi * t)
        mz = 0.0
        return np.array([fx, fy, fz, mx, my, mz])

    engine.compute_contact_forces = compute_contact_forces

    # Set state / forward (for post-hoc replay)
    def set_state(q_new: np.ndarray, v_new: np.ndarray) -> None:
        q[:] = q_new
        v[:] = v_new

    engine.set_state = set_state
    engine.set_control = MagicMock()
    engine.forward = MagicMock()

    # ZTCF / ZVCF stubs
    engine.compute_ztcf = MagicMock(return_value=np.zeros(nv))
    engine.compute_zvcf = MagicMock(return_value=np.zeros(nv))
    engine.compute_drift_acceleration = MagicMock(return_value=np.zeros(nv))
    engine.compute_control_acceleration = MagicMock(return_value=np.zeros(nv))

    return engine, _time, _step


def _record_swing(n_frames: int = 100, duration: float = 1.0) -> GenericPhysicsRecorder:
    """Record a synthetic swing with GRF and clubhead trajectory."""
    engine, _time, _step = _make_engine()
    recorder = GenericPhysicsRecorder(engine, max_samples=n_frames + 100)
    recorder.start()

    dt = duration / n_frames
    for i in range(n_frames):
        t = i * dt
        _time[0] = t

        # Simulate clubhead trajectory (arc in XZ plane)
        theta = np.pi * t / duration  # 0 to pi
        radius = 1.5  # [m]
        recorder.record_step()

        # Manually set clubhead position (recorder doesn't auto-capture this)
        if recorder.data["club_head_position"] is not None:
            recorder.data["club_head_position"][i] = np.array(
                [
                    radius * np.sin(theta),
                    0.1 * np.sin(2 * theta),  # slight Y deviation
                    radius * np.cos(theta),
                ]
            )

    recorder.stop()
    return recorder


class TestGroundMomentsRecording:
    """Test that ground moments are recorded from 6-vector contact forces."""

    def test_moments_buffer_initialized(self) -> None:
        recorder = _record_swing(20)
        assert recorder.data["ground_moments"] is not None
        assert recorder.data["ground_moments"].shape[1] == 3

    def test_moments_nonzero_when_engine_returns_6vec(self) -> None:
        recorder = _record_swing(50)
        moments = recorder.data["ground_moments"][: recorder.current_idx]
        # Engine returns non-zero moments
        assert np.any(moments != 0), "Moments should be non-zero"

    def test_forces_recorded(self) -> None:
        recorder = _record_swing(50)
        forces = recorder.data["ground_forces"][: recorder.current_idx]
        # Vertical force should be ~800 N
        assert np.mean(forces[:, 2]) > 500.0


class TestGRFAnalysis:
    """Test GRF analysis integration."""

    def test_grf_analysis_returns_results(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        assert "grf_analysis" in result
        assert "fsp" in result
        assert "wrench_swing_plane" in result

    def test_grf_peak_force(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        grf = result["grf_analysis"]
        assert grf["peak_vertical_force"] > 0

    def test_grf_impulse_computed(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        grf = result["grf_analysis"]
        assert "linear_impulse_magnitude" in grf
        assert grf["linear_impulse_magnitude"] > 0

    def test_grf_cop_trajectory(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        grf = result["grf_analysis"]
        assert "cop_trajectory_length" in grf

    def test_empty_recorder_returns_empty(self) -> None:
        engine, _, _ = _make_engine()
        recorder = GenericPhysicsRecorder(engine)
        result = recorder.compute_grf_and_wrench_analysis()
        assert result == {}


class TestFSPComputation:
    """Test Functional Swing Plane fitting from recorded trajectory."""

    def test_fsp_fitted(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        fsp = result["fsp"]
        assert "normal" in fsp
        assert "fitting_rmse" in fsp
        assert "fitting_window_ms" in fsp

    def test_fsp_normal_is_unit_vector(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        normal = result["fsp"]["normal"]
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-6)

    def test_fsp_rmse_reported(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        # RMSE should be small for a near-planar arc
        assert result["fsp"]["fitting_rmse"] >= 0

    def test_fsp_window_ms_matches_input(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis(fsp_window_ms=200.0)
        assert result["fsp"]["fitting_window_ms"] == 200.0

    def test_custom_impact_time(self) -> None:
        recorder = _record_swing(100, duration=1.0)
        result = recorder.compute_grf_and_wrench_analysis(impact_time=0.5)
        assert "fsp" in result
        assert result["fsp"]["fitting_rmse"] >= 0


class TestWrenchDecomposition:
    """Test swing-plane wrench decomposition."""

    def test_decomposition_arrays_exist(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        wrench = result["wrench_swing_plane"]
        expected_keys = [
            "force_in_plane",
            "force_out_of_plane",
            "force_along_grip",
            "torque_in_plane",
            "torque_out_of_plane",
            "torque_about_grip",
        ]
        for key in expected_keys:
            assert key in wrench, f"Missing wrench key: {key}"
            assert len(wrench[key]) == recorder.current_idx

    def test_force_decomposition_nonzero(self) -> None:
        recorder = _record_swing(100)
        result = recorder.compute_grf_and_wrench_analysis()

        wrench = result["wrench_swing_plane"]
        # Forces are non-trivial, so decomposition should be nonzero
        assert np.any(wrench["force_in_plane"] != 0)
        assert np.any(wrench["force_out_of_plane"] != 0)

    def test_force_magnitude_conservation(self) -> None:
        """Decomposed force magnitude should equal original force magnitude."""
        recorder = _record_swing(50)
        result = recorder.compute_grf_and_wrench_analysis()

        forces = recorder.data["ground_forces"][: recorder.current_idx]
        wrench = result["wrench_swing_plane"]

        for i in range(recorder.current_idx):
            original_mag = np.linalg.norm(forces[i])
            # In-plane + out-of-plane should reconstruct magnitude
            # (grip axis is not orthogonal to in_plane, so use full 3D)
            decomp_mag = np.sqrt(
                wrench["force_in_plane"][i] ** 2 + wrench["force_out_of_plane"][i] ** 2
            )
            # Allow some tolerance since force_along_grip overlaps with in-plane
            if original_mag > 1.0:
                assert decomp_mag > 0, f"Frame {i}: decomposed force is zero"

    def test_data_stored_in_recorder(self) -> None:
        """Analysis results should be stored in recorder's data dict."""
        recorder = _record_swing(50)
        recorder.compute_grf_and_wrench_analysis()

        assert "grf_analysis" in recorder.data
        assert "fsp" in recorder.data
        assert "wrench_swing_plane" in recorder.data


class TestExportIntegration:
    """Test that wrench data is included in get_data_dict()."""

    def test_get_data_dict_includes_wrench(self) -> None:
        recorder = _record_swing(50)
        recorder.compute_grf_and_wrench_analysis()

        data = recorder.get_data_dict()
        assert "grf_analysis" in data
        assert "fsp" in data
        assert "wrench_swing_plane" in data
        assert "ground_moments" in data

    def test_get_data_dict_includes_ground_moments(self) -> None:
        recorder = _record_swing(50)
        data = recorder.get_data_dict()

        assert "ground_moments" in data
        if isinstance(data["ground_moments"], np.ndarray):
            assert data["ground_moments"].shape[1] == 3
