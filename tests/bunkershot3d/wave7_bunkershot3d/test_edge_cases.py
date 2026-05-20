"""Wave 7: Edge-case and contract tests across bunkershot3d modules.

Focuses on:
- WrenchTrace high cutoff frequency (returns unfiltered copy).
- WrenchTrace impulses and resampling correctness.
- SwingTrajectory missing-column ValueError.
- SwingTrajectory clamps out-of-range times.
- ClubheadGenerator zero-norm normal handling in STL export.
- ClubheadGenerator vertex/face shapes.
- CalibrationOptimizer raises when experiment has no known targets.
- BunkerShotConfig pydantic validation rejects bad inputs.
- BackendNotImplementedError message format.
- CoupledDoublePendulum & CoSimulator step shapes.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment
from bunkershot3d.calibration.drained_shear_cell import DrainedShearCellExperiment
from bunkershot3d.calibration.optimizer import CalibrationOptimizer
from bunkershot3d.config import BunkerShotConfig
from bunkershot3d.exceptions import BackendNotImplementedError
from bunkershot3d.geometry.clubhead import ClubheadGenerator
from bunkershot3d.io.schema import BunkerShotResultReader, BunkerShotResultWriter
from bunkershot3d.kinematics.coupling import CoSimulator, CoupledDoublePendulum
from bunkershot3d.kinematics.trajectory import (
    SwingTrajectory,
    generate_reference_trajectory,
)
from bunkershot3d.postproc.wrench_trace import WrenchTrace


# ---------------------------------------------------------------------------
# WrenchTrace
# ---------------------------------------------------------------------------


class TestWrenchTrace:
    def _make(self, n: int = 50) -> WrenchTrace:
        t = np.linspace(0, 0.05, n)
        f = np.column_stack(
            [np.sin(2 * np.pi * 100 * t), np.cos(2 * np.pi * 100 * t), np.zeros(n)]
        )
        tq = np.column_stack([np.zeros(n), np.sin(2 * np.pi * 50 * t), np.zeros(n)])
        return WrenchTrace(t, f, tq)

    def test_filter_with_cutoff_above_nyquist_returns_unfiltered_copy(self) -> None:
        """When cutoff_freq >= Nyquist, filter() returns an unfiltered copy."""
        w = self._make(n=20)  # fs = 19/0.05 = 380 Hz, Nyquist = 190 Hz
        # cutoff = 200 Hz > Nyquist -> normal_cutoff >= 1.0 branch
        out = w.filter(cutoff_freq=200.0)
        np.testing.assert_array_equal(out.force_world, w.force_world)
        np.testing.assert_array_equal(out.torque_world, w.torque_world)
        # Should be a copy, not the same array
        assert out.force_world is not w.force_world

    def test_filter_low_cutoff_smooths_signal(self) -> None:
        w = self._make(n=200)
        # fs = 199/0.05 ~ 3980 Hz, Nyquist ~ 1990 Hz; cutoff 50 Hz is well below.
        out = w.filter(cutoff_freq=50.0, order=2)
        # Filtered force should have lower peak-to-peak than original
        assert out.force_world.shape == w.force_world.shape

    def test_resample_onto_uniform_grid(self) -> None:
        w = self._make(n=50)
        new_t = np.linspace(0.0, 0.05, 25)
        out = w.resample(new_t)
        assert out.time.shape == (25,)
        assert out.force_world.shape == (25, 3)
        assert out.torque_world.shape == (25, 3)

    def test_impulses_zero_for_sine(self) -> None:
        """Integral of pure sine over one period is approximately zero."""
        n = 1001
        t = np.linspace(0, 0.01, n)  # exactly one period of 100 Hz
        f = np.column_stack([np.sin(2 * np.pi * 100 * t), np.zeros(n), np.zeros(n)])
        tq = np.zeros((n, 3))
        w = WrenchTrace(t, f, tq)
        lin, ang = w.get_impulses()
        assert lin.shape == (3,) and ang.shape == (3,)
        # x impulse near zero, y/z exactly zero
        assert abs(lin[0]) < 1e-3
        assert lin[1] == 0.0 and lin[2] == 0.0


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestSwingTrajectory:
    def test_missing_column_raises_value_error(self, tmp_path: Path) -> None:
        # CSV missing "qx" column
        csv = tmp_path / "bad.csv"
        csv.write_text(
            "time,px,py,pz,qw,qy,qz,vx,vy,vz,wx,wy,wz\n0,0,0,0,1,0,0,0,0,0,0,0,0\n"
        )
        with pytest.raises(ValueError, match="qx"):
            SwingTrajectory.from_csv(csv)

    def test_interpolate_clamps_below_range(self, tmp_path: Path) -> None:
        csv = tmp_path / "ref.csv"
        generate_reference_trajectory(csv)
        traj = SwingTrajectory.from_csv(csv)
        # Time before first sample -> clamped
        pos, quat, lv, av = traj.interpolate(-5.0)
        assert pos.shape == (3,) and quat.shape == (4,)
        # Quaternion is renormalized to unit length
        assert math.isclose(np.linalg.norm(quat), 1.0, abs_tol=1e-9)

    def test_interpolate_clamps_above_range(self, tmp_path: Path) -> None:
        csv = tmp_path / "ref.csv"
        generate_reference_trajectory(csv)
        traj = SwingTrajectory.from_csv(csv)
        pos, quat, lv, av = traj.interpolate(999.0)
        assert pos.shape == (3,)
        assert math.isclose(np.linalg.norm(quat), 1.0, abs_tol=1e-9)

    def test_interpolate_in_range(self, tmp_path: Path) -> None:
        csv = tmp_path / "ref.csv"
        generate_reference_trajectory(csv)
        traj = SwingTrajectory.from_csv(csv)
        pos, quat, lv, av = traj.interpolate(0.05)
        assert pos.shape == (3,)
        assert quat.shape == (4,)
        assert lv.shape == (3,)
        assert av.shape == (3,)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestClubheadGenerator:
    def test_generate_mesh_vertex_and_face_shapes(self) -> None:
        gen = ClubheadGenerator(loft_deg=56.0, bounce_deg=10.0)
        verts, faces = gen.generate_mesh()
        assert verts.shape == (6, 3)
        assert faces.shape == (8, 3)
        # Each face references existing vertices
        assert faces.max() < verts.shape[0]
        assert faces.min() >= 0

    def test_zero_loft_zero_bounce_produces_finite_vertices(self) -> None:
        gen = ClubheadGenerator(loft_deg=0.0, bounce_deg=0.0)
        verts, _ = gen.generate_mesh()
        assert np.all(np.isfinite(verts))

    def test_export_stl_writes_file(self, tmp_path: Path) -> None:
        gen = ClubheadGenerator()
        path = tmp_path / "wedge.stl"
        gen.export_stl(path)
        text = path.read_text()
        assert text.startswith("solid wedge")
        assert text.rstrip().endswith("endsolid wedge")
        # 8 facets total
        assert text.count("facet normal") == 8

    def test_export_stl_handles_degenerate_face_with_zero_normal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If a face yields zero cross-product norm, export still writes facet."""
        gen = ClubheadGenerator()

        # Force generate_mesh to return a mesh with one degenerate triangle
        def bad_mesh() -> tuple[np.ndarray, np.ndarray]:
            verts = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            )  # collinear
            faces = np.array([[0, 1, 2]])
            return verts, faces

        monkeypatch.setattr(gen, "generate_mesh", bad_mesh)
        path = tmp_path / "degen.stl"
        gen.export_stl(path)
        text = path.read_text()
        assert "facet normal" in text


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class TestCalibrationOptimizer:
    def test_raises_on_unknown_experiment(self) -> None:
        class _FakeExp:
            def run_simulation(self, params: dict) -> float:
                return 0.0

        opt = CalibrationOptimizer(_FakeExp())
        with pytest.raises(ValueError, match="target"):
            opt._objective(np.array([0.5, 0.3]))

    def test_objective_for_angle_of_repose(self) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        opt = CalibrationOptimizer(exp)
        val = opt._objective(np.array([0.5, 0.3]))
        # angle = 20 + 0.5*24 = 32; target = 32; residual^2 = 0
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_objective_for_shear_cell(self) -> None:
        exp = DrainedShearCellExperiment(backend="mock")
        opt = CalibrationOptimizer(exp)
        # Shear-cell mock: phi_peak = 20 + friction*30; phi_res = phi_peak - 5
        # target_phi_peak=35, target_phi_res=30 -> friction=0.5 hits both exactly
        val = opt._objective(np.array([0.5, 0.5]))
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_objective_clips_out_of_bounds_inputs(self) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        opt = CalibrationOptimizer(exp)
        # friction > 1 should be clipped to 1.0 (angle = 44)
        val = opt._objective(np.array([5.0, 5.0]))
        # angle = 20 + 1.0 * 24 = 44; target = 32; residual^2 = 144
        assert val == pytest.approx(144.0, abs=1e-9)

    def test_optimize_returns_dict_with_error(self) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        opt = CalibrationOptimizer(exp)
        result = opt.optimize()
        assert "friction_coefficient" in result
        assert "restitution_coefficient" in result
        assert "error" in result
        assert 0.01 <= result["friction_coefficient"] <= 1.0


# ---------------------------------------------------------------------------
# Config / Pydantic
# ---------------------------------------------------------------------------


class TestBunkerShotConfig:
    def test_rejects_negative_length(self, tmp_path: Path) -> None:
        bad = """
bunker_bed:
  domain: {length_x: -1.0, width_y: 1.0, depth_z: 0.5}
  boundary: "fixed"
grain_population:
  count: 10
  diameter_mean: 0.002
  diameter_sigma_log: 0.1
  density: 2650.0
contact_model:
  friction_coefficient: 0.5
  restitution_coefficient: 0.3
  youngs_modulus: 1.0e7
  poisson_ratio: 0.25
clubhead: {loft_deg: 56.0, bounce_deg: 10.0, width: 0.1, height: 0.05, mass: 0.3}
trajectory: {file: "x.csv"}
output: {rate_hz: 500.0}
"""
        p = tmp_path / "bad.yaml"
        p.write_text(bad)
        with pytest.raises(ValueError):
            BunkerShotConfig.from_yaml(p)

    def test_rejects_friction_above_one(self, tmp_path: Path) -> None:
        bad = """
bunker_bed:
  domain: {length_x: 1.0, width_y: 1.0, depth_z: 0.5}
  boundary: "fixed"
grain_population:
  count: 10
  diameter_mean: 0.002
  diameter_sigma_log: 0.1
  density: 2650.0
contact_model:
  friction_coefficient: 1.5
  restitution_coefficient: 0.3
  youngs_modulus: 1.0e7
  poisson_ratio: 0.25
clubhead: {loft_deg: 56.0, bounce_deg: 10.0, width: 0.1, height: 0.05, mass: 0.3}
trajectory: {file: "x.csv"}
output: {rate_hz: 500.0}
"""
        p = tmp_path / "bad.yaml"
        p.write_text(bad)
        with pytest.raises(ValueError):
            BunkerShotConfig.from_yaml(p)

    def test_rejects_invalid_boundary(self, tmp_path: Path) -> None:
        bad = """
bunker_bed:
  domain: {length_x: 1.0, width_y: 1.0, depth_z: 0.5}
  boundary: "weird"
grain_population:
  count: 10
  diameter_mean: 0.002
  diameter_sigma_log: 0.1
  density: 2650.0
contact_model:
  friction_coefficient: 0.5
  restitution_coefficient: 0.3
  youngs_modulus: 1.0e7
  poisson_ratio: 0.25
clubhead: {loft_deg: 56.0, bounce_deg: 10.0, width: 0.1, height: 0.05, mass: 0.3}
trajectory: {file: "x.csv"}
output: {rate_hz: 500.0}
"""
        p = tmp_path / "bad.yaml"
        p.write_text(bad)
        with pytest.raises(ValueError):
            BunkerShotConfig.from_yaml(p)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestBackendNotImplementedError:
    def test_message_includes_backend_name(self) -> None:
        err = BackendNotImplementedError("foobar")
        assert "foobar" in str(err)
        assert err.backend == "foobar"

    def test_message_includes_feature_when_provided(self) -> None:
        err = BackendNotImplementedError("foo", feature="bar")
        assert "bar" in str(err)
        assert err.feature == "bar"

    def test_is_not_implemented_error(self) -> None:
        err = BackendNotImplementedError("foo")
        assert isinstance(err, NotImplementedError)


# ---------------------------------------------------------------------------
# Coupling / Co-simulation
# ---------------------------------------------------------------------------


class TestCoupling:
    def test_step_advances_time(self) -> None:
        p = CoupledDoublePendulum()
        t0 = p.time
        wrench = (np.array([1.0, 0.0, 0.5]), np.array([0.0, 0.1, 0.0]))
        p.step(0.001, wrench)
        assert p.time > t0

    def test_get_clubhead_pose_shapes(self) -> None:
        p = CoupledDoublePendulum()
        pos, quat, lvel, avel = p.get_clubhead_pose()
        assert pos.shape == (3,)
        assert quat.shape == (4,)
        assert lvel.shape == (3,)
        assert avel.shape == (3,)
        # quat is unit-norm by construction
        assert math.isclose(np.linalg.norm(quat), 1.0, abs_tol=1e-9)

    def test_cosimulator_step_returns_wrench(self) -> None:
        p = CoupledDoublePendulum()
        backend = MagicMock()
        sim = CoSimulator(p, backend)
        force, torque = sim.step(0.001)
        assert force.shape == (3,)
        assert torque.shape == (3,)
        # Default mock force in CoSimulator
        assert np.allclose(force, [10.0, 0.0, 5.0])
        assert np.allclose(torque, [0.0, 1.0, 0.0])

    def test_cosimulator_advances_pendulum_time(self) -> None:
        p = CoupledDoublePendulum()
        sim = CoSimulator(p, backend_driver=MagicMock())
        t0 = p.time
        sim.step(0.002)
        assert p.time == pytest.approx(t0 + 0.002)


# ---------------------------------------------------------------------------
# IO Schema
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    def test_grain_state_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "r.h5"
        w = BunkerShotResultWriter(path)
        pos = np.random.default_rng(0).normal(size=(10, 3))
        vel = np.random.default_rng(1).normal(size=(10, 3))
        w.write_grain_state(0.001, pos, vel)
        w.close()

        r = BunkerShotResultReader(path)
        times, positions, velocities = r.read_grain_states()
        r.close()
        assert times.shape == (1,)
        np.testing.assert_allclose(positions[0], pos)
        np.testing.assert_allclose(velocities[0], vel)

    def test_clubhead_and_wrench_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "r.h5"
        w = BunkerShotResultWriter(path)
        pos = np.array([0.1, 0.0, 0.2])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        force = np.array([5.0, 0.0, -1.0])
        torque = np.array([0.0, 0.1, 0.0])
        w.write_clubhead_state(0.001, pos, quat)
        w.write_clubhead_state(0.002, pos * 2, quat)
        w.write_contact_wrench(0.001, force, torque)
        w.write_contact_wrench(0.002, force * 2, torque)
        w.close()

        r = BunkerShotResultReader(path)
        t, p, q = r.read_clubhead_states()
        tw, f, tq = r.read_contact_wrenches()
        r.close()
        assert t.shape == (2,)
        assert p.shape == (2, 3)
        assert q.shape == (2, 4)
        assert tw.shape == (2,)
        assert f.shape == (2, 3)
        assert tq.shape == (2, 3)
