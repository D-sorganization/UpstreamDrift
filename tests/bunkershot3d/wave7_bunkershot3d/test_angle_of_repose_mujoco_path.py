"""Wave 7: ``_mujoco_angle_of_repose`` coverage with a mocked ``mujoco`` module.

These tests use ``patch.dict("sys.modules", ...)`` to inject a fake
``mujoco`` so the MuJoCo physical path of the angle-of-repose experiment
runs through its geometry parsing, settling loop, and slope-fitting
post-processing without launching a real simulation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bunkershot3d.calibration.angle_of_repose import (
    AngleOfReposeExperiment,
    _mujoco_angle_of_repose,
)


def _build_mock_mujoco(positions: np.ndarray) -> MagicMock:
    """Return a MagicMock ``mujoco`` module that yields ``positions`` after settle."""
    mj = MagicMock(name="mujoco")
    model = MagicMock()
    model.nbody = positions.shape[0] + 1  # +1 for the world body at index 0

    data = MagicMock()
    # xpos[bid] -> 3-vector. Index 0 is world (any value), 1..n are grains.
    xpos = np.zeros((positions.shape[0] + 1, 3))
    xpos[1:] = positions
    data.xpos = xpos

    mj.MjModel.from_xml_string.return_value = model
    mj.MjData.return_value = data
    mj.mj_step = MagicMock()
    return mj


class TestMujocoAngleOfReposeMocked:
    def test_settles_into_pile_and_returns_angle(self) -> None:
        """A synthetic conical pile yields a finite angle in [5,70]."""
        rng = np.random.default_rng(0)
        n = 100
        # Make a synthetic pile: r decreases linearly with z
        z = rng.uniform(0.0, 0.1, n)
        max_r_at_z = 0.05 - 0.4 * z
        theta = rng.uniform(0, 2 * np.pi, n)
        r = rng.uniform(0, np.clip(max_r_at_z, 0.001, None))
        positions = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

        mj = _build_mock_mujoco(positions)
        with patch.dict("sys.modules", {"mujoco": mj}):
            angle = _mujoco_angle_of_repose(friction=0.5, n_grains=20, settle_steps=10)
        assert 5.0 <= angle <= 70.0

    def test_returns_30_when_no_grains_placed(self) -> None:
        """No grains -> early return of 30.0 fallback."""
        mj = _build_mock_mujoco(np.zeros((0, 3)))
        with patch.dict("sys.modules", {"mujoco": mj}):
            angle = _mujoco_angle_of_repose(friction=0.5, n_grains=5, settle_steps=5)
        assert angle == 30.0

    def test_returns_30_when_flat_pile(self) -> None:
        """All grains at identical z -> z_max <= z_min + r -> 30.0 fallback."""
        positions = np.zeros((10, 3))
        positions[:, 0] = np.linspace(-0.05, 0.05, 10)  # spread in x only
        # z_max == z_min == 0; with r = 0.005, z_max <= z_min + r holds
        mj = _build_mock_mujoco(positions)
        with patch.dict("sys.modules", {"mujoco": mj}):
            angle = _mujoco_angle_of_repose(
                friction=0.5, n_grains=10, settle_steps=5, grain_radius=0.005
            )
        assert angle == 30.0

    def test_invokes_mj_step_for_settle_steps(self) -> None:
        positions = np.array([[0.0, 0.0, 0.01]])
        mj = _build_mock_mujoco(positions)
        with patch.dict("sys.modules", {"mujoco": mj}):
            _mujoco_angle_of_repose(friction=0.5, n_grains=1, settle_steps=7)
        assert mj.mj_step.call_count == 7

    def test_settles_with_only_two_z_bins(self) -> None:
        """Tall narrow pile triggers the fallback branch (<2 valid bins)."""
        # All grains nearly at same z -> only one populated bin
        positions = np.zeros((15, 3))
        rng = np.random.default_rng(1)
        positions[:, 0] = rng.uniform(-0.02, 0.02, 15)
        positions[:, 1] = rng.uniform(-0.02, 0.02, 15)
        # spread z by more than grain_radius so we don't hit z_max<=z_min+r,
        # but cluster nearly all in one bin
        positions[:, 2] = rng.uniform(0.0, 0.05, 15)
        positions[0, 2] = 0.20  # one outlier high -> z_filtered keeps only this one
        mj = _build_mock_mujoco(positions)
        with patch.dict("sys.modules", {"mujoco": mj}):
            angle = _mujoco_angle_of_repose(
                friction=0.5, n_grains=15, settle_steps=2, grain_radius=0.005
            )
        # Should produce a finite, in-range angle via the fallback formula
        assert 5.0 <= angle <= 70.0


class TestAngleOfReposeExperimentMujocoBackend:
    def test_mujoco_backend_dispatches_to_real_path(self) -> None:
        """backend='mujoco' (no override) goes through _mujoco_angle_of_repose."""
        exp = AngleOfReposeExperiment(backend="mujoco", use_mock=False)
        positions = np.array([[0.0, 0.0, 0.01], [0.01, 0.0, 0.05]])
        mj = _build_mock_mujoco(positions)
        with patch.dict("sys.modules", {"mujoco": mj}):
            angle = exp.run_simulation({"friction_coefficient": 0.4})
        assert isinstance(angle, float)

    def test_default_use_mock_false_when_backend_real(self) -> None:
        exp = AngleOfReposeExperiment(backend="mpm")
        assert exp._use_mock is False

    def test_default_use_mock_true_when_backend_mock(self) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        assert exp._use_mock is True

    def test_run_simulation_uses_default_friction_when_missing(self) -> None:
        """Missing friction_coefficient key -> defaults to 0.5."""
        exp = AngleOfReposeExperiment(backend="mock")
        angle = exp.run_simulation({})
        assert angle == pytest.approx(20.0 + 0.5 * 24.0)
