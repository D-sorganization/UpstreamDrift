"""Unit tests for motion_pipeline.ik.base."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import JointTrajectory
from src.shared.python.motion_pipeline.ik.base import (
    BaseIKSolver,
    IKBackendType,
    IKConfig,
    InverseKinematicsSolver,
    MarkerWeights,
    make_ik_solver,
)

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_marker_weights_default() -> None:
    w = MarkerWeights()
    assert w.default_weight == 1.0
    assert w.get_weight("any") == 1.0


def test_marker_weights_per_marker_override() -> None:
    w = MarkerWeights(default_weight=1.0, marker_weights={"RASI": 5.0})
    assert w.get_weight("RASI") == 5.0
    assert w.get_weight("LASI") == 1.0


def test_ik_config_defaults() -> None:
    c = IKConfig()
    assert c.max_iterations == 100
    assert c.tolerance == pytest.approx(1e-6)
    assert c.use_orientation is True
    assert c.regularization == pytest.approx(0.01)


def test_ik_backend_type_enum_members() -> None:
    members = {b.value for b in IKBackendType}
    assert "mujoco" in members
    assert "opensim" in members
    assert "drake" in members
    assert "pinocchio" in members
    assert "geometric" in members


def test_make_ik_solver_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="(?i)unknown|enum"):
        make_ik_solver("not-a-backend")


def test_make_ik_solver_geometric_raises_module_not_found() -> None:
    """Geometric backend module is referenced but not present."""
    # Either ImportError or ValueError is acceptable; the production code
    # imports geometric_backend, which does not exist.
    with pytest.raises((ImportError, ModuleNotFoundError)):
        make_ik_solver(IKBackendType.GEOMETRIC)


def test_make_ik_solver_returns_protocol_compatible() -> None:
    """MuJoCo backend instantiates without requiring the mujoco package."""
    solver = make_ik_solver(IKBackendType.MUJOCO)
    assert hasattr(solver, "solve")
    assert hasattr(solver, "solve_frame")


def test_make_ik_solver_accepts_string_alias() -> None:
    solver = make_ik_solver("mujoco")
    assert hasattr(solver, "solve")


class _StubSolver(BaseIKSolver):
    """Concrete solver for testing BaseIKSolver helper methods."""

    def solve(self, markers, rig, weights=None, config=None):  # type: ignore[no-untyped-def]
        return JointTrajectory(
            id="stub",
            skeleton=rig,
            frames=[],
        )

    def solve_frame(self, markers, rig, weights=None):  # type: ignore[no-untyped-def]
        return [0.0] * rig.num_dofs


def test_base_ik_solver_apply_weights_no_weights_returns_ones() -> None:
    s = _StubSolver()
    out = s._apply_weights(["A", "B", "C"], None)
    assert out == [1.0, 1.0, 1.0]


def test_base_ik_solver_apply_weights_with_overrides() -> None:
    s = _StubSolver()
    w = MarkerWeights(default_weight=1.0, marker_weights={"B": 3.0})
    out = s._apply_weights(["A", "B", "C"], w)
    assert out == [1.0, 3.0, 1.0]


def test_base_ik_solver_clamp_to_limits_clamps_high_value() -> None:
    s = _StubSolver()
    rig = make_3dof_phantom_rig()
    # 10.0 exceeds upper bound 1.5 for link1
    q = [0.0] * (rig.num_dofs - 1) + [10.0]
    clamped = s._clamp_to_limits(q, rig)
    # last value is clamped to <= 1.5 due to limits
    assert clamped[-1] <= 1.5 + 1e-6


def test_base_ik_solver_validate_result_rejects_nan() -> None:
    s = _StubSolver()
    rig = make_3dof_phantom_rig()
    q = [float("nan")] * rig.num_dofs
    assert s._validate_result(q, rig) is False


def test_base_ik_solver_validate_result_accepts_zeros() -> None:
    s = _StubSolver()
    rig = make_3dof_phantom_rig()
    q = [0.0] * rig.num_dofs
    assert s._validate_result(q, rig) is True


def test_inverse_kinematics_protocol_cannot_be_instantiated_directly() -> None:
    """Protocol has no implementation; abstract base requires methods."""
    with pytest.raises(TypeError):
        BaseIKSolver()  # type: ignore[abstract]
