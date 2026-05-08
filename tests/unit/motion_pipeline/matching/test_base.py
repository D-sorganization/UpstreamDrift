"""Unit tests for motion_pipeline.matching.base."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.contracts import (
    MotionMatchingResult as ContractMotionMatchingResult,
)
from src.shared.python.motion_pipeline.matching.base import (
    BaseMotionMatchingSolver,
    CostWeights,
    MatchingBackendType,
    MotionMatchingRequest,
    MotionMatchingResult,
    make_matching_solver,
)

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


def test_cost_weights_defaults_are_non_negative() -> None:
    w = CostWeights()
    for v in (
        w.joint_tracking,
        w.marker_tracking,
        w.smoothness,
        w.effort,
        w.contact,
        w.residual,
    ):
        assert v >= 0.0


def test_matching_backend_type_enum_coverage() -> None:
    members = {b.value for b in MatchingBackendType}
    assert {
        "cmc",
        "rra",
        "drake_trajopt",
        "mujoco_torque",
        "pinocchio_inverse_dyn",
    } == members


def test_make_matching_solver_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        make_matching_solver("not-a-backend")


@pytest.mark.parametrize(
    "backend",
    [
        MatchingBackendType.CMC,
        MatchingBackendType.RRA,
        MatchingBackendType.TRAJOPT_DRAKE,
        MatchingBackendType.TORQUE_MUJOCO,
    ],
)
def test_make_matching_solver_returns_protocol(backend: MatchingBackendType) -> None:
    solver = make_matching_solver(backend)
    assert hasattr(solver, "match")


def test_make_matching_solver_pinocchio_inverse_dyn_module_missing() -> None:
    """The inverse_dyn_pinocchio module is not present in the source tree."""
    with pytest.raises((ImportError, ModuleNotFoundError)):
        make_matching_solver(MatchingBackendType.INVERSE_DYN_PINOCCHIO)


def test_motion_matching_request_post_init_sets_horizon() -> None:
    rig = make_simple_rig(num_joints=1)
    ref = make_pendulum_reference_trajectory(num_frames=10)
    req = MotionMatchingRequest(id="r1", reference=ref, rig=rig)
    # __post_init__ should set time_horizon to reference duration
    assert req.time_horizon == pytest.approx(ref.duration)


def test_motion_matching_request_explicit_horizon_preserved() -> None:
    rig = make_simple_rig(num_joints=1)
    ref = make_pendulum_reference_trajectory(num_frames=10)
    req = MotionMatchingRequest(id="r1", reference=ref, rig=rig, time_horizon=5.0)
    assert req.time_horizon == 5.0


def test_motion_matching_result_to_contract_round_trip() -> None:
    rig = make_simple_rig(num_joints=1)
    ref = make_pendulum_reference_trajectory(num_frames=5)
    result = MotionMatchingResult(
        request_id="r1",
        success=True,
        tracked_trajectory=ref,
        fit_metrics={"rmse": 0.0},
        message="ok",
    )
    contract = result.to_contract()
    assert isinstance(contract, ContractMotionMatchingResult)
    assert contract.request_id == "r1"
    assert contract.success is True
    assert contract.matched_trajectory is ref


def test_base_solver_compute_rmse_zero_on_identical_trajectory() -> None:
    class _Dummy(BaseMotionMatchingSolver):
        def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
            return MotionMatchingResult(request_id="x", success=True)

    s = _Dummy()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rmse = s._compute_rmse(ref, ref)
    assert rmse == pytest.approx(0.0)


def test_base_solver_residual_report_keys() -> None:
    class _Dummy(BaseMotionMatchingSolver):
        def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
            return MotionMatchingResult(request_id="x", success=True)

    s = _Dummy()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    report = s._compute_residual_report(ref, ref)
    assert {"mean_residual", "max_residual", "std_residual", "num_frames"} <= set(
        report.keys()
    )
    assert report["mean_residual"] >= 0.0


def test_base_solver_validate_result_rejects_nan() -> None:
    class _Dummy(BaseMotionMatchingSolver):
        def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
            return MotionMatchingResult(request_id="x", success=True)

    from src.shared.python.motion_pipeline.contracts import (
        JointStateFrame,
        JointTrajectory,
    )

    s = _Dummy()
    rig = make_simple_rig(num_joints=1)
    bad = JointTrajectory(
        id="bad",
        skeleton=rig,
        frames=[JointStateFrame(timestamp=0.0, q=[float("nan")])],
    )
    assert s._validate_result(bad) is False
