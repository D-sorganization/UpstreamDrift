"""Unit tests for motion_pipeline.matching.base."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointStateFrame,
    JointTrajectory,
    MotionMatchingResult as ContractMotionMatchingResult,
    TorqueFrame,
    TorqueTrajectory,
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


def _clone_trajectory_with_frames(
    reference: JointTrajectory,
    frames: list[JointStateFrame],
    *,
    trajectory_id: str = "tracked",
) -> JointTrajectory:
    return JointTrajectory.model_construct(
        id=trajectory_id,
        skeleton=reference.skeleton,
        frames=frames,
        metadata={},
    )


def _make_dummy_solver() -> BaseMotionMatchingSolver:
    class _Dummy(BaseMotionMatchingSolver):
        def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
            return MotionMatchingResult(
                request_id="x",
                success=False,
                message="dummy solver did not run",
            )

    return _Dummy()


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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("joint_tracking", -0.1),
        ("marker_tracking", float("nan")),
        ("smoothness", float("inf")),
    ],
)
def test_cost_weights_reject_invalid_numeric_weights(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field_name):
        CostWeights(**{field_name: value})


def test_cost_weights_allow_all_zero_weights_for_diagnostic_runs() -> None:
    weights = CostWeights(
        joint_tracking=0.0,
        marker_tracking=0.0,
        smoothness=0.0,
        effort=0.0,
        contact=0.0,
        residual=0.0,
    )
    assert weights.joint_tracking == 0.0


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


def test_make_matching_solver_pinocchio_inverse_dyn_returns_solver() -> None:
    """The inverse_dyn_pinocchio module exists and make_matching_solver returns a solver.

    This test was originally written when the pinocchio module was absent from the
    source tree. The module has since been written; the test now verifies it returns a
    solver that implements the expected `match` interface.
    """
    solver = make_matching_solver(MatchingBackendType.INVERSE_DYN_PINOCCHIO)
    assert hasattr(solver, "match")


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


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"id": "   "}, "id"),
        ({"time_horizon": 0.0}, "time_horizon"),
        ({"integrator_step": -0.01}, "integrator_step"),
        ({"max_iterations": 0}, "max_iterations"),
    ],
)
def test_motion_matching_request_rejects_invalid_solver_configuration(
    kwargs: dict[str, object], field_name: str
) -> None:
    rig = make_simple_rig(num_joints=1)
    ref = make_pendulum_reference_trajectory(num_frames=10)
    params = {"id": "r1", "reference": ref, "rig": rig}
    params.update(kwargs)
    with pytest.raises(ValueError, match=field_name):
        MotionMatchingRequest(**params)  # type: ignore[arg-type]


def test_motion_matching_request_rejects_empty_reference_frames() -> None:
    rig = make_simple_rig(num_joints=1)
    ref = JointTrajectory.model_construct(id="empty", skeleton=rig, frames=[])
    with pytest.raises(ValueError, match="reference.frames"):
        MotionMatchingRequest(id="r1", reference=ref, rig=rig)


def test_motion_matching_result_to_contract_round_trip() -> None:
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
    assert contract.matched_trajectory is not None
    assert contract.matched_trajectory.trajectory is ref


def test_motion_matching_result_rejects_success_without_payload() -> None:
    with pytest.raises(ValueError, match="success.*tracked_trajectory"):
        MotionMatchingResult(request_id="r1", success=True, message="ok")


def test_motion_matching_result_rejects_blank_request_id() -> None:
    ref = make_pendulum_reference_trajectory(num_frames=5)
    with pytest.raises(ValueError, match="request_id"):
        MotionMatchingResult(request_id=" ", success=True, tracked_trajectory=ref)


def test_motion_matching_result_requires_failed_message() -> None:
    with pytest.raises(ValueError, match="message"):
        MotionMatchingResult(request_id="r1", success=False)


def test_motion_matching_result_requires_mapping_metric_containers() -> None:
    ref = make_pendulum_reference_trajectory(num_frames=5)
    with pytest.raises(ValueError, match="fit_metrics"):
        MotionMatchingResult(
            request_id="r1",
            success=True,
            tracked_trajectory=ref,
            fit_metrics=[("rmse", 0.0)],  # type: ignore[arg-type]
        )


def test_base_solver_compute_rmse_zero_on_identical_trajectory() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rmse = s._compute_rmse(ref, ref)
    assert rmse == pytest.approx(0.0)


def test_base_solver_residual_report_keys() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    report = s._compute_residual_report(ref, ref)
    assert {"mean_residual", "max_residual", "std_residual", "num_frames"} <= set(
        report.keys()
    )
    assert report["mean_residual"] >= 0.0


@pytest.mark.parametrize(
    ("tracked_factory", "message"),
    [
        (
            lambda ref: _clone_trajectory_with_frames(
                ref, [*ref.frames, ref.frames[-1]], trajectory_id="extra-frame"
            ),
            "expected 5 frames, actual 6",
        ),
        (
            lambda ref: _clone_trajectory_with_frames(
                ref, ref.frames[:-1], trajectory_id="missing-frame"
            ),
            "expected 5 frames, actual 4",
        ),
        (
            lambda ref: _clone_trajectory_with_frames(
                ref,
                [
                    JointStateFrame.model_construct(
                        timestamp=f.timestamp,
                        q=[*f.q, 1.0] if i == 2 else list(f.q),
                        frame_index=f.frame_index,
                    )
                    for i, f in enumerate(ref.frames)
                ],
                trajectory_id="extra-dof",
            ),
            "frame 2.*expected 1 DOFs, actual 2",
        ),
        (
            lambda ref: _clone_trajectory_with_frames(
                ref,
                [
                    JointStateFrame.model_construct(
                        timestamp=f.timestamp,
                        q=[] if i == 2 else list(f.q),
                        frame_index=f.frame_index,
                    )
                    for i, f in enumerate(ref.frames)
                ],
                trajectory_id="missing-dof",
            ),
            "frame 2.*expected 1 DOFs, actual 0",
        ),
    ],
)
def test_base_solver_metric_helpers_reject_shape_mismatch(
    tracked_factory, message: str
) -> None:  # type: ignore[no-untyped-def]
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    tracked = tracked_factory(ref)
    with pytest.raises(ValueError, match=message):
        s._compute_rmse(ref, tracked)
    with pytest.raises(ValueError, match=message):
        s._compute_residual_report(ref, tracked)


def test_base_solver_validate_result_against_reference_accepts_valid_payload() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    result = MotionMatchingResult(
        request_id="r1",
        success=True,
        tracked_trajectory=ref,
        message="ok",
    )
    assert s._validate_result(ref, result) is True


def test_base_solver_validate_result_rejects_nan_joint_payload() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    # Use model_construct to bypass Pydantic's finite-value validator so we can
    # test that _validate_result itself detects the NaN in an already-constructed
    # trajectory (e.g. one produced by a solver that doesn't guard its outputs).
    bad_frames = [
        JointStateFrame.model_construct(
            timestamp=f.timestamp,
            q=[float("nan")] if i == 0 else list(f.q),
            frame_index=f.frame_index,
        )
        for i, f in enumerate(ref.frames)
    ]
    bad = _clone_trajectory_with_frames(
        ref,
        bad_frames,
        trajectory_id="bad",
    )
    result = MotionMatchingResult(
        request_id="r1",
        success=True,
        tracked_trajectory=bad,
        message="ok",
    )
    with pytest.raises(ValueError, match="tracked_trajectory.*frame 0.*q"):
        s._validate_result(ref, result)


def test_base_solver_validate_result_rejects_wrong_time_grid() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    shifted = _clone_trajectory_with_frames(
        ref,
        [
            JointStateFrame.model_construct(
                timestamp=f.timestamp + (0.001 if i == 2 else 0.0),
                q=list(f.q),
                frame_index=f.frame_index,
            )
            for i, f in enumerate(ref.frames)
        ],
        trajectory_id="shifted",
    )
    result = MotionMatchingResult(
        request_id="r1",
        success=True,
        tracked_trajectory=shifted,
        message="ok",
    )
    with pytest.raises(ValueError, match="time grid.*frame 2"):
        s._validate_result(ref, result)


def test_base_solver_validate_result_rejects_missing_dofs() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    missing = _clone_trajectory_with_frames(
        ref,
        [
            JointStateFrame.model_construct(
                timestamp=f.timestamp,
                q=[] if i == 1 else list(f.q),
                frame_index=f.frame_index,
            )
            for i, f in enumerate(ref.frames)
        ],
        trajectory_id="missing-dof",
    )
    result = MotionMatchingResult(
        request_id="r1",
        success=True,
        tracked_trajectory=missing,
        message="ok",
    )
    with pytest.raises(ValueError, match="frame 1.*expected 1 DOFs, actual 0"):
        s._validate_result(ref, result)


def test_base_solver_validate_result_rejects_nan_torque_payload() -> None:
    s = _make_dummy_solver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    torque = TorqueTrajectory.model_construct(
        rig_joint_names=["j0"],
        frames=[
            TorqueFrame.model_construct(
                timestamp=f.timestamp,
                tau=[float("nan") if i == 3 else 0.0],
            )
            for i, f in enumerate(ref.frames)
        ],
        metadata={},
    )
    result = MotionMatchingResult(
        request_id="r1",
        success=True,
        torque_trajectory=torque,
        message="ok",
    )
    with pytest.raises(ValueError, match="torque_trajectory.*frame 3.*tau"):
        s._validate_result(ref, result)
