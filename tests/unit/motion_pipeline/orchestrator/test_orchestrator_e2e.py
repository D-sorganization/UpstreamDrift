"""End-to-end orchestrator run on a synthetic MarkerTrajectory (#7047).

Exercises the full adapter -> preprocessing -> scaling -> IK -> motion-
matching chain using the real, dependency-free ``geometric`` IK backend and
the matching backends that are installed. Asserts the run completes with a
finite-metric :class:`MotionMatchingResult` - the acceptance criterion for
issue #7047.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _have(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def _have_real_pinocchio() -> bool:
    """True only when the real Pinocchio C extension is importable.

    The repo's engine-isolation fixtures often leave a ``MagicMock`` in
    ``sys.modules['pinocchio']``; ``pin.Model().nq`` then returns a mock
    rather than an int. Mirror the matching_completeness guard.
    """
    try:
        import pinocchio as pin  # type: ignore[import-not-found]

        return isinstance(pin.Model().nq, int)
    except Exception:  # noqa: BLE001 - any failure means "not real"
        return False


_HAVE_PINOCCHIO = _have_real_pinocchio()
_HAVE_MUJOCO = _have("mujoco")

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    MotionMatchingResult,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.ik.geometric_backend import forward_kinematics
from src.shared.python.motion_pipeline.orchestrator import (
    AdapterOverride,
    InvalidInputError,
    MotionPipeline,
    PipelineConfig,
)


def _synthetic_rig() -> SkeletonRig:
    """A 3-joint serial chain (1 DOF each about Z) with marker-named joints."""
    joints = {
        "root": JointDef(
            name="root",
            parent=None,
            children=["mid"],
            tpose_offset=[0.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.0, upper=3.0)],
        ),
        "mid": JointDef(
            name="mid",
            parent="root",
            children=["tip"],
            tpose_offset=[0.4, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.0, upper=3.0)],
        ),
        "tip": JointDef(
            name="tip",
            parent="mid",
            children=[],
            tpose_offset=[0.3, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.0, upper=3.0)],
        ),
    }
    return SkeletonRig(id="e2e-rig", joints=joints, root_joint="root")


def _synthetic_markers(rig: SkeletonRig, num_frames: int = 12) -> MarkerTrajectory:
    """Forward-generate marker positions from a smooth joint motion."""
    frames = []
    for i in range(num_frames):
        t = i / 60.0
        q = [0.2 * math.sin(2 * math.pi * t), 0.15 * math.cos(2 * math.pi * t), 0.0]
        pos = forward_kinematics(rig, q)
        markers = {
            name: Marker(name=name, x=p[0], y=p[1], z=p[2]) for name, p in pos.items()
        }
        frames.append(MarkerFrame(timestamp=t, markers=markers, frame_index=i))
    return MarkerTrajectory(id="e2e-traj", frames=frames)


class _PipelineWithDefaultRig(MotionPipeline):
    """Pipeline variant that supplies a default skeleton for marker input.

    The stock orchestrator's default-skeleton hook is tracked separately
    (#4649); injecting one here lets the matching-wiring fix (#7047) be
    exercised end-to-end without depending on that work.
    """

    def __init__(self, config: PipelineConfig, rig: SkeletonRig) -> None:
        super().__init__(config)
        self._default_rig = rig

    def _get_default_skeleton(self) -> SkeletonRig:
        return self._default_rig


@pytest.mark.skipif(not _HAVE_PINOCCHIO, reason="pinocchio not installed")
def test_orchestrator_full_run_finite_metrics_pinocchio() -> None:
    """Pinocchio matching requires a production URDF to claim readiness."""
    rig = _synthetic_rig()
    traj = _synthetic_markers(rig)

    config = PipelineConfig(
        adapter=AdapterOverride(format="passthrough"),
        ik_backend="geometric",
        matching_backend="pinocchio",
    )
    pipeline = _PipelineWithDefaultRig(config, rig)
    with pytest.raises(InvalidInputError, match="matching_model_urdf"):
        pipeline.run(traj)


def test_orchestrator_matching_routes_through_make_matching_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matching stage uses make_matching_solver().match().to_contract().

    Engine-independent regression for #7047: the old code imported a
    nonexistent ``run_matching`` symbol. We stub the matching base so the
    test never depends on a physics engine, and assert the orchestrator
    produces a finite-metric, successful :class:`MotionMatchingResult`.
    """
    import src.shared.python.motion_pipeline.matching.base as matching_base

    rig = _synthetic_rig()
    traj = _synthetic_markers(rig)

    captured: dict = {}

    class _FakeResult:
        def to_contract(self) -> MotionMatchingResult:
            return MotionMatchingResult(
                request_id="fake",
                success=True,
                error_metrics={"rmse": 0.123, "max_error": 0.456},
                message="fake solve OK",
            )

    class _FakeSolver:
        def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
            captured["reference_frames"] = reference.num_frames
            captured["rig_dofs"] = rig.num_dofs
            return _FakeResult()

    def _fake_factory(backend, cost_weights=None, *, urdf_path=None):  # type: ignore[no-untyped-def]
        captured["backend"] = backend
        captured["urdf_path"] = urdf_path
        return _FakeSolver()

    monkeypatch.setattr(matching_base, "make_matching_solver", _fake_factory)

    config = PipelineConfig(
        adapter=AdapterOverride(format="passthrough"),
        ik_backend="geometric",
        matching_backend="pinocchio",
    )
    pipeline = _PipelineWithDefaultRig(config, rig)
    result = pipeline.run(traj)

    assert isinstance(result, MotionMatchingResult)
    assert result.success is True
    assert result.error_metrics == {"rmse": 0.123, "max_error": 0.456}
    for value in result.error_metrics.values():
        assert np.isfinite(value)
    # The reference passed to the solver is the IK JointTrajectory.
    assert captured["rig_dofs"] == rig.num_dofs
    assert captured["reference_frames"] == traj.num_frames
    assert captured["backend"].value == "pinocchio_inverse_dyn"
    assert captured["urdf_path"] is None


@pytest.mark.unit
def test_mujoco_placeholder_matching_failure_is_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MuJoCo placeholder path must surface as a 4xx-class failure."""
    import src.shared.python.motion_pipeline.matching.base as matching_base

    rig = _synthetic_rig()
    traj = _synthetic_markers(rig)

    class _FakeResult:
        def to_contract(self) -> MotionMatchingResult:
            return MotionMatchingResult(
                request_id="fake",
                success=False,
                error_metrics={"rmse": 0.0, "max_error": 0.0, "max_torque": 0.0},
                message=(
                    "MuJoCo present but produced zero torques from the placeholder "
                    "model (no real solve)"
                ),
                metadata={
                    "backend": "mujoco_torque",
                    "mujoco_available": True,
                    "n_frames": traj.num_frames,
                },
            )

    class _FakeSolver:
        def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
            return _FakeResult()

    monkeypatch.setattr(
        matching_base,
        "make_matching_solver",
        lambda backend, cost_weights=None, *, urdf_path=None: _FakeSolver(),
    )

    config = PipelineConfig(
        adapter=AdapterOverride(format="passthrough"),
        ik_backend="geometric",
        matching_backend="mujoco",
    )
    pipeline = _PipelineWithDefaultRig(config, rig)
    with pytest.raises(InvalidInputError, match="no real solve"):
        pipeline.run(traj)


@pytest.mark.skipif(not _HAVE_MUJOCO, reason="mujoco not installed")
def test_orchestrator_mujoco_matching_uses_generated_rig_model() -> None:
    """The MuJoCo torque backend no longer runs on an empty placeholder model."""
    rig = _synthetic_rig()
    traj = _synthetic_markers(rig)

    config = PipelineConfig(
        adapter=AdapterOverride(format="passthrough"),
        ik_backend="geometric",
        matching_backend="mujoco",
    )
    pipeline = _PipelineWithDefaultRig(config, rig)
    result = pipeline.run(traj)

    assert result.success is True
    assert result.metadata["model_source"] == "generated_mjcf"
    assert result.metadata["placeholder_model"] is False
    assert result.metadata["model_nq"] == rig.num_dofs
