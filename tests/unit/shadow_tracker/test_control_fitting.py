"""Unit and contract tests for Shadow Tracker control fitting (ST-08, #10131).

Validates:
1. OptimizationConfig validation and DbC invariants.
2. Objective breakdown and valid-pixel silhouette loss weighting.
3. Checkpoint recording and serialization.
4. Known-control synthetic recovery under forward dynamics.
5. Physics priority: lower image loss cannot override failed physics (Gate G4).
6. Execution budget exhaustion (time and iteration limits).
7. Graceful cancellation handling.
8. Divergent / non-finite dynamics resilience.
9. Fresh independent replay audit scoring.
"""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
import pytest

from src.shared.python.shadow_tracker.contracts import (
    CandidateResult,
    ModelCapabilities,
    POINT_LANDMARKS_CONVENTION,
    ReplayAudit,
    RenderRequest,
    RenderResult,
    RolloutRequest,
    RolloutResult,
)
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.projection import (
    AnalyticSilhouetteRenderer,
    PinholeCameraModel,
)
from src.shared.python.shadow_tracker.fitting import (
    BaselineComparisonResult,
    ControlFitter,
    FittingOutcome,
    ObjectiveBreakdown,
    OptimizationCheckpoint,
    OptimizationConfig,
    compare_equal_budget_baselines,
)
from src.shared.python.shadow_tracker.forward_model import FullBodyForwardModel

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Test Fixtures & Minimal Test Doubles
# ---------------------------------------------------------------------------


class MockForwardModel:
    """Mock dynamic forward model for unit testing control optimization."""

    def __init__(
        self,
        *,
        fail_physics: bool = False,
        diverge_on_call: bool = False,
        closure_error_m: float = 0.0005,
    ) -> None:
        self.fail_physics = fail_physics
        self.diverge_on_call = diverge_on_call
        self.closure_error_m = closure_error_m
        self.rollout_count = 0

    def capabilities(self) -> ModelCapabilities:
        """Return declared test model capabilities."""
        return ModelCapabilities(
            supported_bodies=tuple(f"b_{i}" for i in range(41)),
            state_convention=POINT_LANDMARKS_CONVENTION,
            actuator_modes=("torque_polynomial_deg6",),
            contact_modes=("hunt_crossley_regularized_coulomb",),
            is_available=True,
        )

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        self.rollout_count += 1
        if self.diverge_on_call:
            raise FloatingPointError("Numerical instability in forward simulation")

        times = np.asarray(request.time_points_s, dtype=np.float64)
        n_steps = len(times)
        controls_arr = np.asarray(request.controls, dtype=np.float64)

        # Simple linear response for testing: position shifts with control value
        # State dimension: 6 (3 body landmark, 3 club landmark)
        tau_mean = float(np.mean(controls_arr))
        traj: list[tuple[float, ...]] = []
        for t in times:
            # Body landmark at (0.0, tau_mean * t, 1.0)
            # Club landmark at (0.0, tau_mean * t, 0.2)
            pos_y = tau_mean * float(t)
            traj.append((0.0, pos_y, 1.0, 0.0, pos_y, 0.2))

        is_phys = (not self.fail_physics) and (self.closure_error_m <= 0.001)
        audit = ReplayAudit(
            schema_version="shadow-tracker/replay-audit/1.0.0",
            candidate_id=f"cand_{self.rollout_count}",
            reset_count=1,
            integrator_name="mock_integrator",
            integrator_version="1.0.0",
            coverage_start_s=float(times[0]),
            coverage_end_s=float(times[-1]),
            max_grip_translation_error_m=self.closure_error_m,
            max_grip_rotation_error_rad=0.001,
            is_physically_accepted=is_phys,
        )

        return RolloutResult(
            trajectory=tuple(traj),
            realized_controls=request.controls,
            time_points_s=request.time_points_s,
            audit=audit,
        )


def _create_test_camera() -> PinholeCameraModel:
    return PinholeCameraModel(
        camera_id="cam_01",
        width_px=64,
        height_px=64,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=32.0,
        translation_world_to_camera=(0.0, 0.0, 2.0),
    )


from src.shared.python.shadow_tracker.source_records import FrameIdentity


def _create_frame_identity(frame_id: str, camera_id: str) -> FrameIdentity:
    return FrameIdentity(
        schema_version="shadow-tracker/frame/1.0.0",
        asset_id="asset_01",
        shot_id="shot_01",
        swing_id="swing_01",
        camera_id=camera_id,
        frame_id=frame_id,
        pts_ticks=0,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.0,
        physical_time_reason="synthetic_fixture",
        frame_sha256="0" * 64,
    )


from src.shared.python.shadow_tracker._validation import MASK_SCHEMA_VERSION


def _render_mask_frame(
    renderer: AnalyticSilhouetteRenderer,
    camera: PinholeCameraModel,
    state: tuple[float, ...],
    frame_id: str,
) -> MaskFrame:
    res = renderer.render(
        RenderRequest(
            camera_id=camera.camera_id,
            state=state,
            image_size_px=(camera.width_px, camera.height_px),
            state_convention=POINT_LANDMARKS_CONVENTION,
        )
    )
    return MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=_create_frame_identity(frame_id, camera.camera_id),
        width_px=camera.width_px,
        height_px=camera.height_px,
        body=bytes(res.body_mask),
        club=bytes(res.club_mask),
        valid=bytes(res.visibility_mask),
        revision_id="rev_01",
        parent_revision_id=None,
        producer_id="test_producer",
        correction_note="synthetic",
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_optimization_config_validation() -> None:
    """Validate OptimizationConfig preconditions and default values."""
    config = OptimizationConfig(
        budget_seconds=5.0,
        max_iterations=20,
        body_weight=0.6,
        club_weight=0.4,
    )
    assert config.budget_seconds == 5.0
    assert config.max_iterations == 20
    assert config.body_weight == 0.6
    assert config.club_weight == 0.4

    with pytest.raises(ValueError, match="budget_seconds must be positive"):
        OptimizationConfig(budget_seconds=0.0)

    with pytest.raises(ValueError, match="max_iterations must be positive"):
        OptimizationConfig(max_iterations=0)

    with pytest.raises(ValueError, match="body_weight must be non-negative"):
        OptimizationConfig(body_weight=-0.1)


def test_objective_breakdown_immutability() -> None:
    """Validate ObjectiveBreakdown creation and invariants."""
    breakdown = ObjectiveBreakdown(
        total_loss=0.15,
        silhouette_loss=0.10,
        body_iou=0.92,
        club_iou=0.88,
        torque_regularization=0.05,
        physics_penalty=0.0,
        is_physically_accepted=True,
    )
    assert breakdown.total_loss == 0.15
    assert breakdown.is_physically_accepted is True

    with pytest.raises((AttributeError, TypeError)):
        breakdown.total_loss = 0.20  # type: ignore[misc]


def test_checkpoint_recording() -> None:
    """Validate OptimizationCheckpoint field capture."""
    ckpt = OptimizationCheckpoint(
        step_index=1,
        elapsed_seconds=0.12,
        current_loss=0.45,
        image_loss=0.35,
        physics_loss=0.10,
        best_loss=0.45,
        is_physically_accepted=True,
        parameters=((0.1, 0.2), (0.3, 0.4)),
    )
    assert ckpt.step_index == 1
    assert ckpt.elapsed_seconds == 0.12
    assert ckpt.is_physically_accepted is True
    assert ckpt.parameters == ((0.1, 0.2), (0.3, 0.4))


def test_known_control_synthetic_recovery() -> None:
    """Validate that optimization recovers a known control parameter delta."""
    camera = _create_test_camera()
    renderer = AnalyticSilhouetteRenderer(
        {camera.camera_id: camera},
        body_radius_m=0.1,
        club_radius_m=0.05,
    )
    model = MockForwardModel()

    times = (0.0, 0.05, 0.10)
    init_state = (0.0, 0.0, 1.0, 0.0, 0.0, 0.2)

    # True control value = 0.5
    true_controls = tuple((0.5,) * 7 for _ in range(41))
    true_rollout = model.rollout(
        RolloutRequest(
            initial_state=init_state,
            controls=true_controls,
            time_points_s=times,
        )
    )

    observed_masks: list[MaskFrame] = []
    for k, st in enumerate(true_rollout.trajectory):
        observed_masks.append(_render_mask_frame(renderer, camera, st, f"frame_{k}"))

    # Perturbed initial control = 0.2
    perturbed_controls = tuple((0.2,) * 7 for _ in range(41))

    config = OptimizationConfig(
        budget_seconds=10.0,
        max_iterations=30,
        body_weight=0.7,
        club_weight=0.3,
        torque_reg_weight=1e-5,
    )

    fitter = ControlFitter(
        forward_model=model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
        config=config,
    )

    outcome = fitter.fit(initial_controls=perturbed_controls)

    assert outcome.status in ("completed", "budget_exhausted")
    assert outcome.candidate.is_accepted is True
    assert outcome.candidate.replay_audit is not None
    assert outcome.candidate.replay_audit.is_physically_accepted is True
    # Final image loss must be substantially lower than initial perturbed loss
    assert outcome.final_loss.silhouette_loss < 0.15
    assert outcome.final_loss.body_iou > 0.85


def test_physics_priority_over_image_loss() -> None:
    """Validate that a candidate violating physics CANNOT be accepted regardless of image loss."""
    camera = _create_test_camera()
    renderer = AnalyticSilhouetteRenderer(
        {camera.camera_id: camera},
        body_radius_m=0.1,
        club_radius_m=0.05,
    )
    # Model configured with excessive closure error (fails Gate G4 physics)
    model = MockForwardModel(closure_error_m=0.05, fail_physics=True)

    times = (0.0, 0.05)
    init_state = (0.0, 0.0, 1.0, 0.0, 0.0, 0.2)
    controls = tuple((0.0,) * 7 for _ in range(41))

    rollout = model.rollout(
        RolloutRequest(
            initial_state=init_state,
            controls=controls,
            time_points_s=times,
        )
    )
    observed_masks = [
        _render_mask_frame(renderer, camera, st, f"f_{k}")
        for k, st in enumerate(rollout.trajectory)
    ]

    config = OptimizationConfig(
        budget_seconds=5.0,
        max_iterations=10,
    )

    fitter = ControlFitter(
        forward_model=model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
        config=config,
    )

    outcome = fitter.fit(initial_controls=controls)

    # Even though silhouette loss might be low or zero, physics failed!
    assert outcome.candidate.is_accepted is False
    assert outcome.candidate.replay_audit is not None
    assert outcome.candidate.replay_audit.is_physically_accepted is False


def test_execution_budget_exhaustion() -> None:
    """Validate that fitter respects small budget and sets status='budget_exhausted'."""
    camera = _create_test_camera()
    renderer = AnalyticSilhouetteRenderer({camera.camera_id: camera})
    model = MockForwardModel()

    times = (0.0, 0.05)
    init_state = (0.0, 0.0, 1.0, 0.0, 0.0, 0.2)
    controls = tuple((0.0,) * 7 for _ in range(41))
    observed_masks = [
        _render_mask_frame(renderer, camera, init_state, "f_0"),
        _render_mask_frame(renderer, camera, init_state, "f_1"),
    ]

    # Extremely small budget (0.0001s)
    config = OptimizationConfig(
        budget_seconds=0.0001,
        max_iterations=500,
    )

    fitter = ControlFitter(
        forward_model=model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
        config=config,
    )

    outcome = fitter.fit(initial_controls=controls)
    assert outcome.status == "budget_exhausted"


def test_cancellation_handling() -> None:
    """Validate that cancellation callback stops execution cleanly."""
    camera = _create_test_camera()
    renderer = AnalyticSilhouetteRenderer({camera.camera_id: camera})
    model = MockForwardModel()

    times = (0.0, 0.05)
    init_state = (0.0, 0.0, 1.0, 0.0, 0.0, 0.2)
    controls = tuple((0.0,) * 7 for _ in range(41))
    observed_masks = [
        _render_mask_frame(renderer, camera, init_state, "f_0"),
        _render_mask_frame(renderer, camera, init_state, "f_1"),
    ]

    config = OptimizationConfig(
        budget_seconds=10.0,
        max_iterations=50,
    )

    call_count = 0

    def cancel_fn() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 2

    fitter = ControlFitter(
        forward_model=model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
        config=config,
        cancel_callback=cancel_fn,
    )

    outcome = fitter.fit(initial_controls=controls)
    assert outcome.status == "cancelled"


def test_divergent_dynamics_resilience() -> None:
    """Validate that non-finite or crashing dynamics return honest failed status."""
    camera = _create_test_camera()
    renderer = AnalyticSilhouetteRenderer({camera.camera_id: camera})
    model = MockForwardModel(diverge_on_call=True)

    times = (0.0, 0.05)
    init_state = (0.0, 0.0, 1.0, 0.0, 0.0, 0.2)
    controls = tuple((0.0,) * 7 for _ in range(41))
    observed_masks = [
        _render_mask_frame(renderer, camera, init_state, "f_0"),
        _render_mask_frame(renderer, camera, init_state, "f_1"),
    ]

    config = OptimizationConfig(
        budget_seconds=5.0,
        max_iterations=5,
    )

    fitter = ControlFitter(
        forward_model=model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
        config=config,
    )

    outcome = fitter.fit(initial_controls=controls)
    assert outcome.status == "failed"
    assert outcome.candidate.is_accepted is False


def test_compare_equal_budget_baselines() -> None:
    """Validate equal-budget comparison demonstrates physics failure for kinematic baseline."""
    camera = _create_test_camera()
    renderer = AnalyticSilhouetteRenderer(
        {camera.camera_id: camera},
        body_radius_m=0.1,
        club_radius_m=0.05,
    )
    model = MockForwardModel()

    times = (0.0, 0.05)
    init_state = (0.0, 0.0, 1.0, 0.0, 0.0, 0.2)
    controls = tuple((0.5,) * 7 for _ in range(41))
    rollout = model.rollout(
        RolloutRequest(
            initial_state=init_state,
            controls=controls,
            time_points_s=times,
        )
    )
    observed_masks = [
        _render_mask_frame(renderer, camera, st, f"f_{k}")
        for k, st in enumerate(rollout.trajectory)
    ]

    fitter = ControlFitter(
        forward_model=model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
    )

    comp = compare_equal_budget_baselines(fitter, budget_seconds=2.0)
    assert isinstance(comp, BaselineComparisonResult)
    assert comp.budget_seconds == 2.0
    assert comp.kinematic_physically_accepted is False
    assert comp.forward_dynamics_physically_accepted is True


class MockSkeletalModel:
    """Mock full-body skeletal dynamics model with 41 coordinates."""

    def __init__(self, ground_height: float = 0.0) -> None:
        self.coordinate_order: list[str] = [f"coord_{i}" for i in range(41)]
        self.ground_plane = _MockGround(ground_height)
        self.resets = 0

    def accelerations(
        self, q: dict[str, float], qd: dict[str, float], tau: dict[str, float]
    ) -> dict[str, float]:
        return {
            name: -0.1 * qd[name] + tau.get(name, 0.0) for name in self.coordinate_order
        }

    def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
        err_p = np.array([0.0005, 0.0, 0.0, 0.005, 0.0, 0.0], dtype=np.float64)
        err_v = np.zeros(6, dtype=np.float64)
        return err_p, err_v

    def evaluate_contact_samples(
        self, q: dict[str, float], qd: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "sphere_heel_r": _MockContactSample(
                normal_force_n=np.array([0.0, 0.0, 450.0]),
                friction_force_n=np.array([20.0, 10.0, 0.0]),
                penetration_m=0.0005,
            )
        }


class _MockGround:
    def __init__(self, height_m: float = 0.0) -> None:
        self.normal = (0.0, 0.0, 1.0)
        self.height_m = height_m


class _MockContactSample:
    def __init__(
        self,
        normal_force_n: np.ndarray,
        friction_force_n: np.ndarray,
        penetration_m: float,
    ) -> None:
        self.normal_force_n = normal_force_n
        self.friction_force_n = friction_force_n
        self.penetration_m = penetration_m


class MockArticulatedRenderer:
    """Mock renderer accepting canonical articulated states for testing ControlFitter."""

    def __init__(self, camera: PinholeCameraModel) -> None:
        self.camera = camera

    def render(self, request: RenderRequest) -> RenderResult:
        w, h = request.image_size_px
        total_px = w * h
        body_mask = [1 if i < total_px // 2 else 0 for i in range(total_px)]
        club_mask = [0] * total_px
        vis_mask = [1] * total_px
        return RenderResult(
            body_mask=tuple(body_mask),
            club_mask=tuple(club_mask),
            visibility_mask=tuple(vis_mask),
        )


def test_control_fitting_with_full_body_forward_model() -> None:
    """Validate ControlFitter integration with real FullBodyForwardModel."""
    camera = _create_test_camera()
    renderer = MockArticulatedRenderer(camera)
    skeletal_model = MockSkeletalModel()
    fb_forward_model = FullBodyForwardModel(skeletal_model)

    times = (0.0, 0.02)
    # 42-element canonical initial state
    init_state = (0.0, 0.0, 0.0, 1.0) + tuple(0.0 for _ in range(38))
    controls = tuple((0.0,) * 7 for _ in range(41))

    res_0 = renderer.render(
        RenderRequest(
            camera_id=camera.camera_id,
            state=init_state,
            image_size_px=(camera.width_px, camera.height_px),
            state_convention="canonical_articulated_v1",
        )
    )
    observed_masks = [
        MaskFrame(
            schema_version=MASK_SCHEMA_VERSION,
            frame=_create_frame_identity("f_0", camera.camera_id),
            width_px=camera.width_px,
            height_px=camera.height_px,
            body=bytes(res_0.body_mask),
            club=bytes(res_0.club_mask),
            valid=bytes(res_0.visibility_mask),
            revision_id="rev_01",
            parent_revision_id=None,
            producer_id="test_producer",
            correction_note="synthetic",
        ),
        MaskFrame(
            schema_version=MASK_SCHEMA_VERSION,
            frame=_create_frame_identity("f_1", camera.camera_id),
            width_px=camera.width_px,
            height_px=camera.height_px,
            body=bytes(res_0.body_mask),
            club=bytes(res_0.club_mask),
            valid=bytes(res_0.visibility_mask),
            revision_id="rev_01",
            parent_revision_id=None,
            producer_id="test_producer",
            correction_note="synthetic",
        ),
    ]

    fitter = ControlFitter(
        forward_model=fb_forward_model,
        renderer=renderer,
        camera=camera,
        observed_masks=observed_masks,
        time_points_s=times,
        initial_state=init_state,
        config=OptimizationConfig(budget_seconds=5.0, max_iterations=2),
    )

    outcome = fitter.fit(initial_controls=controls)
    assert outcome.status in ("completed", "budget_exhausted")
    assert len(outcome.candidate.trajectory[0]) == 42
    assert outcome.candidate.replay_audit is not None
    assert outcome.candidate.replay_audit.is_physically_accepted is True
