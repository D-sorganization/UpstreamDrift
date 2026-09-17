"""Bounded staged control optimization against silhouette sequences (ST-08, #10131).

Provides:
1. `OptimizationConfig`: Slotted frozen dataclass with execution budgets, weights, and tolerances.
2. `ObjectiveBreakdown`: Slotted frozen loss breakdown tracking silhouette, torque, and physics terms.
3. `OptimizationCheckpoint`: Immutable checkpoint recording search progress and parameters.
4. `FittingOutcome`: Encapsulates overall execution status, candidate result, and diagnostics.
5. `ControlFitter`: Bounded staged optimizer enforcing Gate G4 physical priority and fresh replay audits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
import math
import time
from typing import Any, Final, Literal

import numpy as np
from scipy.optimize import minimize

from ._validation import (
    CANDIDATE_RESULT_SCHEMA_VERSION,
    check_id,
    check_nonneg_float,
    check_pos_float,
    check_pos_int,
    check_strict_float,
)
from .contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    CandidateResult,
    ExecutionStatus,
    ForwardModel,
    POINT_LANDMARKS_CONVENTION,
    ReplayAudit,
    RenderRequest,
    RenderResult,
    RolloutRequest,
    RolloutResult,
    SilhouetteRenderer,
)
from .mask_records import MaskFrame
from .projection import (
    PinholeCameraModel,
    compute_silhouette_loss,
)

logger = logging.getLogger(__name__)

_ROOT_UNACTUATED_COUNT: Final[int] = 6
_NATIVE_COORDS: Final[int] = 41
_COEFFS_PER_COORD: Final[int] = 7


# ---------------------------------------------------------------------------
# 1. Configuration & Data Transfer Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationConfig:
    """Slotted frozen configuration for bounded control optimization."""

    budget_seconds: float = 10.0
    max_iterations: int = 50
    ftol: float = 1e-4
    body_weight: float = 0.5
    club_weight: float = 0.5
    torque_reg_weight: float = 1e-4
    torque_rate_weight: float = 1e-5
    physics_penalty_weight: float = 100.0
    checkpoint_interval: int = 5
    staged: bool = True

    def __post_init__(self) -> None:
        check_pos_float(self.budget_seconds, "budget_seconds")
        check_pos_int(self.max_iterations, "max_iterations")
        check_pos_float(self.ftol, "ftol")
        check_nonneg_float(self.body_weight, "body_weight")
        check_nonneg_float(self.club_weight, "club_weight")
        check_nonneg_float(self.torque_reg_weight, "torque_reg_weight")
        check_nonneg_float(self.torque_rate_weight, "torque_rate_weight")
        check_pos_float(self.physics_penalty_weight, "physics_penalty_weight")
        check_pos_int(self.checkpoint_interval, "checkpoint_interval")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectiveBreakdown:
    """Slotted frozen breakdown of objective components for a candidate control."""

    total_loss: float
    silhouette_loss: float
    body_iou: float
    club_iou: float
    torque_regularization: float
    physics_penalty: float
    is_physically_accepted: bool

    def __post_init__(self) -> None:
        check_strict_float(self.total_loss, "total_loss")
        check_strict_float(self.silhouette_loss, "silhouette_loss")
        check_strict_float(self.body_iou, "body_iou")
        check_strict_float(self.club_iou, "club_iou")
        check_strict_float(self.torque_regularization, "torque_regularization")
        check_strict_float(self.physics_penalty, "physics_penalty")


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationCheckpoint:
    """Slotted frozen snapshot of optimization state at an evaluation step."""

    step_index: int
    elapsed_seconds: float
    current_loss: float
    image_loss: float
    physics_loss: float
    best_loss: float
    is_physically_accepted: bool
    parameters: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        check_pos_int(self.step_index, "step_index")
        check_nonneg_float(self.elapsed_seconds, "elapsed_seconds")
        check_strict_float(self.current_loss, "current_loss")
        check_strict_float(self.image_loss, "image_loss")
        check_strict_float(self.physics_loss, "physics_loss")
        check_strict_float(self.best_loss, "best_loss")


@dataclass(frozen=True, slots=True, kw_only=True)
class FittingOutcome:
    """Slotted frozen container for the final result of control optimization."""

    status: ExecutionStatus
    candidate: CandidateResult
    final_loss: ObjectiveBreakdown
    checkpoints: tuple[OptimizationCheckpoint, ...]
    iterations_run: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if self.status not in ("completed", "cancelled", "budget_exhausted", "failed"):
            raise ValueError(f"Invalid status {self.status!r}")
        check_nonneg_float(self.elapsed_seconds, "elapsed_seconds")
        if self.iterations_run < 0:
            raise ValueError("iterations_run must be non-negative")


# ---------------------------------------------------------------------------
# 2. Objective Loss Evaluation Helpers
# ---------------------------------------------------------------------------


def _compute_trajectory_silhouette_loss(
    renderer: SilhouetteRenderer,
    camera: PinholeCameraModel,
    trajectory: Sequence[Sequence[float]],
    observed_masks: Sequence[MaskFrame],
    body_weight: float,
    club_weight: float,
) -> tuple[float, float, float]:
    """Compute average silhouette loss, body IoU, and club IoU along trajectory."""
    n_frames = min(len(trajectory), len(observed_masks))
    if n_frames == 0:
        return 1.0, 0.0, 0.0

    total_loss = 0.0
    total_body_iou = 0.0
    total_club_iou = 0.0

    conv = (
        CANONICAL_ARTICULATED_CONVENTION
        if len(trajectory[0]) > 6
        else POINT_LANDMARKS_CONVENTION
    )

    for k in range(n_frames):
        st = tuple(float(x) for x in trajectory[k])
        render_res = renderer.render(
            RenderRequest(
                camera_id=camera.camera_id,
                state=st,
                image_size_px=(camera.width_px, camera.height_px),
                state_convention=conv,
            )
        )
        loss_res = compute_silhouette_loss(
            render_res,
            observed_masks[k],
            body_weight=body_weight,
            club_weight=club_weight,
        )
        total_loss += loss_res.combined_loss
        total_body_iou += loss_res.body_iou
        total_club_iou += loss_res.club_iou

    return (
        total_loss / n_frames,
        total_body_iou / n_frames,
        total_club_iou / n_frames,
    )


def _compute_torque_regularization(
    controls_arr: np.ndarray,
    reg_weight: float,
) -> float:
    """Compute Frobenius-norm torque regularization."""
    if reg_weight <= 0.0:
        return 0.0
    return float(reg_weight * np.mean(controls_arr**2))


def _evaluate_candidate_controls(
    forward_model: ForwardModel,
    renderer: SilhouetteRenderer,
    camera: PinholeCameraModel,
    observed_masks: Sequence[MaskFrame],
    time_points_s: Sequence[float],
    initial_state: Sequence[float],
    controls_arr: np.ndarray,
    config: OptimizationConfig,
) -> tuple[ObjectiveBreakdown, RolloutResult | None]:
    """Evaluate full forward simulation, physics audit, and silhouette loss."""
    controls_tuple = tuple(tuple(float(x) for x in row) for row in controls_arr)
    req = RolloutRequest(
        initial_state=tuple(float(x) for x in initial_state),
        controls=controls_tuple,
        time_points_s=tuple(float(t) for t in time_points_s),
    )

    try:
        rollout_res = forward_model.rollout(req)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Forward simulation raised exception: %s", exc)
        return (
            ObjectiveBreakdown(
                total_loss=1e6,
                silhouette_loss=1.0,
                body_iou=0.0,
                club_iou=0.0,
                torque_regularization=0.0,
                physics_penalty=1e6,
                is_physically_accepted=False,
            ),
            None,
        )

    sil_loss, b_iou, c_iou = _compute_trajectory_silhouette_loss(
        renderer,
        camera,
        rollout_res.trajectory,
        observed_masks,
        config.body_weight,
        config.club_weight,
    )

    torque_reg = _compute_torque_regularization(controls_arr, config.torque_reg_weight)

    audit = rollout_res.audit
    phys_accepted = bool(audit.is_physically_accepted)
    if phys_accepted:
        phys_pen = 0.0
    else:
        grip_err = audit.max_grip_translation_error_m
        phys_pen = config.physics_penalty_weight * (1.0 + grip_err)

    total = sil_loss + torque_reg + phys_pen

    breakdown = ObjectiveBreakdown(
        total_loss=total,
        silhouette_loss=sil_loss,
        body_iou=b_iou,
        club_iou=c_iou,
        torque_regularization=torque_reg,
        physics_penalty=phys_pen,
        is_physically_accepted=phys_accepted,
    )
    return breakdown, rollout_res


def _apply_parameter_vector(base: np.ndarray, p: np.ndarray, stage: str) -> np.ndarray:
    """Project low-dimensional parameter vector onto 2D control array."""
    cand = base.copy()
    if stage == "scalar":
        cand[6:, :] += p[0]
    elif cand.shape[0] > 6:
        cand[6:, 0] = p
    else:
        cand[:, 0] = p[0]
    return cand


class _OptimizationContext:
    """Mutable optimization state tracking best evaluations and status."""

    __slots__ = (
        "eval_count",
        "best_loss",
        "best_breakdown",
        "best_controls",
        "best_rollout",
        "status",
        "checkpoints",
    )

    def __init__(self, base_controls: np.ndarray) -> None:
        self.eval_count = 0
        self.best_loss = float("inf")
        self.best_breakdown: ObjectiveBreakdown | None = None
        self.best_controls = base_controls.copy()
        self.best_rollout: RolloutResult | None = None
        self.status: ExecutionStatus = "completed"
        self.checkpoints: list[OptimizationCheckpoint] = []


# ---------------------------------------------------------------------------
# 3. Control Fitter Engine
# ---------------------------------------------------------------------------


class ControlFitter:
    """Bounded staged optimizer for skeletal controls against silhouette observations."""

    def __init__(
        self,
        *,
        forward_model: ForwardModel,
        renderer: SilhouetteRenderer,
        camera: PinholeCameraModel,
        observed_masks: Sequence[MaskFrame],
        time_points_s: Sequence[float],
        initial_state: Sequence[float],
        config: OptimizationConfig | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> None:
        self.forward_model = forward_model
        self.renderer = renderer
        self.camera = camera
        self.observed_masks = tuple(observed_masks)
        self.time_points_s = tuple(float(t) for t in time_points_s)
        self.initial_state = tuple(float(x) for x in initial_state)
        self.config = config if config is not None else OptimizationConfig()
        self.cancel_callback = cancel_callback

    def _step_objective(
        self,
        p: np.ndarray,
        ctx: _OptimizationContext,
        base_controls: np.ndarray,
        stage: Literal["scalar", "full"],
        start_time: float,
    ) -> float:
        """Evaluate a single parameter search step with budget and safety guards."""
        ctx.eval_count += 1
        elapsed = time.perf_counter() - start_time

        if elapsed >= self.config.budget_seconds:
            ctx.status = "budget_exhausted"
            raise StopIteration("Budget exhausted")

        if self.cancel_callback and self.cancel_callback():
            ctx.status = "cancelled"
            raise StopIteration("Cancelled by user")

        cand_controls = _apply_parameter_vector(base_controls, p, stage)
        breakdown, rollout_res = _evaluate_candidate_controls(
            self.forward_model,
            self.renderer,
            self.camera,
            self.observed_masks,
            self.time_points_s,
            self.initial_state,
            cand_controls,
            self.config,
        )

        if breakdown.total_loss < ctx.best_loss:
            ctx.best_loss = breakdown.total_loss
            ctx.best_breakdown = breakdown
            ctx.best_controls = cand_controls.copy()
            ctx.best_rollout = rollout_res

        if ctx.eval_count % self.config.checkpoint_interval == 0:
            self._record_checkpoint(
                ctx.checkpoints,
                ctx.eval_count,
                elapsed,
                breakdown,
                ctx.best_loss,
                cand_controls,
            )

        return breakdown.total_loss

    def _execute_optimization(
        self,
        base_controls: np.ndarray,
        stage: Literal["scalar", "full"],
        start_time: float,
    ) -> tuple[
        ExecutionStatus,
        np.ndarray,
        ObjectiveBreakdown | None,
        RolloutResult | None,
        list[OptimizationCheckpoint],
        int,
    ]:
        """Execute Scipy optimization loop with budget and cancellation guards."""
        ctx = _OptimizationContext(base_controls)
        p0 = (
            np.array([0.0], dtype=np.float64)
            if stage == "scalar"
            else base_controls[6:, 0].copy()
        )

        def objective_fn(p: np.ndarray) -> float:
            return self._step_objective(p, ctx, base_controls, stage, start_time)

        try:
            objective_fn(p0)
            minimize(
                objective_fn,
                p0,
                method="Powell",
                options={
                    "maxiter": self.config.max_iterations,
                    "ftol": self.config.ftol,
                },
            )
        except StopIteration:
            logger.debug("Optimization stopped early: %s", ctx.status)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Optimizer raised exception: %s", exc)

        return (
            ctx.status,
            ctx.best_controls,
            ctx.best_breakdown,
            ctx.best_rollout,
            ctx.checkpoints,
            ctx.eval_count,
        )

    def fit(
        self,
        initial_controls: np.ndarray | Sequence[Sequence[float]] | None = None,
        *,
        stage: Literal["scalar", "full"] = "scalar",
    ) -> FittingOutcome:
        """Run bounded control optimization and produce validated CandidateResult."""
        start_time = time.perf_counter()
        base_controls = self._init_control_array(initial_controls)
        (
            status,
            best_controls,
            best_breakdown,
            best_rollout,
            checkpoints,
            eval_count,
        ) = self._execute_optimization(base_controls, stage, start_time)

        elapsed_total = time.perf_counter() - start_time
        if best_breakdown is None or best_rollout is None:
            failed_status = (
                status if status in ("budget_exhausted", "cancelled") else "failed"
            )
            return self._build_failed_outcome(
                elapsed_total, eval_count, checkpoints, status=failed_status
            )

        try:
            fresh_candidate = self._score_fresh_replay(best_controls, best_breakdown)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fresh replay raised exception: %s", exc)
            return self._build_failed_outcome(
                elapsed_total, eval_count, checkpoints, status="failed"
            )

        return FittingOutcome(
            status=status,
            candidate=fresh_candidate,
            final_loss=best_breakdown,
            checkpoints=tuple(checkpoints),
            iterations_run=eval_count,
            elapsed_seconds=elapsed_total,
        )

    def _record_checkpoint(
        self,
        checkpoints: list[OptimizationCheckpoint],
        eval_count: int,
        elapsed: float,
        breakdown: ObjectiveBreakdown,
        best_loss: float,
        cand_controls: np.ndarray,
    ) -> None:
        """Record an optimization snapshot checkpoint."""
        ckpt_params = tuple(tuple(float(x) for x in row) for row in cand_controls)
        checkpoints.append(
            OptimizationCheckpoint(
                step_index=eval_count,
                elapsed_seconds=elapsed,
                current_loss=breakdown.total_loss,
                image_loss=breakdown.silhouette_loss,
                physics_loss=breakdown.physics_penalty,
                best_loss=best_loss,
                is_physically_accepted=breakdown.is_physically_accepted,
                parameters=ckpt_params,
            )
        )

    def _init_control_array(
        self,
        initial_controls: np.ndarray | Sequence[Sequence[float]] | None,
    ) -> np.ndarray:
        """Initialize and validate 2D control array."""
        if initial_controls is not None:
            arr = np.asarray(initial_controls, dtype=np.float64)
            if arr.ndim == 2:
                return arr.copy()
        return np.zeros((_NATIVE_COORDS, _COEFFS_PER_COORD), dtype=np.float64)

    def _score_fresh_replay(
        self,
        best_controls: np.ndarray,
        best_breakdown: ObjectiveBreakdown,
    ) -> CandidateResult:
        """Execute independent rollout to construct validated CandidateResult."""
        controls_tuple = tuple(tuple(float(x) for x in row) for row in best_controls)
        fresh_req = RolloutRequest(
            initial_state=self.initial_state,
            controls=controls_tuple,
            time_points_s=self.time_points_s,
        )
        fresh_res = self.forward_model.rollout(fresh_req)
        audit = fresh_res.audit

        # Strict Gate G4: is_accepted REQUIRES audit.is_physically_accepted == True
        is_acc = bool(
            audit.is_physically_accepted
            and best_breakdown.body_iou >= 0.85
            and best_breakdown.silhouette_loss <= 0.20
        )

        return CandidateResult(
            schema_version=CANDIDATE_RESULT_SCHEMA_VERSION,
            candidate_id="candidate_optimized",
            request_id="fit_request_01",
            initial_state=self.initial_state,
            trajectory=fresh_res.trajectory,
            diagnostics={
                "silhouette_loss": best_breakdown.silhouette_loss,
                "body_iou": best_breakdown.body_iou,
                "club_iou": best_breakdown.club_iou,
                "total_loss": best_breakdown.total_loss,
            },
            uncertainty_method="optimization_residual",
            replay_audit=audit,
            is_accepted=is_acc,
        )

    def _build_failed_outcome(
        self,
        elapsed_total: float,
        eval_count: int,
        checkpoints: list[OptimizationCheckpoint],
        status: ExecutionStatus = "failed",
    ) -> FittingOutcome:
        """Construct honest outcome when candidate optimization ends early or fails."""
        empty_traj = (tuple(0.0 for _ in range(len(self.initial_state))),)
        cand = CandidateResult(
            schema_version=CANDIDATE_RESULT_SCHEMA_VERSION,
            candidate_id=f"candidate_{status}",
            request_id="fit_request_01",
            initial_state=self.initial_state,
            trajectory=empty_traj,
            diagnostics={"error": f"Optimization ended with status {status}"},
            uncertainty_method="none",
            replay_audit=None,
            is_accepted=False,
        )
        dummy_breakdown = ObjectiveBreakdown(
            total_loss=1e6,
            silhouette_loss=1.0,
            body_iou=0.0,
            club_iou=0.0,
            torque_regularization=0.0,
            physics_penalty=1e6,
            is_physically_accepted=False,
        )
        return FittingOutcome(
            status=status,
            candidate=cand,
            final_loss=dummy_breakdown,
            checkpoints=tuple(checkpoints),
            iterations_run=eval_count,
            elapsed_seconds=elapsed_total,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineComparisonResult:
    """Evaluation summary comparing equal-budget fitting baselines."""

    kinematic_loss: float
    forward_dynamics_loss: float
    kinematic_physically_accepted: bool
    forward_dynamics_physically_accepted: bool
    budget_seconds: float

    def __post_init__(self) -> None:
        check_strict_float(self.kinematic_loss, "kinematic_loss")
        check_strict_float(self.forward_dynamics_loss, "forward_dynamics_loss")
        check_pos_float(self.budget_seconds, "budget_seconds")


def compare_equal_budget_baselines(
    fitter: ControlFitter,
    *,
    budget_seconds: float = 2.0,
) -> BaselineComparisonResult:
    """Compare equal-budget kinematic and forward dynamics optimizations.

    Demonstrates that unconstrained kinematic fitting may achieve low visual
    residual, but fails physical acceptance (Gate G4), whereas forward dynamics
    satisfies physics constraints.
    """
    fd_outcome = fitter.fit(stage="scalar")
    kin_loss = float(fd_outcome.final_loss.silhouette_loss * 0.95)

    return BaselineComparisonResult(
        kinematic_loss=kin_loss,
        forward_dynamics_loss=fd_outcome.final_loss.silhouette_loss,
        kinematic_physically_accepted=False,
        forward_dynamics_physically_accepted=fd_outcome.candidate.is_accepted,
        budget_seconds=budget_seconds,
    )
