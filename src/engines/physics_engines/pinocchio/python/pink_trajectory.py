"""Pink full-body trajectory inverse kinematics service (Packet P2, #10277).

Implements the ConstrainedIKBackend protocol using engine-local Pink tasks,
decoupled physical vs projection time semantics, structured failure reporting,
deterministic dropout handling, and independent rate and constraint audits.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.pink_tasks import (
    ConfigurationState,
    FrameResiduals,
    FrameTaskOptions,
    FrameTaskRequest,
    FullBodyPinkTasks,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.constrained_ik import (
    ConstrainedIKBackend,
    FrameRateAudit,
    IKOptions,
    IKTrajectoryRequest,
    IKTrajectoryResult,
)
from src.shared.python.motion_matching.full_body_ik import parse_specification

logger = get_logger(__name__)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

# Optional heavy dependencies: Pinocchio and Pink
pin: Any = None
try:
    import pinocchio as _pin_module

    pin = _pin_module
    PINOCCHIO_AVAILABLE = True
except (ImportError, OSError):
    PINOCCHIO_AVAILABLE = False

try:
    import pink

    PINK_AVAILABLE = True
except (ImportError, OSError):
    PINK_AVAILABLE = False
    pink = None  # type: ignore[assignment]


class PinkTrajectoryService(ConstrainedIKBackend):
    """Engine-local Pink implementation of the ConstrainedIKBackend protocol."""

    def __init__(
        self,
        specification: Mapping[str, Any] | bytes | str,
        model: Any | None = None,
    ) -> None:
        self.specification: dict[str, Any] = parse_specification(specification)
        self.tasks_builder = FullBodyPinkTasks(self.specification, model=model)
        self.coordinate_order: tuple[str, ...] = self.tasks_builder.coordinate_order
        self.nq = len(self.coordinate_order)
        self.nv = self.nq

        self._init_limits()
        self._init_pinocchio_model()
        self._configuration_cache: Any = None

    def _init_limits(self) -> None:
        """Extract velocity and configuration limits from specification."""
        vel_spec = self.specification.get("velocity_limits", {})
        if vel_spec and isinstance(vel_spec, dict):
            self.velocity_limits: Array | None = np.array(
                [float(vel_spec.get(c, 10.0)) for c in self.coordinate_order],
                dtype=np.float64,
            )
        else:
            self.velocity_limits = None

    def _init_pinocchio_model(self) -> None:
        """Cache references to native Pinocchio model and data if available."""
        self._plant = getattr(self.tasks_builder, "_plant", None)
        self._pin_model = getattr(self.tasks_builder, "pin_model", None)
        self._pin_data = getattr(self.tasks_builder, "pin_data", None)
        if self._pin_model is not None:
            if self._pin_data is None and hasattr(self._pin_model, "createData"):
                self._pin_data = self._pin_model.createData()
            if hasattr(self._pin_model, "nv"):
                self.nv = int(self._pin_model.nv)
                self.nq = int(self._pin_model.nq)

    @property
    def backend_name(self) -> str:
        return "pink_pinocchio"

    def solve_trajectory(
        self, request: IKTrajectoryRequest, options: IKOptions
    ) -> IKTrajectoryResult:
        """Solve constrained IK across all frames in request."""
        if request.initial_q.size != self.nq:
            raise ValueError(
                f"initial_q size {request.initial_q.size} must match model nq {self.nq}"
            )

        num_frames = request.num_frames
        configs = np.zeros((num_frames, self.nq), dtype=np.float64)
        frame_success = np.ones(num_frames, dtype=bool)
        residuals_list: list[FrameResiduals] = []
        rate_audits_list: list[FrameRateAudit] = []
        timing_list = np.zeros(num_frames, dtype=np.float64)
        failure_reasons: list[str] = []

        q_curr = np.asarray(request.initial_q, dtype=np.float64).copy()
        first_failed: int | None = None
        cancelled = False
        t_start_total = time.perf_counter()

        for f in range(num_frames):
            if request.cancellation_token is not None and request.cancellation_token():
                cancelled = True
                frame_success[f:] = False
                if first_failed is None:
                    first_failed = f
                failure_reasons.append(f"Cancelled at frame {f} boundary")
                self._fill_unattempted_frames(
                    f,
                    num_frames,
                    q_curr,
                    configs,
                    residuals_list,
                    rate_audits_list,
                    request,
                )
                break

            t0 = time.perf_counter()
            dt_phys = self._get_physical_dt(request, f, options.dt_numerical)

            q_next, success, reason, residuals = self._solve_frame(
                q_curr, f, request, options, dt_phys
            )
            t_frame = (time.perf_counter() - t0) * 1000.0

            configs[f] = q_next
            frame_success[f] = success
            timing_list[f] = t_frame
            residuals_list.append(residuals)

            delta_q = self._compute_delta_q(q_curr, q_next)
            rate_audits_list.append(
                FrameRateAudit.compute(
                    frame_index=f,
                    dt_s=dt_phys,
                    delta_q=delta_q,
                    velocity_limits=self.velocity_limits,
                    joint_names=self.coordinate_order,
                )
            )

            if not success:
                if first_failed is None:
                    first_failed = f
                if reason:
                    failure_reasons.append(f"Frame {f}: {reason}")
            else:
                q_curr = q_next.copy()

        total_time_ms = (time.perf_counter() - t_start_total) * 1000.0
        rates_ok = all(len(a.exceeded_joints) == 0 for a in rate_audits_list)
        if not rates_ok:
            failure_reasons.append("Joint velocity limits exceeded")

        passed = bool(np.all(frame_success)) and not cancelled and rates_ok

        return IKTrajectoryResult(
            configurations=configs,
            frame_success=frame_success,
            frame_residuals=tuple(residuals_list),
            rate_audits=tuple(rate_audits_list),
            timing_ms=timing_list,
            total_time_ms=total_time_ms,
            backend_name=self.backend_name,
            passed=passed,
            first_failed_frame=first_failed,
            cancelled=cancelled,
            failure_reasons=tuple(failure_reasons),
            metadata={"solver": options.solver, "step_mode": options.step_mode},
        )

    def _get_physical_dt(
        self, request: IKTrajectoryRequest, f: int, default_dt: float
    ) -> float:
        """Compute the physical time interval for frame f."""
        if f > 0:
            return float(request.time_s[f] - request.time_s[f - 1])
        if request.num_frames > 1:
            return float(request.time_s[1] - request.time_s[0])
        return float(default_dt)

    def _compute_delta_q(self, q_prev: Array, q_curr: Array) -> Array:
        """Compute coordinate displacement handling manifold difference if available."""
        if PINOCCHIO_AVAILABLE and pin is not None and self._pin_model is not None:
            return np.asarray(
                pin.difference(self._pin_model, q_prev, q_curr), dtype=np.float64
            )
        return np.asarray(q_curr - q_prev, dtype=np.float64)

    def _fill_unattempted_frames(
        self,
        start_f: int,
        num_frames: int,
        q_curr: Array,
        configs: Array,
        residuals_list: list[FrameResiduals],
        rate_audits_list: list[FrameRateAudit],
        request: IKTrajectoryRequest,
    ) -> None:
        """Fill remainder of trajectory buffers after cancellation."""
        for f in range(start_f, num_frames):
            configs[f] = q_curr
            residuals_list.append(
                FrameResiduals(
                    marker_errors_m={},
                    weld_translation_error_m=0.0,
                    weld_rotation_error_rad=0.0,
                    bound_violations={},
                    stance_errors_m={},
                )
            )
            dt_phys = self._get_physical_dt(request, f, 1.0 / 360.0)
            rate_audits_list.append(
                FrameRateAudit.compute(
                    frame_index=f,
                    dt_s=dt_phys,
                    delta_q=np.zeros(self.nv, dtype=np.float64),
                    velocity_limits=self.velocity_limits,
                    joint_names=self.coordinate_order,
                )
            )

    def _solve_frame(
        self,
        q_curr: Array,
        frame_idx: int,
        request: IKTrajectoryRequest,
        options: IKOptions,
        dt_phys: float,
    ) -> tuple[Array, bool, str | None, FrameResiduals]:
        """Solve a single frame respecting physical vs projection mode."""
        marker_targets = {
            lbl: request.marker_targets[frame_idx, i]
            for i, lbl in enumerate(request.labels)
        }
        validity_mask = {
            lbl: True if request.validity_mask[frame_idx, i] else False
            for i, lbl in enumerate(request.labels)
        }

        dt_step = dt_phys if options.step_mode == "physical" else options.dt_numerical
        task_opts = FrameTaskOptions(
            solver=options.solver,
            damping=options.damping,
            dt=dt_step,
        )
        task_req = FrameTaskRequest(
            marker_targets=marker_targets,
            validity_mask=validity_mask,
            posture_target=(
                request.posture_target if request.posture_target is not None else q_curr
            ),
            policy=request.policy,
            options=task_opts,
        )

        try:
            bundle = self.tasks_builder.build(task_req)
        except ValueError as exc:
            residuals = self._audit_configuration(q_curr, task_req)
            return q_curr, False, str(exc), residuals
        max_iters = 1 if options.step_mode == "physical" else options.max_iterations

        q_iter = q_curr.copy()
        last_reason: str | None = None
        success = True

        for _ in range(max_iters):
            q_iter, step_ok, step_err = self._execute_qp_step(
                q_iter, bundle, dt_step, options
            )
            if not step_ok:
                success = False
                last_reason = step_err
                break

        residuals = self._audit_configuration(q_iter, task_req)
        # Check tolerance on marker, weld residuals, and bounds
        if success:
            if (
                not np.isfinite(residuals.weld_translation_error_m)
                or residuals.weld_translation_error_m
                > options.weld_translation_tolerance_m
            ):
                success = False
                last_reason = (
                    f"Weld translation error {residuals.weld_translation_error_m:.4e} "
                    f"exceeds tolerance ({options.weld_translation_tolerance_m:.4e}) or unevaluated"
                )
            elif (
                not np.isfinite(residuals.weld_rotation_error_rad)
                or residuals.weld_rotation_error_rad
                > options.weld_rotation_tolerance_rad
            ):
                success = False
                last_reason = (
                    f"Weld rotation error {residuals.weld_rotation_error_rad:.4e} "
                    f"exceeds tolerance ({options.weld_rotation_tolerance_rad:.4e}) or unevaluated"
                )
            elif any(
                not np.isfinite(err) or err > options.marker_tolerance_m
                for err in residuals.marker_errors_m.values()
            ):
                success = False
                last_reason = (
                    "One or more marker residuals exceed marker tolerance "
                    f"({options.marker_tolerance_m:.4e}) or unevaluated"
                )
            elif len(residuals.bound_violations) > 0:
                success = False
                last_reason = f"Coordinate bound violations: {list(residuals.bound_violations.keys())}"

        return q_iter, success, last_reason, residuals

    def _execute_qp_step(
        self,
        q: Array,
        bundle: Any,
        dt: float,
        options: IKOptions,
    ) -> tuple[Array, bool, str | None]:
        """Execute one differential step in Pink with strict fail-closed semantics."""
        if not (
            PINK_AVAILABLE
            and PINOCCHIO_AVAILABLE
            and self._pin_model is not None
            and self._pin_data is not None
        ):
            return (
                q.copy(),
                False,
                "Native Pink/Pinocchio stack or plant model is unavailable",
            )

        try:
            config = pink.Configuration(self._pin_model, self._pin_data, q)
            solve_kwargs: dict[str, Any] = {
                "solver": options.solver,
                "damping": options.damping,
            }
            if bundle.constraints:
                solve_kwargs["constraints"] = bundle.constraints
            if bundle.limits and options.limit_policy == "enforce":
                solve_kwargs["limits"] = bundle.limits

            all_tasks = list(bundle.tasks)
            if bundle.posture_task is not None:
                all_tasks.append(bundle.posture_task)

            velocity = pink.solve_ik(config, all_tasks, dt, **solve_kwargs)
            velocity_arr = np.asarray(velocity, dtype=np.float64)
            if not np.isfinite(velocity_arr).all():
                return q, False, "Non-finite velocity returned by QP"

            q_next = np.asarray(
                pin.integrate(self._pin_model, q, velocity_arr * dt), dtype=np.float64
            )
            if not np.isfinite(q_next).all():
                return q, False, "Non-finite coordinates after integration"
            return q_next, True, None
        except Exception as exc:
            return q, False, str(exc)

    def _audit_configuration(
        self, q: Array, task_req: FrameTaskRequest
    ) -> FrameResiduals:
        """Compute residuals for configuration q against task request."""
        config_state = ConfigurationState(q=q)
        return self.tasks_builder.audit(config_state, request=task_req)

    @staticmethod
    def _build_frame_task_request(
        request: IKTrajectoryRequest, f: int
    ) -> FrameTaskRequest:
        marker_targets = {
            lbl: request.marker_targets[f, i] for i, lbl in enumerate(request.labels)
        }
        validity_mask = {
            lbl: True if request.validity_mask[f, i] else False
            for i, lbl in enumerate(request.labels)
        }
        return FrameTaskRequest(
            marker_targets=marker_targets,
            validity_mask=validity_mask,
            posture_target=request.posture_target,
            policy=request.policy,
        )

    @staticmethod
    def _is_frame_audit_successful(
        residuals: FrameResiduals, audit: FrameRateAudit, opts: IKOptions
    ) -> bool:
        has_nan_marker = any(
            not np.isfinite(err) or err > opts.marker_tolerance_m
            for err in residuals.marker_errors_m.values()
        )
        weld_trans_fail = (
            not np.isfinite(residuals.weld_translation_error_m)
            or residuals.weld_translation_error_m > opts.weld_translation_tolerance_m
        )
        weld_rot_fail = (
            not np.isfinite(residuals.weld_rotation_error_rad)
            or residuals.weld_rotation_error_rad > opts.weld_rotation_tolerance_rad
        )
        bounds_fail = len(residuals.bound_violations) > 0
        rates_fail = len(audit.exceeded_joints) > 0

        return not (
            has_nan_marker
            or weld_trans_fail
            or weld_rot_fail
            or bounds_fail
            or rates_fail
        )

    def audit_trajectory(
        self,
        trajectory: Array,
        request: IKTrajectoryRequest,
        options: IKOptions | None = None,
    ) -> IKTrajectoryResult:
        """Independently audit an existing trajectory against constraints and limits."""
        opts = options or IKOptions()
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.shape[0] != request.num_frames or traj.shape[1] != self.nq:
            raise ValueError(
                f"Trajectory shape {traj.shape} must match (frames={request.num_frames}, nq={self.nq})"
            )

        num_frames = request.num_frames
        frame_success = np.ones(num_frames, dtype=bool)
        residuals_list: list[FrameResiduals] = []
        rate_audits_list: list[FrameRateAudit] = []
        timing_list = np.zeros(num_frames, dtype=np.float64)
        failure_reasons: list[str] = []

        for f in range(num_frames):
            t0 = time.perf_counter()
            dt_phys = self._get_physical_dt(request, f, 1.0 / 360.0)
            task_req = self._build_frame_task_request(request, f)

            residuals = self._audit_configuration(traj[f], task_req)
            residuals_list.append(residuals)

            delta_q = self._compute_delta_q(traj[f - 1] if f > 0 else traj[0], traj[f])
            audit = FrameRateAudit.compute(
                frame_index=f,
                dt_s=dt_phys,
                delta_q=delta_q,
                velocity_limits=self.velocity_limits,
                joint_names=self.coordinate_order,
            )
            rate_audits_list.append(audit)

            frame_success[f] = self._is_frame_audit_successful(residuals, audit, opts)
            timing_list[f] = (time.perf_counter() - t0) * 1000.0

        passed = bool(np.all(frame_success))
        first_failed = int(np.where(~frame_success)[0][0]) if not passed else None
        if not passed:
            failure_reasons.append(
                "Audit failed: velocity, residual, or bound constraints violated or unevaluated"
            )

        return IKTrajectoryResult(
            configurations=traj,
            frame_success=frame_success,
            frame_residuals=tuple(residuals_list),
            rate_audits=tuple(rate_audits_list),
            timing_ms=timing_list,
            total_time_ms=float(np.sum(timing_list)),
            backend_name=self.backend_name,
            passed=passed,
            first_failed_frame=first_failed,
            cancelled=False,
            failure_reasons=tuple(failure_reasons),
            metadata={"audit_only": True},
        )
