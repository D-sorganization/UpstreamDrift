"""Pink full-body task and constraint translation module (Packet P1, #10276).

Translates canonical full-body marker attachments, dual-grip weld loop closure,
coordinate limits, and stance constraints into engine-local Pink tasks, hard
equality constraints, and bounds. Strictly enforces DbC, LoD, and DRY principles.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

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
    import pink.limits
    import pink.tasks
    from pink.limits import ConfigurationLimit
    from pink.tasks import FrameTask, PostureTask, RelativeFrameTask

    PINK_AVAILABLE = True
except (ImportError, OSError):
    PINK_AVAILABLE = False
    pink = None  # type: ignore[assignment]
    ConfigurationLimit = None  # type: ignore[assignment]
    FrameTask = None  # type: ignore[assignment]
    PostureTask = None  # type: ignore[assignment]
    RelativeFrameTask = None  # type: ignore[assignment]

_WELD_LOG_BRANCH_MARGIN_RAD = 1e-7


def _require_weld_log_chart(
    pose_error: NDArray[np.float64], margin_rad: float = _WELD_LOG_BRANCH_MARGIN_RAD
) -> None:
    """Validate that rotation residual stays safely away from the log branch cut at pi."""
    rot_norm = float(np.linalg.norm(pose_error[3:]))
    if rot_norm >= math.pi - margin_rad:
        raise ValueError("Weld log derivative is undefined near the rotation-pi branch")


def _has_frame(model: Any, name: str) -> bool:
    """Check whether a model contains a named frame."""
    if hasattr(model, "existFrame"):
        return bool(model.existFrame(name))
    if hasattr(model, "hasFrame"):
        return bool(model.hasFrame(name))
    return False


def _is_pin_model(obj: Any) -> bool:
    """Safely check whether an object is a Pinocchio Model instance."""
    if not PINOCCHIO_AVAILABLE or pin is None:
        return False
    pin_model_cls = getattr(pin, "Model", None)
    if isinstance(pin_model_cls, type):
        return isinstance(obj, pin_model_cls)
    return False


@dataclass(frozen=True)
class FrameTaskOptions:
    """Options controlling task gain, damping, and solver behavior."""

    gain: float = 1.0
    damping: float = 1e-6
    solver: str = "quadprog"
    dt: float = 1.0 / 360.0


@dataclass(frozen=True)
class StanceClosurePolicy:
    """Declared policy for kinematic weld closure, locked coordinates, and stance."""

    enforce_weld: bool = True
    locked_coordinates: Mapping[str, float] | None = None
    stance_spheres: Sequence[str] | None = None
    weld_position_cost: float = 1.0
    weld_orientation_cost: float = 1.0
    marker_cost: float = 1.0
    posture_cost: float = 1e-3


@dataclass(frozen=True)
class FrameTaskRequest:
    """Immutable request specifying marker targets, validity mask, and policies."""

    marker_targets: Mapping[str, NDArray[np.float64]]
    validity_mask: Mapping[str, bool]
    posture_target: Mapping[str, float] | NDArray[np.float64] | None = None
    policy: StanceClosurePolicy | None = None
    options: FrameTaskOptions | None = None

    def __post_init__(self) -> None:
        if set(self.marker_targets.keys()) != set(self.validity_mask.keys()):
            raise ValueError(
                "Inventory mismatch between marker_targets and validity_mask"
            )


@dataclass(frozen=True)
class ConfigurationState:
    """Kinematic state of coordinates and diagnostic residuals for auditing."""

    q: NDArray[np.float64]
    marker_positions: Mapping[str, NDArray[np.float64]] | None = None
    weld_pose_error: NDArray[np.float64] | None = None
    stance_contacts: Mapping[str, float] | None = None


@dataclass(frozen=True)
class FrameResiduals:
    """Audited kinematic residuals across all task and constraint categories."""

    marker_errors_m: dict[str, float]
    weld_translation_error_m: float
    weld_rotation_error_rad: float
    bound_violations: dict[str, float]
    stance_errors_m: dict[str, float]


@dataclass(frozen=True)
class FrameTaskBundle:
    """Separated soft tasks, hard equality constraints, and configuration limits."""

    tasks: tuple[Any, ...]
    constraints: tuple[Any, ...]
    limits: tuple[Any, ...]
    posture_task: Any | None = None


class LockedCoordinateTask:
    """Hard equality constraint enforcing q[i] == target in the Pink QP."""

    def __init__(
        self,
        coordinate_name: str,
        q_index: int,
        v_index: int,
        target_value: float,
        nv: int,
        gain: float = 1.0,
    ) -> None:
        self.coordinate_name = coordinate_name
        self.q_index = q_index
        self.v_index = v_index
        self.target_value = float(target_value)
        self.nv = nv
        self.gain = float(gain)
        self.cost = 1.0

    def compute_error(self, configuration: Any) -> NDArray[np.float64]:
        q_val = float(configuration.q[self.q_index])
        return np.array([q_val - self.target_value], dtype=np.float64)

    def compute_jacobian(self, configuration: Any) -> NDArray[np.float64]:
        jacobian = np.zeros((1, self.nv), dtype=np.float64)
        jacobian[0, self.v_index] = 1.0
        return jacobian


class MockFrameTask:
    """Engine-agnostic frame task used when native Pink is absent."""

    def __init__(
        self, frame: str, target: NDArray[np.float64], cost: float = 1.0
    ) -> None:
        self.frame = frame
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.cost = float(cost)
        self.gain = 1.0

    def compute_error(self, configuration: Any) -> NDArray[np.float64]:
        return np.zeros(3, dtype=np.float64)

    def compute_jacobian(self, configuration: Any) -> NDArray[np.float64]:
        nv = getattr(getattr(configuration, "model", None), "nv", 41)
        return np.zeros((3, nv), dtype=np.float64)


class MockRelativeFrameTask:
    """Engine-agnostic relative frame task used when native Pink is absent."""

    def __init__(self, frame: str, root: str, is_weld_closure: bool = True) -> None:
        self.frame = frame
        self.root = root
        self.is_weld_closure = is_weld_closure
        self.cost = 1.0
        self.gain = 1.0

    def compute_error(self, configuration: Any) -> NDArray[np.float64]:
        return np.zeros(6, dtype=np.float64)

    def compute_jacobian(self, configuration: Any) -> NDArray[np.float64]:
        nv = getattr(getattr(configuration, "model", None), "nv", 41)
        return np.zeros((6, nv), dtype=np.float64)


class MockPostureTask:
    """Engine-agnostic posture task used when native Pink is absent."""

    def __init__(self, target_q: NDArray[np.float64], cost: float = 1e-3) -> None:
        self.target_q = np.asarray(target_q, dtype=np.float64).copy()
        self.cost = float(cost)
        self.gain = 1.0
        self.frame = None

    def compute_error(self, configuration: Any) -> NDArray[np.float64]:
        return np.asarray(configuration.q, dtype=np.float64) - self.target_q

    def compute_jacobian(self, configuration: Any) -> NDArray[np.float64]:
        nv = getattr(getattr(configuration, "model", None), "nv", len(self.target_q))
        return np.eye(nv, dtype=np.float64)


class FullBodyConfigurationLimit:
    """Configuration limit with explicit radian bounds derived from specification."""

    def __init__(
        self,
        model: Any,
        lower_limit: NDArray[np.float64],
        upper_limit: NDArray[np.float64],
        config_limit_gain: float = 0.5,
    ) -> None:
        self.model = model
        self.lower_limit = np.asarray(lower_limit, dtype=np.float64).copy()
        self.upper_limit = np.asarray(upper_limit, dtype=np.float64).copy()
        self.config_limit_gain = float(config_limit_gain)
        if PINK_AVAILABLE and ConfigurationLimit is not None and _is_pin_model(model):
            self._native_limit = ConfigurationLimit(
                model, config_limit_gain=config_limit_gain
            )
        else:
            self._native_limit = None

    def compute_qp_inequalities(self, configuration: Any, dt: float) -> tuple[Any, Any]:
        if self._native_limit is not None:
            return self._native_limit.compute_qp_inequalities(configuration, dt)
        nv = getattr(self.model, "nv", len(self.lower_limit))
        g = np.vstack([np.eye(nv), -np.eye(nv)])
        h = np.zeros(2 * nv, dtype=np.float64)
        return g, h


class FullBodyPinkTasks:
    """Typed facade translating canonical full-body motion requests to Pink tasks."""

    def __init__(
        self,
        specification: Mapping[str, Any] | bytes | str,
        model: Any | None = None,
    ) -> None:
        if isinstance(specification, (bytes, str)):
            self.specification: dict[str, Any] = json.loads(specification)
        else:
            self.specification = dict(specification)

        schema = self.specification.get("schema_version")
        if schema != "full-body-v1":
            raise ValueError(f"Unsupported specification schema: {schema}")

        self.coordinate_order: tuple[str, ...] = tuple(
            self.specification["coordinate_order"]
        )
        self._model = model
        self._plant: Any | None = None
        self.pin_model: Any | None = None
        self.pin_data: Any | None = None
        if model is not None and hasattr(model, "model") and hasattr(model, "_pin"):
            self._plant = model
            self.pin_model = model.model
            self.pin_data = getattr(model, "data", None)
        elif _is_pin_model(model) and model is not None:
            self.pin_model = model
            if hasattr(model, "createData"):
                self.pin_data = model.createData()
        elif PINOCCHIO_AVAILABLE and pin is not None and model is None:
            try:
                from src.engines.physics_engines.pinocchio.python.native_model import (
                    FullBodyPinocchioModel,
                )

                plant = FullBodyPinocchioModel(self.specification)
                self._plant = plant
                self.pin_model = plant.model
                self.pin_data = plant.data
            except Exception as exc:
                logger.warning(
                    "Could not construct canonical FullBodyPinocchioModel: %s", exc
                )
                self._plant = None
                self.pin_model = None
                self.pin_data = None

        self._init_dimensions_and_coordinates(model)
        self._init_bounds(model)
        self._init_spec_structures()
        self._init_closure_and_marker_frames()

    @property
    def _pin_model(self) -> Any | None:
        """Compatibility accessor for native Pinocchio model."""
        return self.pin_model

    @property
    def _pin_data(self) -> Any | None:
        """Compatibility accessor for native Pinocchio data."""
        return self.pin_data

    def _init_dimensions_and_coordinates(self, model: Any | None) -> None:
        """Initialize configuration/tangent dimensions and coordinate index mappings."""
        if self.pin_model is not None:
            self.nq: int = int(
                getattr(self.pin_model, "nq", len(self.coordinate_order))
            )
            self.nv: int = int(
                getattr(self.pin_model, "nv", len(self.coordinate_order))
            )
        elif model is not None:
            self.nq = int(getattr(model, "nq", len(self.coordinate_order)))
            self.nv = int(getattr(model, "nv", len(self.coordinate_order)))
        else:
            self.nq = len(self.coordinate_order)
            self.nv = len(self.coordinate_order)

        self._coordinates: dict[str, int] = {}
        self._velocity_indices: dict[str, int] = {}
        if model is not None and hasattr(model, "_coordinates"):
            self._coordinates = dict(model._coordinates)
            self._velocity_indices = dict(model._velocity_indices)
        else:
            for i, name in enumerate(self.coordinate_order):
                self._coordinates[name] = i
                self._velocity_indices[name] = i

    def _init_bounds(self, model: Any | None) -> None:
        """Initialize coordinate position limits in radians."""
        self.lower_limit = np.full(self.nq, -math.pi, dtype=np.float64)
        self.upper_limit = np.full(self.nq, math.pi, dtype=np.float64)
        if model is not None and hasattr(model, "lowerPositionLimit"):
            self.lower_limit[: len(model.lowerPositionLimit)] = model.lowerPositionLimit
            self.upper_limit[: len(model.upperPositionLimit)] = model.upperPositionLimit
        elif self.pin_model is not None:
            self.lower_limit[: len(self.pin_model.lowerPositionLimit)] = (
                self.pin_model.lowerPositionLimit
            )
            self.upper_limit[: len(self.pin_model.upperPositionLimit)] = (
                self.pin_model.upperPositionLimit
            )

        ranges_deg = self.specification.get("coordinate_ranges_deg", {})
        for name, (deg_min, deg_max) in ranges_deg.items():
            if name in self._coordinates:
                idx = self._coordinates[name]
                self.lower_limit[idx] = math.radians(float(deg_min))
                self.upper_limit[idx] = math.radians(float(deg_max))

        if self.pin_model is not None:
            if hasattr(self.pin_model, "lowerPositionLimit"):
                self.pin_model.lowerPositionLimit[: len(self.lower_limit)] = (
                    self.lower_limit
                )
            if hasattr(self.pin_model, "upperPositionLimit"):
                self.pin_model.upperPositionLimit[: len(self.upper_limit)] = (
                    self.upper_limit
                )

    def _init_spec_structures(self) -> None:
        """Initialize specification frame, marker, and closure metadata."""
        self.marker_attachments: dict[str, Any] = self.specification.get(
            "marker_attachments", {}
        )
        self._frames: dict[str, int] = {}
        for f in self.specification.get("frames", []):
            self._frames[f["name"]] = len(self._frames) + 1

        closure = self.specification.get("closure", {})
        self._closure_body_a: str = closure.get("body_a", "")
        self._closure_body_b: str = closure.get("body_b", "")
        self._closure_frame_a_name: str = closure.get("frame_a_name", "closure_frame_a")
        self._closure_frame_b_name: str = closure.get("frame_b_name", "closure_frame_b")

        contact_spec = self.specification.get("contact", {})
        self.contact_spheres: tuple[str, ...] = tuple(
            s["name"] for s in contact_spec.get("spheres", [])
        )

    def _init_closure_and_marker_frames(self) -> None:
        """Add closure and marker attachment operational frames to native pin_model."""
        if not (
            PINOCCHIO_AVAILABLE
            and self._plant is not None
            and self.pin_model is not None
        ):
            return

        if hasattr(self._plant, "constraints") and len(self._plant.constraints) > 0:
            c_model = self._plant.constraints[0]
            if not _has_frame(self.pin_model, self._closure_frame_a_name):
                self.pin_model.addFrame(
                    pin.Frame(
                        self._closure_frame_a_name,
                        c_model.joint1_id,
                        c_model.joint1_placement,
                        pin.FrameType.OP_FRAME,
                    )
                )
            if not _has_frame(self.pin_model, self._closure_frame_b_name):
                self.pin_model.addFrame(
                    pin.Frame(
                        self._closure_frame_b_name,
                        c_model.joint2_id,
                        c_model.joint2_placement,
                        pin.FrameType.OP_FRAME,
                    )
                )

        frames_added = False
        for m_name, m_att in self.marker_attachments.items():
            if not _has_frame(self.pin_model, m_name):
                body = m_att.get("body")
                joint = None
                body_pose = None
                if body in self._plant._bodies:
                    joint, body_pose = self._plant._bodies[body]
                elif body in self._plant._frames:
                    fid = self._plant._frames[body]
                    parent_f = self.pin_model.frames[fid]
                    joint = parent_f.parentJoint
                    body_pose = parent_f.placement

                if joint is not None and body_pose is not None:
                    offset = m_att.get("offset_m")
                    if offset is not None:
                        offset_placement = pin.SE3(
                            np.eye(3), np.asarray(offset, dtype=float)
                        )
                    else:
                        offset_placement = pin.SE3.Identity()
                    placement = body_pose * offset_placement
                    self.pin_model.addFrame(
                        pin.Frame(m_name, joint, placement, pin.FrameType.OP_FRAME)
                    )
                    frames_added = True

        if frames_added:
            self._plant.data = self.pin_model.createData()

    def _get_model_frame_name(self, marker_name: str) -> str:
        """Resolve marker name to an operational frame name."""
        if marker_name in self.marker_attachments:
            return marker_name
        if marker_name in self._frames:
            return marker_name
        if self.pin_model is not None and _has_frame(self.pin_model, marker_name):
            return marker_name
        if self._model is not None and _has_frame(self._model, marker_name):
            return marker_name
        raise ValueError(f"Unknown marker label: {marker_name}")

    def _build_marker_tasks(
        self,
        observed_markers: list[str],
        request: FrameTaskRequest,
        policy: StanceClosurePolicy,
        options: FrameTaskOptions,
    ) -> list[Any]:
        """Construct soft frame tracking tasks for observed markers."""
        tasks: list[Any] = []
        for name in observed_markers:
            target_pos = np.asarray(request.marker_targets[name], dtype=np.float64)
            if target_pos.shape != (3,) or not np.all(np.isfinite(target_pos)):
                raise ValueError(f"Target coordinates must be finite for marker {name}")
            frame_name = self._get_model_frame_name(name)
            if (
                PINK_AVAILABLE
                and FrameTask is not None
                and _is_pin_model(self.pin_model)
            ):
                task = FrameTask(
                    frame_name,
                    position_cost=policy.marker_cost,
                    orientation_cost=0.0,
                    gain=options.gain,
                )
                target_se3 = pin.SE3(np.eye(3), target_pos)
                task.set_target(target_se3)
                tasks.append(task)
            else:
                tasks.append(
                    MockFrameTask(frame_name, target_pos, cost=policy.marker_cost)
                )
        return tasks

    def _build_posture_task(
        self,
        request: FrameTaskRequest,
        policy: StanceClosurePolicy,
        options: FrameTaskOptions,
    ) -> Any | None:
        """Construct posture regularization task if requested."""
        if request.posture_target is None:
            return None

        target_q: NDArray[np.float64]
        if isinstance(request.posture_target, Mapping):
            target_q = np.zeros(len(self.coordinate_order), dtype=np.float64)
            for name, val in request.posture_target.items():
                if name in self._coordinates:
                    target_q[self._coordinates[name]] = float(val)
        else:
            target_q = np.asarray(request.posture_target, dtype=np.float64)

        if PINK_AVAILABLE and PostureTask is not None and _is_pin_model(self.pin_model):
            posture_task = PostureTask(cost=policy.posture_cost, gain=options.gain)
            posture_task.set_target(target_q)
            posture_task.target_q = target_q
            posture_task.frame = None
            return posture_task

        return MockPostureTask(target_q, cost=policy.posture_cost)

    def _build_constraints(
        self,
        policy: StanceClosurePolicy,
        options: FrameTaskOptions,
    ) -> list[Any]:
        """Construct hard loop closure and locked coordinate constraints."""
        constraints: list[Any] = []
        if policy.enforce_weld:
            if (
                PINK_AVAILABLE
                and RelativeFrameTask is not None
                and _is_pin_model(self.pin_model)
            ):
                weld_task = RelativeFrameTask(
                    frame=self._closure_frame_a_name,
                    root=self._closure_frame_b_name,
                    position_cost=policy.weld_position_cost,
                    orientation_cost=policy.weld_orientation_cost,
                    gain=options.gain,
                )
                weld_task.set_target(pin.SE3.Identity())
                weld_task.is_weld_closure = True
                constraints.append(weld_task)
            else:
                constraints.append(
                    MockRelativeFrameTask(
                        frame=self._closure_frame_a_name,
                        root=self._closure_frame_b_name,
                        is_weld_closure=True,
                    )
                )

        if policy.locked_coordinates:
            for coord_name, target_val in policy.locked_coordinates.items():
                if coord_name in self._coordinates:
                    q_idx = self._coordinates[coord_name]
                    v_idx = self._velocity_indices[coord_name]
                    constraints.append(
                        LockedCoordinateTask(
                            coordinate_name=coord_name,
                            q_index=q_idx,
                            v_index=v_idx,
                            target_value=float(target_val),
                            nv=self.nv,
                            gain=options.gain,
                        )
                    )
        return constraints

    def build(self, request: FrameTaskRequest) -> FrameTaskBundle:
        """Translate a frame task request into soft tasks, equalities, and limits."""
        policy = request.policy or StanceClosurePolicy()
        options = request.options or FrameTaskOptions()

        # Validate marker names before any native calls
        for marker_name in request.marker_targets:
            self._get_model_frame_name(marker_name)

        observed_markers = [
            name
            for name, valid in request.validity_mask.items()
            if valid and name in request.marker_targets
        ]
        if not observed_markers:
            raise ValueError("Insufficient data: zero observed marker targets")

        soft_tasks = self._build_marker_tasks(
            observed_markers, request, policy, options
        )
        posture_task_obj = self._build_posture_task(request, policy, options)
        if posture_task_obj is not None:
            soft_tasks.append(posture_task_obj)

        constraints = self._build_constraints(policy, options)

        target_limit_model = (
            self.pin_model if self.pin_model is not None else self._model
        )
        limits = (
            FullBodyConfigurationLimit(
                model=target_limit_model,
                lower_limit=self.lower_limit,
                upper_limit=self.upper_limit,
            ),
        )

        return FrameTaskBundle(
            tasks=tuple(soft_tasks),
            constraints=tuple(constraints),
            limits=limits,
            posture_task=posture_task_obj,
        )

    def _resolve_marker_positions(
        self,
        q: np.ndarray,
        configuration: ConfigurationState,
        request: FrameTaskRequest | None,
    ) -> Mapping[str, np.ndarray] | None:
        marker_positions = configuration.marker_positions
        if (
            marker_positions is None
            and self.pin_model is not None
            and PINOCCHIO_AVAILABLE
            and request is not None
        ):
            try:
                data = getattr(self._plant, "data", None) or (
                    self.pin_data
                    if self.pin_data is not None
                    else self.pin_model.createData()
                )
                pin.forwardKinematics(self.pin_model, data, q)
                pin.updateFramePlacements(self.pin_model, data)
                marker_positions = {}
            except Exception:
                marker_positions = None

        if marker_positions is None and request is not None:
            model_target = self._plant if self._plant is not None else self._model
            if model_target is not None and hasattr(model_target, "frame_poses"):
                try:
                    coords = {
                        name: float(q[self._coordinates[name]])
                        for name in self.coordinate_order
                    }
                    poses = model_target.frame_poses(coords)
                    marker_positions = {
                        k: (
                            np.asarray(v)[:3, 3]
                            if np.asarray(v).shape == (4, 4)
                            else np.asarray(v)[:3]
                        )
                        for k, v in poses.items()
                    }
                except Exception:
                    marker_positions = None
        return marker_positions

    def _compute_weld_residuals(
        self,
        q: np.ndarray,
        configuration: ConfigurationState,
        request: FrameTaskRequest | None,
    ) -> tuple[float, float]:
        enforce_weld = True
        if request is not None and request.policy is not None:
            enforce_weld = request.policy.enforce_weld

        if not enforce_weld:
            return 0.0, 0.0

        weld_error = configuration.weld_pose_error
        if weld_error is None and enforce_weld:
            model_target = self._plant if self._plant is not None else self._model
            if model_target is not None and hasattr(model_target, "closure_residuals"):
                try:
                    coords = {
                        name: float(q[self._coordinates[name]])
                        for name in self.coordinate_order
                    }
                    weld_error, _ = model_target.closure_residuals(coords)
                except Exception:
                    weld_error = None

        if weld_error is not None:
            _require_weld_log_chart(weld_error)
            return float(np.linalg.norm(weld_error[:3])), float(
                np.linalg.norm(weld_error[3:])
            )
        return float("nan"), float("nan")

    def audit(
        self,
        configuration: ConfigurationState,
        request: FrameTaskRequest | None = None,
    ) -> FrameResiduals:
        """Audit post-solve residuals, bound violations, and weld closure errors."""
        q = np.asarray(configuration.q, dtype=np.float64)
        marker_positions = self._resolve_marker_positions(q, configuration, request)

        # Marker residuals: unevaluated or missing frame must yield NaN, never zero
        marker_errors: dict[str, float] = {}
        if request is not None:
            if marker_positions is not None:
                for name, valid in request.validity_mask.items():
                    if valid and name in request.marker_targets:
                        if name in marker_positions:
                            pred = np.asarray(marker_positions[name], dtype=np.float64)
                            target = np.asarray(
                                request.marker_targets[name], dtype=np.float64
                            )
                            marker_errors[name] = float(np.linalg.norm(pred - target))
                        else:
                            marker_errors[name] = float("nan")
            else:
                for name, valid in request.validity_mask.items():
                    if valid and name in request.marker_targets:
                        marker_errors[name] = float("nan")

        # Weld closure residuals: unevaluated must yield NaN, never zero
        weld_translation_error, weld_rotation_error = self._compute_weld_residuals(
            q, configuration, request
        )

        # Bound violations
        bound_violations: dict[str, float] = {}
        for name in self.coordinate_order:
            idx = self._coordinates[name]
            val = float(q[idx])
            low = float(self.lower_limit[idx])
            high = float(self.upper_limit[idx])
            if val < low:
                bound_violations[name] = low - val
            elif val > high:
                bound_violations[name] = val - high

        # Stance contact errors
        stance_errors: dict[str, float] = {}
        if configuration.stance_contacts is not None:
            stance_errors = {
                k: float(v) for k, v in configuration.stance_contacts.items()
            }

        return FrameResiduals(
            marker_errors_m=marker_errors,
            weld_translation_error_m=weld_translation_error,
            weld_rotation_error_rad=weld_rotation_error,
            bound_violations=bound_violations,
            stance_errors_m=stance_errors,
        )
