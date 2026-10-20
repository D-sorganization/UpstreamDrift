from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Sequence

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.muscle_allocation import (
    MuscleAllocationResult,
    MuscleRecruitmentPath,
    allocate_muscle_excitations,
)
from src.shared.python.motion_matching.distributed_feedback import FeedbackRequest


class EffortBasis(str, Enum):
    TOTAL_MUSCULOTENDINOUS = "total_musculotendinous"
    ACTIVE_INCREMENT = "active_increment"


class NativePassiveComponent(str, Enum):
    """Allowlisted native passive-fiber readback, excluding active/total force."""

    PASSIVE_FIBER_FORCE_ALONG_TENDON = "getPassiveFiberForceAlongTendon"


@dataclass(frozen=True)
class NativeCoordinateMapping:
    coordinate_path: str
    coordinate_value_state_path: str
    coordinate_speed_state_path: str
    coordinate_index: int
    q_index: int
    u_index: int
    generalized_effort_index: int
    coordinate_count: int
    q_count: int
    u_count: int
    constraint_count: int
    locked: bool
    prescribed: bool
    motion_type: str
    effort_unit: str
    feedback_channel_id: str

    def __post_init__(self) -> None:
        indices = (
            self.coordinate_index,
            self.q_index,
            self.u_index,
            self.generalized_effort_index,
            self.coordinate_count,
            self.q_count,
            self.u_count,
            self.constraint_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in indices
        ):
            raise ValueError("native coordinate indices and counts must be integers")
        if not isinstance(self.locked, bool) or not isinstance(self.prescribed, bool):
            raise ValueError("native coordinate lock flags must be boolean")
        if (
            not self.coordinate_path
            or not self.coordinate_value_state_path
            or not self.coordinate_speed_state_path
            or not self.feedback_channel_id
        ):
            raise ValueError("coordinate and state paths must be source-bound")
        if (
            self.motion_type != "rotational"
            or self.effort_unit != "N*m"
            or self.locked
            or self.prescribed
            or self.constraint_count != 0
            or (self.coordinate_count, self.q_count, self.u_count) != (1, 1, 1)
            or (
                self.coordinate_index,
                self.q_index,
                self.u_index,
                self.generalized_effort_index,
            )
            != (0, 0, 0, 0)
        ):
            raise ValueError(
                "only one unlocked, unconstrained rotational coordinate with "
                "explicit zero-based q/u/effort mapping is supported"
            )

    @property
    def coordinate_name(self) -> str:
        return self.coordinate_path.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class AnalyticRecruitmentConfig:
    max_isometric_force_n: float
    optimal_fiber_length_m: float
    tendon_slack_length_m: float
    pennation_angle_rad: float
    activation_time_constant_s: float
    deactivation_time_constant_s: float
    minimum_activation: float
    excitation_bounds: tuple[float, float]
    maximum_excitation_rate_per_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "excitation_bounds", tuple(self.excitation_bounds))
        if len(self.excitation_bounds) != 2:
            raise ValueError("excitation bounds need exactly two values")
        values = (
            self.max_isometric_force_n,
            self.optimal_fiber_length_m,
            self.tendon_slack_length_m,
            self.pennation_angle_rad,
            self.activation_time_constant_s,
            self.deactivation_time_constant_s,
            self.minimum_activation,
            self.maximum_excitation_rate_per_s,
            *self.excitation_bounds,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("analytic recruitment parameters must be finite")
        low, high = self.excitation_bounds
        if (
            self.max_isometric_force_n <= 0
            or self.optimal_fiber_length_m <= 0
            or self.tendon_slack_length_m <= 0
            or self.activation_time_constant_s <= 0
            or self.deactivation_time_constant_s <= 0
            or not 0 < self.minimum_activation < 1
            or low < 0
            or high > 1
            or low > high
            or self.maximum_excitation_rate_per_s < 0
        ):
            raise ValueError("invalid analytic recruitment parameters")

    def build_models(self) -> tuple[HillMuscleModel, ActivationDynamics]:
        """Create the analytic proposal model; this does not copy native laws."""
        return (
            HillMuscleModel(
                MuscleParameters(
                    F_max=self.max_isometric_force_n,
                    l_opt=self.optimal_fiber_length_m,
                    l_slack=self.tendon_slack_length_m,
                    pennation_angle=self.pennation_angle_rad,
                )
            ),
            ActivationDynamics(
                tau_act=self.activation_time_constant_s,
                tau_deact=self.deactivation_time_constant_s,
                min_activation=self.minimum_activation,
            ),
        )


@dataclass(frozen=True)
class NativeMusclePathState:
    name: str
    absolute_path: str
    muscle_tendon_length_m: float
    moment_arm_m_per_rad: float
    activation: float
    previous_raw_excitation: float
    activation_state_path: str
    fiber_length_state_path: str
    fiber_length_m: float
    fiber_velocity_m_s: float
    analytic: AnalyticRecruitmentConfig
    native_actuation_force_n: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.absolute_path.startswith("/")
            or self.absolute_path.rstrip("/").rsplit("/", 1)[-1] != self.name
            or not self.activation_state_path
            or not self.fiber_length_state_path
        ):
            raise ValueError("muscle and complete-state paths are required")
        numeric = (
            self.muscle_tendon_length_m,
            self.moment_arm_m_per_rad,
            self.activation,
            self.previous_raw_excitation,
            self.fiber_length_m,
            self.fiber_velocity_m_s,
        )
        if any(not math.isfinite(float(value)) for value in numeric) or (
            self.native_actuation_force_n is not None
            and not math.isfinite(self.native_actuation_force_n)
        ):
            raise ValueError("native muscle state values must be finite")
        if self.muscle_tendon_length_m <= 0 or not 0 <= self.activation <= 1:
            raise ValueError("invalid native muscle length or activation")
        if not 0 <= self.previous_raw_excitation <= 1:
            raise ValueError("prior raw excitation must lie in [0, 1]")


@dataclass(frozen=True)
class NativePassiveEffortObservation:
    """Native component readback tied to one exact state and coordinate."""

    source_sha256: str
    runtime_identity: str
    time_s: float
    native_state_sha256: str
    coordinate_path: str
    effort_unit: str
    path_contributions_n_m: tuple[tuple[str, float], ...]
    component_names_by_path: tuple[tuple[str, tuple[NativePassiveComponent, ...]], ...]

    def __post_init__(self) -> None:
        contributions = tuple(
            (str(path), float(value)) for path, value in self.path_contributions_n_m
        )
        components = tuple(
            (str(path), tuple(names)) for path, names in self.component_names_by_path
        )
        object.__setattr__(self, "path_contributions_n_m", contributions)
        object.__setattr__(self, "component_names_by_path", components)
        contribution_paths = tuple(path for path, _ in contributions)
        if (
            len(self.source_sha256) != 64
            or len(self.native_state_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.native_state_sha256.lower()
            )
            or not self.runtime_identity
            or not self.coordinate_path
            or self.effort_unit != "N*m"
            or not math.isfinite(self.time_s)
            or not contributions
            or len(contribution_paths) != len(set(contribution_paths))
            or any(
                not path or not math.isfinite(value) for path, value in contributions
            )
            or tuple(path for path, _ in components) != contribution_paths
            or any(
                not names
                or any(not isinstance(name, NativePassiveComponent) for name in names)
                or len(names) != len(set(names))
                for _, names in components
            )
        ):
            raise ValueError("passive effort must be a complete native observation")

    @property
    def generalized_effort_n_m(self) -> float:
        return sum(value for _, value in self.path_contributions_n_m)


def native_state_fingerprint(
    *,
    source_sha256: str,
    runtime_identity: str,
    time_s: float,
    coordinate: NativeCoordinateMapping,
    paths: Sequence[NativeMusclePathState],
    required_state_paths: Sequence[str],
    complete_state: Sequence[tuple[str, float]],
) -> str:
    """Bind an observation to the exact source, geometry, path order, and state."""
    payload = {
        "source_sha256": source_sha256,
        "runtime_identity": runtime_identity,
        "time_s": float(time_s),
        "coordinate": asdict(coordinate),
        "paths": [asdict(path) for path in paths],
        "required_state_paths": list(required_state_paths),
        "complete_state": [[name, float(value)] for name, value in complete_state],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NativeMuscleStateSnapshot:
    source_sha256: str
    runtime_identity: str
    time_s: float
    coordinate: NativeCoordinateMapping
    paths: tuple[NativeMusclePathState, ...]
    required_state_paths: tuple[str, ...]
    complete_state: tuple[tuple[str, float], ...]
    passive_effort: NativePassiveEffortObservation | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(
            self, "required_state_paths", tuple(self.required_state_paths)
        )
        object.__setattr__(
            self,
            "complete_state",
            tuple((str(name), float(value)) for name, value in self.complete_state),
        )
        if len(self.source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_sha256.lower()
        ):
            raise ValueError("source SHA-256 must contain 64 hexadecimal characters")
        if not self.runtime_identity or not math.isfinite(self.time_s):
            raise ValueError("runtime identity and finite native time are required")
        if len(self.paths) != 2 or len({path.name for path in self.paths}) != 2:
            raise ValueError("the initial adapter requires exactly two named muscles")
        if len({path.absolute_path for path in self.paths}) != 2:
            raise ValueError("native muscle paths must be unique")
        arms = [path.moment_arm_m_per_rad for path in self.paths]
        if not (arms[0] * arms[1] < 0):
            raise ValueError("fixture paths must have opposing native moment-arm signs")
        expected_state_paths = (
            self.coordinate.coordinate_value_state_path,
            self.coordinate.coordinate_speed_state_path,
            *(path.activation_state_path for path in self.paths),
            *(path.fiber_length_state_path for path in self.paths),
        )
        if len(self.required_state_paths) != len(
            set(self.required_state_paths)
        ) or not set(expected_state_paths).issubset(self.required_state_paths):
            raise ValueError(
                "bound state schema omits required coordinate or muscle state"
            )
        state_paths = tuple(name for name, _ in self.complete_state)
        if state_paths != self.required_state_paths:
            raise ValueError(
                "complete native state paths/order differ from bound schema"
            )
        if any(
            not name or not math.isfinite(float(value))
            for name, value in self.complete_state
        ):
            raise ValueError("complete native state values must be finite")
        state_values = dict(self.complete_state)
        for path in self.paths:
            if not math.isclose(
                state_values[path.activation_state_path],
                path.activation,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("path activation differs from complete native state")
            if not math.isclose(
                state_values[path.fiber_length_state_path],
                path.fiber_length_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("path fiber length differs from complete native state")
        if self.passive_effort is not None and (
            self.passive_effort.native_state_sha256
            != native_state_fingerprint(
                source_sha256=self.source_sha256,
                runtime_identity=self.runtime_identity,
                time_s=self.time_s,
                coordinate=self.coordinate,
                paths=self.paths,
                required_state_paths=self.required_state_paths,
                complete_state=self.complete_state,
            )
            or self.passive_effort.source_sha256 != self.source_sha256
            or self.passive_effort.runtime_identity != self.runtime_identity
            or self.passive_effort.time_s != self.time_s
            or self.passive_effort.coordinate_path != self.coordinate.coordinate_path
            or self.passive_effort.effort_unit != self.coordinate.effort_unit
            or tuple(path for path, _ in self.passive_effort.path_contributions_n_m)
            != tuple(path.absolute_path for path in self.paths)
        ):
            raise ValueError("passive effort observation is stale or mismatched")


@dataclass(frozen=True)
class NativeMuscleEffortRequest:
    """Declares how the stateless feedback request contributes to total effort."""

    basis: EffortBasis

    def __post_init__(self) -> None:
        if not isinstance(self.basis, EffortBasis):
            raise ValueError("effort basis must be an explicit EffortBasis")


@dataclass(frozen=True)
class NativeMuscleEffortProposal:
    proposal_id: int
    schedule_index: int
    feedback_channel_id: str
    source_sha256: str
    runtime_identity: str
    coordinate_path: str
    start_time_s: float
    expected_end_time_s: float
    effort_basis: EffortBasis
    requested_total_effort_n_m: float
    allocation_status: str
    feasible: bool
    raw_excitations: tuple[tuple[str, float], ...]
    effective_excitations: tuple[tuple[str, float], ...]
    effective_bounds: tuple[tuple[str, tuple[float, float]], ...]
    activations: tuple[tuple[str, float], ...]
    generalized_effort: tuple[tuple[str, float], ...]
    residual_by_coordinate: tuple[tuple[str, float], ...]
    muscle_forces_n: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    nfev: int
    message: str


@dataclass(frozen=True)
class NativeAdvanceToken:
    proposal_id: int
    expected_end_time_s: float


class NativeMuscleEffortSession:
    """Stateless proposals and fail-stop acceptance around native advancement.

    Callers must keep all nested native model, geometry, and state inputs fixed
    while proposing. A rejected proposal leaves time/raw-history unchanged.
    Once native Manager advancement begins, any exception or invalid returned
    state poisons this session: Python bookkeeping cannot undo a partial plant
    integration.
    """

    def __init__(
        self,
        snapshot: NativeMuscleStateSnapshot,
        *,
        dt_s: float,
        first_schedule_index: int,
    ) -> None:
        self._validate_dt(dt_s)
        if (
            isinstance(first_schedule_index, bool)
            or not isinstance(first_schedule_index, int)
            or first_schedule_index < 0
        ):
            raise ValueError("first schedule index must be a nonnegative integer")
        self._snapshot = snapshot
        self._dt_s = float(dt_s)
        self._next_schedule_index = first_schedule_index
        self._next_id = 1
        self._latest: NativeMuscleEffortProposal | None = None
        self._pending: NativeAdvanceToken | None = None
        self._poisoned = False

    @staticmethod
    def _validate_dt(dt_s: float) -> None:
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("native control step must be positive and finite")

    @property
    def snapshot(self) -> NativeMuscleStateSnapshot:
        return self._snapshot

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def propose(
        self,
        request: NativeMuscleEffortRequest,
        feedback: FeedbackRequest,
        *,
        effort_tolerance_n_m: float,
        max_nfev: int = 80,
    ) -> NativeMuscleEffortProposal:
        self._require_usable()
        if self._pending is not None:
            raise RuntimeError("native advance is pending")
        snapshot = self._snapshot
        expected_end_time = snapshot.time_s + self._dt_s
        if not math.isfinite(expected_end_time) or expected_end_time <= snapshot.time_s:
            raise ValueError(
                "native control step produces a non-finite or non-advancing time"
            )
        target = self._request_target_effort(request, feedback, effort_tolerance_n_m)
        coordinate_name = snapshot.coordinate.coordinate_name
        paths = self._build_recruitment_paths(snapshot, coordinate_name)
        allocation: MuscleAllocationResult = allocate_muscle_excitations(
            paths,
            (coordinate_name,),
            (target,),
            coordinate_effort_units=("N*m",),
            dt_s=self._dt_s,
            effort_tolerances=(effort_tolerance_n_m,),
            max_nfev=max_nfev,
        )
        self._validate_allocation(allocation, paths, coordinate_name)
        proposal = self._freeze_proposal(
            request,
            feedback,
            snapshot,
            expected_end_time,
            coordinate_name,
            target,
            allocation,
        )
        self._latest = proposal
        self._next_id += 1
        return proposal

    def _request_target_effort(
        self,
        request: NativeMuscleEffortRequest,
        feedback: FeedbackRequest,
        effort_tolerance_n_m: float,
    ) -> float:
        snapshot = self._snapshot
        coordinate = snapshot.coordinate
        if not math.isfinite(feedback.time_s) or not math.isfinite(feedback.dt_s):
            raise ValueError("feedback request clock must be finite")
        if abs(feedback.time_s - snapshot.time_s) > 1e-10:
            raise ValueError("feedback request time differs from native snapshot")
        if abs(feedback.dt_s - self._dt_s) > 1e-12:
            raise ValueError("feedback request step differs from session step")
        if (
            isinstance(feedback.schedule_index, bool)
            or not isinstance(feedback.schedule_index, int)
            or feedback.schedule_index != self._next_schedule_index
        ):
            raise ValueError("feedback schedule index is stale or out of sequence")
        if len(feedback.generalized_effort) != coordinate.u_count:
            raise ValueError("feedback effort dimension differs from audited mapping")
        if feedback.selected_coordinate_mask != (True,):
            raise ValueError("feedback coordinate mask differs from audited mapping")
        channel_values = (
            *feedback.nominal_feedforward,
            *feedback.feedback_correction,
            *feedback.total_requested,
        )
        if (
            feedback.channel_ids != (coordinate.feedback_channel_id,)
            or len(feedback.nominal_feedforward) != 1
            or len(feedback.feedback_correction) != 1
            or len(feedback.total_requested) != 1
            or any(not math.isfinite(float(value)) for value in channel_values)
            or not math.isclose(
                feedback.nominal_feedforward[0] + feedback.feedback_correction[0],
                feedback.total_requested[0],
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(
                "feedback channel values/order are incomplete or non-finite"
            )
        if feedback.generalized_indices != (coordinate.generalized_effort_index,):
            raise ValueError("feedback generalized indices differ from audited mapping")
        if coordinate.generalized_effort_index >= len(feedback.generalized_effort):
            raise ValueError("feedback effort omits mapped native coordinate")
        policy_effort = float(
            feedback.generalized_effort[coordinate.generalized_effort_index]
        )
        if not math.isfinite(policy_effort) or not math.isclose(
            policy_effort,
            feedback.total_requested[0],
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("feedback generalized effort differs from channel request")
        if not math.isfinite(effort_tolerance_n_m) or effort_tolerance_n_m <= 0:
            raise ValueError("effort tolerance must be positive and finite")
        if request.basis is EffortBasis.TOTAL_MUSCULOTENDINOUS:
            return policy_effort
        if request.basis is EffortBasis.ACTIVE_INCREMENT:
            passive = snapshot.passive_effort
            if passive is None:
                raise ValueError(
                    "active increment requires native passive effort observation"
                )
            return policy_effort + passive.generalized_effort_n_m
        raise ValueError("unsupported effort basis")

    @staticmethod
    def _build_recruitment_paths(
        snapshot: NativeMuscleStateSnapshot,
        coordinate_name: str,
    ) -> tuple[MuscleRecruitmentPath, ...]:
        paths: list[MuscleRecruitmentPath] = []
        for path in snapshot.paths:
            muscle, dynamics = path.analytic.build_models()
            paths.append(
                MuscleRecruitmentPath(
                    name=path.name,
                    muscle=muscle,
                    activation_dynamics=dynamics,
                    muscle_tendon_length_m=path.muscle_tendon_length_m,
                    moment_arms_m_or_rad={coordinate_name: path.moment_arm_m_per_rad},
                    activation=path.activation,
                    previous_excitation=path.previous_raw_excitation,
                    excitation_bounds=path.analytic.excitation_bounds,
                    max_excitation_rate_per_s=path.analytic.maximum_excitation_rate_per_s,
                )
            )
        return tuple(paths)

    def _freeze_proposal(
        self,
        request: NativeMuscleEffortRequest,
        feedback: FeedbackRequest,
        snapshot: NativeMuscleStateSnapshot,
        expected_end_time: float,
        coordinate_name: str,
        target: float,
        allocation: MuscleAllocationResult,
    ) -> NativeMuscleEffortProposal:
        coordinate = snapshot.coordinate
        return NativeMuscleEffortProposal(
            proposal_id=self._next_id,
            schedule_index=feedback.schedule_index,
            feedback_channel_id=coordinate.feedback_channel_id,
            source_sha256=snapshot.source_sha256,
            runtime_identity=snapshot.runtime_identity,
            coordinate_path=coordinate.coordinate_path,
            start_time_s=snapshot.time_s,
            expected_end_time_s=expected_end_time,
            effort_basis=request.basis,
            requested_total_effort_n_m=target,
            allocation_status=allocation.status,
            feasible=allocation.feasible,
            raw_excitations=tuple(
                (path.name, allocation.excitations[path.name])
                for path in snapshot.paths
            ),
            effective_excitations=tuple(
                (path.name, allocation.effective_excitations[path.name])
                for path in snapshot.paths
            ),
            effective_bounds=tuple(
                (
                    path.name,
                    (
                        float(allocation.effective_bounds[path.name][0]),
                        float(allocation.effective_bounds[path.name][1]),
                    ),
                )
                for path in snapshot.paths
            ),
            activations=tuple(
                (path.name, allocation.activations[path.name])
                for path in snapshot.paths
            ),
            generalized_effort=(
                (coordinate_name, allocation.generalized_effort[coordinate_name]),
            ),
            residual_by_coordinate=(
                (coordinate_name, allocation.residual_by_coordinate[coordinate_name]),
            ),
            muscle_forces_n=tuple(
                (
                    path.name,
                    tuple(allocation.muscle_forces_n[path.name].items()),
                )
                for path in snapshot.paths
            ),
            nfev=allocation.nfev,
            message=allocation.message,
        )

    def _validate_allocation(
        self,
        allocation: MuscleAllocationResult,
        paths: Sequence[MuscleRecruitmentPath],
        coordinate_name: str,
    ) -> None:
        names = {path.name for path in paths}
        if (allocation.status == "feasible") != allocation.feasible:
            raise ValueError("allocator status and feasibility flag disagree")
        maps = (
            allocation.excitations,
            allocation.effective_excitations,
            allocation.effective_bounds,
            allocation.activations,
            allocation.muscle_forces_n,
        )
        if any(set(values) != names for values in maps):
            raise ValueError("allocator omitted or added a muscle control")
        for path in paths:
            raw = allocation.excitations[path.name]
            effective = allocation.effective_excitations[path.name]
            activation = allocation.activations[path.name]
            low, high = path.excitation_bounds
            slew = path.max_excitation_rate_per_s * self._dt_s
            effective_low, effective_high = allocation.effective_bounds[path.name]
            expected_bounds = (
                max(low, path.previous_excitation - slew),
                min(high, path.previous_excitation + slew),
            )
            if (
                not math.isclose(
                    effective_low, expected_bounds[0], rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(
                    effective_high, expected_bounds[1], rel_tol=0.0, abs_tol=1e-12
                )
                or not all(
                    math.isfinite(float(value))
                    for value in (raw, effective, activation)
                )
                or raw < max(low, path.previous_excitation - slew) - 1e-12
                or raw > min(high, path.previous_excitation + slew) + 1e-12
                or not 0 <= effective <= 1
                or not math.isclose(
                    effective,
                    max(raw, float(path.activation_dynamics.min_activation)),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not 0 <= activation <= 1
                or not 0 <= effective_low <= effective_high <= 1
            ):
                raise ValueError(
                    "allocator returned invalid raw/effective control values"
                )
        if (
            set(allocation.generalized_effort) != {coordinate_name}
            or set(allocation.residual_by_coordinate) != {coordinate_name}
            or set(allocation.muscle_forces_n) != names
            or allocation.nfev < 0
        ):
            raise ValueError("allocator returned incomplete candidate diagnostics")
        values = (
            *allocation.generalized_effort.values(),
            *allocation.residual_by_coordinate.values(),
            *(
                value
                for components in allocation.muscle_forces_n.values()
                for value in components.values()
            ),
        )
        if any(not allocation.muscle_forces_n[path.name] for path in paths):
            raise ValueError("allocator omitted per-muscle force diagnostics")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("allocator returned non-finite candidate diagnostics")

    def reject(self, proposal: NativeMuscleEffortProposal) -> None:
        self._require_usable()
        if self._pending is not None:
            raise RuntimeError("cannot reject while native advance is pending")
        if proposal is not self._latest:
            raise ValueError("stale proposal")
        self._latest = None

    def begin_advance(self, proposal: NativeMuscleEffortProposal) -> NativeAdvanceToken:
        self._require_usable()
        if self._pending is not None:
            raise RuntimeError("native advance is already pending")
        if proposal is not self._latest:
            raise ValueError("stale proposal")
        if not proposal.feasible or proposal.allocation_status != "feasible":
            raise ValueError("only feasible analytic proposals may be advanced")
        token = NativeAdvanceToken(proposal.proposal_id, proposal.expected_end_time_s)
        self._pending = token
        return token

    def accept_advanced(
        self, token: NativeAdvanceToken, advanced: NativeMuscleStateSnapshot
    ) -> None:
        self._require_usable()
        self._require_pending(token)
        proposal = self._latest
        assert proposal is not None
        try:
            self._validate_accepted(proposal, advanced)
        except Exception:
            self._poisoned = True
            self._pending = None
            self._latest = None
            raise
        self._snapshot = advanced
        self._next_schedule_index = proposal.schedule_index + 1
        self._pending = None
        self._latest = None

    def fail_advance(self, token: NativeAdvanceToken, error: BaseException) -> None:
        self._require_usable()
        self._require_pending(token)
        self._pending = None
        self._latest = None
        self._poisoned = True
        raise RuntimeError("native advance failed; session is poisoned") from error

    def _validate_accepted(
        self,
        proposal: NativeMuscleEffortProposal,
        advanced: NativeMuscleStateSnapshot,
    ) -> None:
        prior = self._snapshot
        if abs(advanced.time_s - proposal.expected_end_time_s) > 1e-10:
            raise ValueError(
                "advanced native state time differs from expected end time"
            )
        if (
            advanced.source_sha256 != prior.source_sha256
            or advanced.runtime_identity != prior.runtime_identity
            or advanced.coordinate != prior.coordinate
            or tuple(
                (
                    path.name,
                    path.absolute_path,
                    path.analytic,
                    path.activation_state_path,
                    path.fiber_length_state_path,
                )
                for path in advanced.paths
            )
            != tuple(
                (
                    path.name,
                    path.absolute_path,
                    path.analytic,
                    path.activation_state_path,
                    path.fiber_length_state_path,
                )
                for path in prior.paths
            )
        ):
            raise ValueError(
                "advanced native state identity differs from proposal source"
            )
        prior_state_paths = tuple(name for name, _ in prior.complete_state)
        advanced_state_paths = tuple(name for name, _ in advanced.complete_state)
        if advanced_state_paths != prior_state_paths:
            raise ValueError(
                "advanced native state schema differs from accepted baseline"
            )
        expected_raw = dict(proposal.raw_excitations)
        for path in advanced.paths:
            if not math.isclose(
                path.previous_raw_excitation,
                expected_raw[path.name],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "advanced state does not record accepted raw excitation"
                )

    def _require_pending(self, token: NativeAdvanceToken) -> None:
        if self._pending is None or token is not self._pending:
            raise ValueError("no native advance is pending for this token")

    def _require_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError("native muscle session is poisoned")
