"""Portable model exchange, iteration throughput benchmark, and conformance contracts.

Addresses issue #11104 (MMR-18).
Enforces:
1. Measured cold vs warm stage timing, peak memory, and speedup on identical captures/hardware.
2. Content-addressed cache invalidation covering 7 explicit axes (model, geometry,
   marker map, initial state, solver, controls, provider revision).
3. Session reuse guard enforcing rebuild on topology or operating-point changes.
4. Identical input metric parity within declared tolerances.
5. Optimization winner verification via uncached native cold replay.
6. Fail-closed rejection of unsupported neck, contact, and muscle mappings to prohibit
   silent loss of dynamics.
7. Export and import of named coordinates, frames, mass/inertia, contacts, and actuation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

from src.shared.python.contracts import postcondition, precondition, require

PORTABLE_EXCHANGE_SCHEMA_VERSION = "portable-model-exchange/1.0.0"


class EngineType(str, Enum):
    """Supported physics engines and model variants for portable exchange."""

    SIMSCAPE_REDUCED_27 = "simscape_reduced_27"
    SIMSCAPE_GS3DX = "simscape_gs3dx"
    PINOCCHIO = "pinocchio"
    OPENSIM = "opensim"
    MUJOCO = "mujoco"
    DRAKE = "drake"


class FeatureKind(str, Enum):
    """Physical and actuation capabilities present in a model specification."""

    RIGID_MULTIBODY = "rigid_multibody"
    POLYNOMIAL_ACTUATION = "polynomial_actuation"
    TORQUE_ACTUATION = "torque_actuation"
    INDEPENDENT_NECK_DOFS = "independent_neck_dofs"
    HILL_MUSCLE_DYNAMICS = "hill_muscle_dynamics"
    VOLUMETRIC_PENALTY_CONTACT = "volumetric_penalty_contact"
    BILATERAL_CONSTRAINT = "bilateral_constraint"


class UnsupportedMappingError(ValueError):
    """Raised when an engine mapping would drop unsupported physical dynamics."""


class SessionRebuildRequiredError(RuntimeError):
    """Raised when a warm session is requested but model rebuild is required."""


class ParityToleranceExceededError(ValueError):
    """Raised when metric parity between cold and warm or repeat runs exceeds tolerance."""


class UncachedColdReplayRequiredError(ValueError):
    """Raised when an optimization winner is not verified with an uncached cold replay."""


class WinnerReplayVerificationError(ValueError):
    """Raised when an optimization winner's cold replay metrics diverge from declared metrics."""


class ExchangeValidationError(ValueError):
    """Raised when portable exchange contracts or payloads are invalid."""


@dataclass(frozen=True, slots=True)
class TopologySpec:
    """Kinematic tree and coordinate definition."""

    coordinate_names: tuple[str, ...]
    dof_count: int
    has_independent_neck: bool
    joint_types: Mapping[str, str]

    def __post_init__(self) -> None:
        require(len(self.coordinate_names) > 0, "coordinate_names must be non-empty")
        require(
            self.dof_count == len(self.coordinate_names),
            f"dof_count ({self.dof_count}) must match coordinate_names count ({len(self.coordinate_names)})",
        )
        require(
            isinstance(self.has_independent_neck, bool),
            "has_independent_neck must be bool",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_names": list(self.coordinate_names),
            "dof_count": self.dof_count,
            "has_independent_neck": self.has_independent_neck,
            "joint_types": dict(self.joint_types),
        }


@dataclass(frozen=True, slots=True)
class OperatingPointSpec:
    """Nominal base orientation, ground placement, and operating environment."""

    plane_tilt_deg: float
    ground_offset_m: tuple[float, float, float]
    gravity_m_s2: tuple[float, float, float]
    nominal_cadence_hz: float

    def __post_init__(self) -> None:
        require(math.isfinite(self.plane_tilt_deg), "plane_tilt_deg must be finite")
        require(len(self.ground_offset_m) == 3, "ground_offset_m must be a 3-tuple")
        require(len(self.gravity_m_s2) == 3, "gravity_m_s2 must be a 3-tuple")
        require(self.nominal_cadence_hz > 0.0, "nominal_cadence_hz must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_tilt_deg": self.plane_tilt_deg,
            "ground_offset_m": list(self.ground_offset_m),
            "gravity_m_s2": list(self.gravity_m_s2),
            "nominal_cadence_hz": self.nominal_cadence_hz,
        }


@dataclass(frozen=True, slots=True)
class PortableModelSpec:
    """Complete, self-contained model specification for portable exchange."""

    model_id: str
    model_sha256: str
    provider_revision: str
    geometry: Mapping[str, Any]
    marker_map: Mapping[str, str]
    initial_state: Mapping[str, float]
    solver: Mapping[str, Any]
    controls: Mapping[str, Any]
    topology: TopologySpec
    operating_point: OperatingPointSpec
    features: tuple[FeatureKind, ...] = ()

    def __post_init__(self) -> None:
        require(bool(self.model_id.strip()), "model_id must be non-empty")
        require(bool(self.model_sha256.strip()), "model_sha256 must be non-empty")
        require(
            bool(self.provider_revision.strip()), "provider_revision must be non-empty"
        )
        require(len(self.geometry) > 0, "geometry must be non-empty")
        require(len(self.marker_map) > 0, "marker_map must be non-empty")
        require(len(self.initial_state) > 0, "initial_state must be non-empty")
        require(len(self.solver) > 0, "solver configuration must be non-empty")
        require(len(self.controls) > 0, "controls must be non-empty")


@dataclass(frozen=True, slots=True)
class IterationBenchmarkSample:
    """Stage throughput and peak memory benchmark measurement."""

    capture_id: str
    hardware_id: str
    stage: str
    cold_wall_clock_s: float
    warm_wall_clock_s: float
    cold_peak_memory_mb: float
    warm_peak_memory_mb: float
    cache_key: str
    cache_hit: bool
    speedup_ratio: float
    memory_delta_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "hardware_id": self.hardware_id,
            "stage": self.stage,
            "cold_wall_clock_s": self.cold_wall_clock_s,
            "warm_wall_clock_s": self.warm_wall_clock_s,
            "cold_peak_memory_mb": self.cold_peak_memory_mb,
            "warm_peak_memory_mb": self.warm_peak_memory_mb,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "speedup_ratio": self.speedup_ratio,
            "memory_delta_mb": self.memory_delta_mb,
        }


def record_iteration_benchmark(
    capture_id: str,
    hardware_id: str,
    stage: str,
    cold_wall_clock_s: float,
    warm_wall_clock_s: float,
    cold_peak_memory_mb: float,
    warm_peak_memory_mb: float,
    **kwargs: Any,
) -> IterationBenchmarkSample:
    """Record cold vs warm timing and memory metrics on the same capture/hardware."""
    if not str(capture_id).strip():
        raise ValueError("capture_id must be non-empty")
    if not str(hardware_id).strip():
        raise ValueError("hardware_id must be non-empty")
    if cold_wall_clock_s < 0.0 or warm_wall_clock_s < 0.0:
        raise ValueError("timings must be non-negative")
    if not math.isfinite(cold_peak_memory_mb) or not math.isfinite(warm_peak_memory_mb):
        raise ValueError("memory values must be finite")
    if cold_peak_memory_mb < 0.0 or warm_peak_memory_mb < 0.0:
        raise ValueError("memory values must be non-negative")

    raw_cache_key = kwargs.pop("cache_key", "")
    cache_key = str(raw_cache_key).strip()
    if not cache_key:
        raise ValueError("cache_key must be non-empty")
    cache_hit = bool(kwargs.pop("cache_hit", False))
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    speedup = cold_wall_clock_s / warm_wall_clock_s if warm_wall_clock_s > 0.0 else 1.0
    mem_delta = warm_peak_memory_mb - cold_peak_memory_mb

    return IterationBenchmarkSample(
        capture_id=str(capture_id).strip(),
        hardware_id=str(hardware_id).strip(),
        stage=str(stage).strip(),
        cold_wall_clock_s=float(cold_wall_clock_s),
        warm_wall_clock_s=float(warm_wall_clock_s),
        cold_peak_memory_mb=float(cold_peak_memory_mb),
        warm_peak_memory_mb=float(warm_peak_memory_mb),
        cache_key=cache_key,
        cache_hit=cache_hit,
        speedup_ratio=float(speedup),
        memory_delta_mb=float(mem_delta),
    )


def compute_content_cache_key(spec: PortableModelSpec) -> str:
    """Compute a 64-character SHA-256 digest covering all 7 invalidation axes.

    Axes:
    1. model (model_id, model_sha256)
    2. geometry
    3. marker_map
    4. initial_state
    5. solver
    6. controls
    7. provider_revision
    """
    canonical_payload = {
        "1_model": {
            "model_id": spec.model_id,
            "model_sha256": spec.model_sha256.lower(),
        },
        "2_geometry": sorted(spec.geometry.items()),
        "3_marker_map": sorted(spec.marker_map.items()),
        "4_initial_state": sorted(spec.initial_state.items()),
        "5_solver": sorted(spec.solver.items()),
        "6_controls": sorted(spec.controls.items()),
        "7_provider_revision": spec.provider_revision,
    }
    encoded = json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionRebuildDecision:
    """Decision indicating whether warm session reuse is permitted or rebuild is required."""

    must_rebuild: bool
    can_reuse_session: bool
    reason: str


def _find_session_rebuild_reason(
    prev_spec: PortableModelSpec, new_spec: PortableModelSpec
) -> str | None:
    """Check spec fields for changes that mandate rebuilding session."""
    if (
        prev_spec.topology.coordinate_names != new_spec.topology.coordinate_names
        or prev_spec.topology.dof_count != new_spec.topology.dof_count
        or prev_spec.topology.has_independent_neck
        != new_spec.topology.has_independent_neck
        or prev_spec.topology.joint_types != new_spec.topology.joint_types
    ):
        return "topology_changed: coordinate or joint tree modified"

    op_prev, op_new = prev_spec.operating_point, new_spec.operating_point
    if (
        abs(op_prev.plane_tilt_deg - op_new.plane_tilt_deg) > 1e-9
        or op_prev.ground_offset_m != op_new.ground_offset_m
        or op_prev.gravity_m_s2 != op_new.gravity_m_s2
        or abs(op_prev.nominal_cadence_hz - op_new.nominal_cadence_hz) > 1e-9
    ):
        return "operating_point_changed: base orientation, ground or gravity altered"

    if (
        prev_spec.model_id != new_spec.model_id
        or prev_spec.model_sha256 != new_spec.model_sha256
        or prev_spec.provider_revision != new_spec.provider_revision
    ):
        return "model_or_provider_changed: binary/source definition changed"

    if prev_spec.geometry != new_spec.geometry:
        return "geometry_changed: link length or mass properties modified"
    if prev_spec.solver != new_spec.solver:
        return "solver_changed: integration method or tolerances modified"
    if prev_spec.marker_map != new_spec.marker_map:
        return "marker_map_changed: attachment association modified"
    if prev_spec.initial_state != new_spec.initial_state:
        return "initial_state_changed: start pose modified"
    return None


def evaluate_session_reuse(
    prev_spec: PortableModelSpec,
    new_spec: PortableModelSpec,
    *,
    fail_closed: bool = False,
) -> SessionRebuildDecision:
    """Determine whether session reuse is permitted or a model rebuild is required.

    Topology changes and operating-point changes trigger a mandatory rebuild.
    Geometry, solver, and provider changes also invalidate session state.
    Only tunable controls alone permit warm session reuse.
    """
    reason = _find_session_rebuild_reason(prev_spec, new_spec)
    if reason is not None:
        if fail_closed:
            raise SessionRebuildRequiredError(f"Rebuild required: {reason}")
        return SessionRebuildDecision(
            must_rebuild=True,
            can_reuse_session=False,
            reason=reason,
        )

    return SessionRebuildDecision(
        must_rebuild=False,
        can_reuse_session=True,
        reason="tunable_controls_alone: session reuse permitted",
    )


@dataclass(frozen=True, slots=True)
class ParityVerdict:
    """Outcome of numerical parity evaluation between runs."""

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    diffs: Mapping[str, float]


def evaluate_metric_parity(
    reference_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    fail_closed: bool = False,
) -> ParityVerdict:
    """Evaluate whether candidate metrics match reference within declared tolerances."""
    require(len(reference_metrics) > 0, "reference_metrics must not be empty")
    require(len(candidate_metrics) > 0, "candidate_metrics must not be empty")

    diffs: dict[str, float] = {}
    max_abs = 0.0
    max_rel = 0.0
    passed = True

    for k, ref_val in reference_metrics.items():
        if k not in candidate_metrics:
            raise ValueError(f"Missing metric {k!r} in candidate_metrics")
        cand_val = candidate_metrics[k]
        abs_diff = abs(cand_val - ref_val)
        diffs[k] = abs_diff
        if abs_diff > max_abs:
            max_abs = abs_diff

        denom = max(abs(ref_val), 1e-12)
        rel_diff = abs_diff / denom
        if rel_diff > max_rel:
            max_rel = rel_diff

        if abs_diff > atol and rel_diff > rtol:
            passed = False

    if not passed and fail_closed:
        raise ParityToleranceExceededError(
            f"Metric parity check failed: max_abs_diff={max_abs:.6e} > atol={atol:.6e} "
            f"or max_rel_diff={max_rel:.6e} > rtol={rtol:.6e} (exceeded declared tolerance)"
        )

    return ParityVerdict(
        passed=passed,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        diffs=diffs,
    )


@dataclass(frozen=True, slots=True)
class WinnerReplayReceipt:
    """Audit receipt verifying that the optimization winner was evaluated on native cold replay."""

    candidate_id: str
    candidate_sha256: str
    is_uncached_cold: bool
    verified: bool
    replay_metrics: Mapping[str, float]
    max_discrepancy: float


def verify_optimization_winner_cold_replay(
    *,
    candidate: Mapping[str, Any],
    cold_replay_fn: Callable[[Mapping[str, Any], bool], Mapping[str, float]],
    tolerance_atol: float = 1e-5,
    enforce_uncached: bool = True,
    is_uncached_cold: bool = True,
) -> WinnerReplayReceipt:
    """Execute an uncached native cold replay for the optimization winner candidate.

    Enforces that:
    1. Replay is strictly uncached native cold (`is_uncached_cold=True`).
    2. Replay reproduced declared candidate metrics within `tolerance_atol`.
    """
    if enforce_uncached and not is_uncached_cold:
        raise UncachedColdReplayRequiredError(
            "Optimization winner cold replay must be executed with is_uncached_cold=True"
        )

    candidate_id = str(candidate.get("candidate_id", "unknown"))
    candidate_sha = str(candidate.get("candidate_sha256", ""))
    declared = candidate.get("declared_metrics", {})
    if not declared:
        raise ValueError("candidate missing declared_metrics")

    # Run native cold replay
    replay_metrics = cold_replay_fn(candidate, is_uncached_cold)

    # Evaluate parity against declared metrics
    verdict = evaluate_metric_parity(
        reference_metrics=declared,
        candidate_metrics=replay_metrics,
        atol=tolerance_atol,
        rtol=1e-4,
        fail_closed=False,
    )

    if not verdict.passed:
        raise WinnerReplayVerificationError(
            f"Optimization winner cold replay metrics diverged from declared metrics: "
            f"max_diff={verdict.max_abs_diff:.6e} > atol={tolerance_atol:.6e}"
        )

    return WinnerReplayReceipt(
        candidate_id=candidate_id,
        candidate_sha256=candidate_sha,
        is_uncached_cold=True,
        verified=True,
        replay_metrics=dict(replay_metrics),
        max_discrepancy=verdict.max_abs_diff,
    )


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Declares capabilities of a physics engine / model platform."""

    engine: EngineType
    supports_rigid_multibody: bool
    supports_polynomial_actuation: bool
    supports_torque_actuation: bool
    supports_neck: bool
    supports_muscle: bool
    supports_volumetric_contact: bool
    supports_bilateral_constraint: bool


@dataclass(frozen=True, slots=True)
class InterchangeConformanceMatrix:
    """Authoritative matrix of engine support across modeling and actuation features."""

    capabilities: Mapping[EngineType, EngineCapabilities]

    @classmethod
    def default(cls) -> InterchangeConformanceMatrix:
        caps = {
            EngineType.SIMSCAPE_REDUCED_27: EngineCapabilities(
                engine=EngineType.SIMSCAPE_REDUCED_27,
                supports_rigid_multibody=True,
                supports_polynomial_actuation=True,
                supports_torque_actuation=True,
                supports_neck=False,
                supports_muscle=False,
                supports_volumetric_contact=True,
                supports_bilateral_constraint=True,
            ),
            EngineType.SIMSCAPE_GS3DX: EngineCapabilities(
                engine=EngineType.SIMSCAPE_GS3DX,
                supports_rigid_multibody=True,
                supports_polynomial_actuation=True,
                supports_torque_actuation=True,
                supports_neck=True,
                supports_muscle=False,
                supports_volumetric_contact=True,
                supports_bilateral_constraint=True,
            ),
            EngineType.PINOCCHIO: EngineCapabilities(
                engine=EngineType.PINOCCHIO,
                supports_rigid_multibody=True,
                supports_polynomial_actuation=False,
                supports_torque_actuation=True,
                supports_neck=True,
                supports_muscle=False,
                supports_volumetric_contact=False,
                supports_bilateral_constraint=True,
            ),
            EngineType.OPENSIM: EngineCapabilities(
                engine=EngineType.OPENSIM,
                supports_rigid_multibody=True,
                supports_polynomial_actuation=False,
                supports_torque_actuation=True,
                supports_neck=True,
                supports_muscle=True,
                supports_volumetric_contact=False,
                supports_bilateral_constraint=True,
            ),
            EngineType.MUJOCO: EngineCapabilities(
                engine=EngineType.MUJOCO,
                supports_rigid_multibody=True,
                supports_polynomial_actuation=False,
                supports_torque_actuation=True,
                supports_neck=True,
                supports_muscle=True,
                supports_volumetric_contact=True,
                supports_bilateral_constraint=True,
            ),
            EngineType.DRAKE: EngineCapabilities(
                engine=EngineType.DRAKE,
                supports_rigid_multibody=True,
                supports_polynomial_actuation=False,
                supports_torque_actuation=True,
                supports_neck=True,
                supports_muscle=False,
                supports_volumetric_contact=True,
                supports_bilateral_constraint=True,
            ),
        }
        return cls(capabilities=caps)

    def get_capabilities(self, engine: EngineType | str) -> EngineCapabilities:
        key = EngineType(engine) if isinstance(engine, str) else engine
        if key not in self.capabilities:
            raise ValueError(f"Unrecognized engine {engine}")
        return self.capabilities[key]


def export_portable_model(
    spec: PortableModelSpec,
    target_engine: EngineType | str,
    conformance_matrix: InterchangeConformanceMatrix | None = None,
) -> dict[str, Any]:
    """Export a portable model specification while enforcing physical conformance.

    Rejects mappings that would silently drop neck, muscle, or contact dynamics.
    """
    matrix = conformance_matrix or InterchangeConformanceMatrix.default()
    target_type = (
        EngineType(target_engine) if isinstance(target_engine, str) else target_engine
    )
    caps = matrix.get_capabilities(target_type)

    # 1. Independent neck check
    has_neck = (
        spec.topology.has_independent_neck
        or FeatureKind.INDEPENDENT_NECK_DOFS in spec.features
    )
    if has_neck and not caps.supports_neck:
        raise UnsupportedMappingError(
            f"Target engine {target_type.value!r} independent neck DOFs not supported; "
            f"silent drop of neck dynamics is strictly prohibited."
        )

    # 2. Muscle dynamics check
    if FeatureKind.HILL_MUSCLE_DYNAMICS in spec.features and not caps.supports_muscle:
        raise UnsupportedMappingError(
            f"Target engine {target_type.value!r} muscle dynamics not supported; "
            f"silent drop of Hill-type muscle dynamics is strictly prohibited."
        )

    # 3. Volumetric penalty contact check
    if (
        FeatureKind.VOLUMETRIC_PENALTY_CONTACT in spec.features
        and not caps.supports_volumetric_contact
    ):
        raise UnsupportedMappingError(
            f"Target engine {target_type.value!r} volumetric penalty contact not supported; "
            f"silent drop of contact dynamics is strictly prohibited."
        )

    return {
        "schema_version": PORTABLE_EXCHANGE_SCHEMA_VERSION,
        "target_engine": target_type.value,
        "model_id": spec.model_id,
        "model_sha256": spec.model_sha256,
        "provider_revision": spec.provider_revision,
        "geometry": dict(spec.geometry),
        "marker_map": dict(spec.marker_map),
        "initial_state": dict(spec.initial_state),
        "solver": dict(spec.solver),
        "controls": dict(spec.controls),
        "topology": spec.topology.to_dict(),
        "operating_point": spec.operating_point.to_dict(),
        "features": [f.value for f in spec.features],
    }


def import_portable_model(data: Mapping[str, Any]) -> PortableModelSpec:
    """Import and validate a portable model specification payload."""
    require("schema_version" in data, "schema_version required")
    require(
        data["schema_version"] == PORTABLE_EXCHANGE_SCHEMA_VERSION,
        "invalid schema_version",
    )

    topo_raw = data["topology"]
    topology = TopologySpec(
        coordinate_names=tuple(topo_raw["coordinate_names"]),
        dof_count=int(topo_raw["dof_count"]),
        has_independent_neck=bool(topo_raw["has_independent_neck"]),
        joint_types=dict(topo_raw.get("joint_types", {})),
    )

    op_raw = data["operating_point"]
    go = op_raw["ground_offset_m"]
    gv = op_raw["gravity_m_s2"]
    operating_point = OperatingPointSpec(
        plane_tilt_deg=float(op_raw["plane_tilt_deg"]),
        ground_offset_m=(float(go[0]), float(go[1]), float(go[2])),
        gravity_m_s2=(float(gv[0]), float(gv[1]), float(gv[2])),
        nominal_cadence_hz=float(op_raw["nominal_cadence_hz"]),
    )

    features = tuple(FeatureKind(f) for f in data.get("features", ()))

    return PortableModelSpec(
        model_id=str(data["model_id"]),
        model_sha256=str(data["model_sha256"]),
        provider_revision=str(data["provider_revision"]),
        geometry=dict(data["geometry"]),
        marker_map=dict(data["marker_map"]),
        initial_state=dict(data["initial_state"]),
        solver=dict(data["solver"]),
        controls=dict(data["controls"]),
        topology=topology,
        operating_point=operating_point,
        features=features,
    )
