# fmt: off
# ruff: noqa: E301, E302, E305, E306
"""SimulationOutput contract, shared wrench/CoP invariants, and #14756 lineage gate (#15731)."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json, math, re  # noqa: E401
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from src.shared.python.biomechanics.ground_reaction import center_of_pressure
from src.shared.python.biomechanics.metric_definitions.schema import GolfMetricDefinitionEntry
from src.shared.python.contracts import ensure, require

SCHEMA_VERSION, DEFAULT_COP_FZ_THRESHOLD_N, DEFAULT_SUPERPOSITION_RTOL = "simulation-output/1.0.0", 5.0, 1e-4
METRIC_REGISTRY_ISSUE, METRIC_REGISTRY_OWNER = "#14421", "biomechanics.metric_definitions"
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_MATCH_ID_RE = re.compile(r"^(?:sha256:[0-9a-f]{64}|[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9_.-]{7,127})$")
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?$")
_MUTABLE = frozenset({"latest", "main", "master", "head", "dev", "develop", "current", "draft", "tip"})
_CORE = ("left_hand_force_n", "right_hand_force_n", "left_hand_couple_nm", "right_hand_couple_nm", "midhands_force_n", "midhands_moment_nm", "ground_force_n", "ground_moment_nm", "cop_m")
class SimulationOutputContractError(ValueError):
    """Base error raised when a SimulationOutput contract invariant fails."""
class UnavailableFieldError(SimulationOutputContractError):
    """Raised when callers attempt to read raw values from an UNAVAILABLE field."""
class SuperpositionGateError(SimulationOutputContractError):
    """Raised when the ZTCF + ZVCF linear superposition gate fails."""
class LineageValidationError(SimulationOutputContractError):
    """Raised when SimulationLineage or #14421 registry alignment is invalid."""
class FieldAvailability(str, Enum):
    """Explicit availability state for every physical output quantity."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
def _guard(ok: bool, msg: str, exc: type[ValueError] = SimulationOutputContractError) -> None:
    """Raise ``exc(msg)`` when ``ok`` is falsy."""
    if not ok:
        raise exc(msg)
def _non_empty(val: Any, name: str, exc: type[ValueError] = SimulationOutputContractError) -> str:
    """Validate a non-empty trimmed string."""
    _guard(isinstance(val, str) and bool(val.strip()) and val.strip() == val, f"{name} must be non-empty", exc)
    return str(val)
def _finite(val: Any) -> bool:
    """Return True iff ``val`` is a finite int/float (excluding bool)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))
def _vec3(val: Any, name: str) -> tuple[float, float, float]:
    """Validate and freeze a finite 3-vector."""
    _guard(not isinstance(val, (str, bytes)) and isinstance(val, (Sequence, np.ndarray)), f"{name} must be 3-vector")
    arr = np.asarray(val, dtype=float)
    _guard(arr.shape == (3,) and bool(np.all(np.isfinite(arr))), f"{name} must be finite 3-vector")
    return (float(arr[0]), float(arr[1]), float(arr[2]))
def _norm_values(raw: Any, *, allow_2d: bool = True) -> tuple[Any, ...]:
    """Normalize 1D or 2D values into immutable tuples, rejecting non-finite numbers."""
    seq = raw.tolist() if isinstance(raw, np.ndarray) else raw
    _guard(isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)) and bool(seq), "values must be non-empty")
    first = next((x for x in seq if x is not None), None)
    _guard(first is not None, "all-None values must be marked UNAVAILABLE")
    if allow_2d and isinstance(first, (Sequence, np.ndarray)) and not isinstance(first, (str, bytes)):
        rows = [None if x is None else _norm_values(x, allow_2d=False) for x in seq]
        _guard(len({len(r) for r in rows if r is not None}) == 1, "inconsistent 2D sample width")
        return tuple(rows)
    out = [None if (allow_2d and x is None) else float(x) for x in seq if (allow_2d and x is None) or _finite(x)]
    _guard(len(out) == len(seq), "values must be finite numbers")
    return tuple(out)
def _immutable(val: Any, name: str, pattern: re.Pattern[str], label: str) -> str:
    """Validate an immutable hash or identifier, rejecting mutable refs and paths."""
    tok = _non_empty(val, name, LineageValidationError)
    bad = any(c.isspace() for c in tok) or "/" in tok or "\\" in tok or ".." in tok or PureWindowsPath(tok).is_absolute() or PurePosixPath(tok).is_absolute()
    _guard(not bad and tok.casefold() not in _MUTABLE and bool(pattern.fullmatch(tok)), f"{name} must be {label}, got {tok!r}", LineageValidationError)
    return tok
def _rel_asset(val: Any) -> str:
    """Validate a relative asset key without path traversal or absolute roots."""
    rel = _non_empty(val, "model_assets_sha256 key", LineageValidationError)
    _guard(not (PureWindowsPath(rel).is_absolute() or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel.replace("\\", "/")).parts), f"unsafe asset path {rel!r}", LineageValidationError)
    return rel
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert frozen contract dataclasses/mappings/tuples to JSON types."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj) if getattr(obj, f.name) is not None}
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return [_to_jsonable(v) for v in obj] if isinstance(obj, tuple) else obj
@dataclass(frozen=True)
class QuantityField:
    """Immutable tensor/series field with explicit availability (never zero-filled)."""
    availability: FieldAvailability
    units: str
    frame: str
    values: tuple[Any, ...] | None = None
    reason: str | None = None
    application_point_m: tuple[float, float, float] | None = None
    capability_note: str | None = None
    def __post_init__(self) -> None:
        """Enforce availability, finiteness, and no-zero-fill invariants."""
        av = FieldAvailability(self.availability)
        object.__setattr__(self, "availability", av)
        _non_empty(self.units, "units")
        _non_empty(self.frame, "frame")
        if self.application_point_m is not None:
            object.__setattr__(self, "application_point_m", _vec3(self.application_point_m, "application_point_m"))
        if av is FieldAvailability.UNAVAILABLE:
            _guard(self.values is None, "UNAVAILABLE field must have values=None (no zero-fill)")
            _non_empty(self.reason, "reason")
            return
        _guard(self.values is not None and self.reason is None, "AVAILABLE field requires values and no reason")
        object.__setattr__(self, "values", _norm_values(self.values))
    @property
    def is_available(self) -> bool:
        """Return True iff the field carries valid physical values."""
        return self.availability is FieldAvailability.AVAILABLE
    def require_values(self) -> tuple[Any, ...]:
        """Return raw values when AVAILABLE or fail closed with UnavailableFieldError."""
        if self.availability is FieldAvailability.UNAVAILABLE or self.values is None:
            raise UnavailableFieldError(f"field is UNAVAILABLE: {self.reason}")
        return self.values
    @classmethod
    def available(cls, values: Any, *, units: str, frame: str, application_point_m: tuple[float, float, float] | None = None, capability_note: str | None = None) -> QuantityField:
        """Construct an AVAILABLE QuantityField from numeric values."""
        return cls(FieldAvailability.AVAILABLE, units, frame, values, None, application_point_m, capability_note)
    @classmethod
    def unavailable(cls, *, units: str, frame: str, reason: str, capability_note: str | None = None) -> QuantityField:
        """Construct an explicit UNAVAILABLE QuantityField with no numeric payload."""
        return cls(FieldAvailability.UNAVAILABLE, units, frame, None, reason, None, capability_note)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantityField:
        """Deserialize a QuantityField from a dictionary."""
        raw_pt = data.get("application_point_m")
        return cls(FieldAvailability(str(data["availability"])), str(data["units"]), str(data["frame"]), data.get("values"), data.get("reason"), _vec3(raw_pt, "application_point_m") if raw_pt is not None else None, data.get("capability_note"))
@dataclass(frozen=True)
class CapabilityFlags:
    """Explicit capability availability flags for engine outputs."""
    hand_forces: FieldAvailability = FieldAvailability.UNAVAILABLE
    hand_couples: FieldAvailability = FieldAvailability.UNAVAILABLE
    ground_wrench: FieldAvailability = FieldAvailability.UNAVAILABLE
    ground_cop: FieldAvailability = FieldAvailability.UNAVAILABLE
    ztcf_zvcf_superposition: FieldAvailability = FieldAvailability.UNAVAILABLE
    intrinsic_couples_included: bool = False
    fixed_state_contact_verified: bool = False
    def __post_init__(self) -> None:
        """Coerce FieldAvailability enum flags."""
        for attr in ("hand_forces", "hand_couples", "ground_wrench", "ground_cop", "ztcf_zvcf_superposition"):
            object.__setattr__(self, attr, FieldAvailability(getattr(self, attr)))
@dataclass(frozen=True)
class RegistryAlignment:
    """Alignment record binding #14421 canonical metric-definitions ownership."""
    registry_epic: str = METRIC_REGISTRY_ISSUE
    registry_owner: str = METRIC_REGISTRY_OWNER
    definition_version: str = "1.0.0"
    metric_ids: tuple[str, ...] = ()
    metric_definitions_sha256: str | None = None
    superposition_rtol: float = DEFAULT_SUPERPOSITION_RTOL
    cop_fz_threshold_n: float = DEFAULT_COP_FZ_THRESHOLD_N
    def __post_init__(self) -> None:
        """Enforce #14421 registry ownership, immutable version, and tolerances."""
        _guard(self.registry_epic == METRIC_REGISTRY_ISSUE and self.registry_owner == METRIC_REGISTRY_OWNER, "registry ownership must align with #14421", LineageValidationError)
        ver = _non_empty(self.definition_version, "definition_version", LineageValidationError)
        _guard(ver.casefold() not in _MUTABLE and bool(_SEMVER_RE.fullmatch(ver)), f"definition_version must be semver, got {ver!r}", LineageValidationError)
        object.__setattr__(self, "metric_ids", tuple(_non_empty(m, "metric_ids", LineageValidationError) for m in self.metric_ids))
        if self.metric_definitions_sha256 is not None:
            object.__setattr__(self, "metric_definitions_sha256", _immutable(self.metric_definitions_sha256, "metric_definitions_sha256", _SHA256_RE, "SHA-256"))
        _guard(_finite(self.superposition_rtol) and float(self.superposition_rtol) > 0.0 and _finite(self.cop_fz_threshold_n) and float(self.cop_fz_threshold_n) >= 0.0, "invalid registry tolerances", LineageValidationError)
def validate_registry_entry_alignment(entry: GolfMetricDefinitionEntry, alignment: RegistryAlignment) -> None:
    """Verify a #14421 GolfMetricDefinitionEntry aligns with RegistryAlignment."""
    _guard(isinstance(entry, GolfMetricDefinitionEntry), "entry must be GolfMetricDefinitionEntry", LineageValidationError)
    _guard((not alignment.metric_ids or entry.metric_id in alignment.metric_ids) and entry.definition_version == alignment.definition_version, f"metric {entry.metric_id!r} not aligned", LineageValidationError)
@dataclass(frozen=True)
class SimulationLineage:
    """Content-addressed lineage binding #14756 run identity and #14421 registry."""
    run_id: str
    match_id: str
    model_definition_sha256: str
    subject_profile_sha256: str
    match_artifact_sha256: str
    model_assets_sha256: Mapping[str, str] = field(default_factory=dict)
    registry_alignment: RegistryAlignment = field(default_factory=RegistryAlignment)
    def __post_init__(self) -> None:
        """Reject mutable refs, paths, and invalid content hashes."""
        object.__setattr__(self, "run_id", _immutable(self.run_id, "run_id", _SHA256_RE, "SHA-256"))
        object.__setattr__(self, "match_id", _immutable(self.match_id, "match_id", _MATCH_ID_RE, "immutable id"))
        for attr in ("model_definition_sha256", "subject_profile_sha256", "match_artifact_sha256"):
            object.__setattr__(self, attr, _immutable(getattr(self, attr), attr, _SHA256_RE, "SHA-256"))
        _guard(isinstance(self.model_assets_sha256, Mapping) and isinstance(self.registry_alignment, RegistryAlignment), "invalid lineage mapping/alignment", LineageValidationError)
        object.__setattr__(self, "model_assets_sha256", MappingProxyType({_rel_asset(k): _immutable(v, f"model_assets_sha256[{k!r}]", _SHA256_RE, "SHA-256") for k, v in sorted(self.model_assets_sha256.items())}))
@dataclass(frozen=True)
class MidhandsWrenchResult:
    """Equivalent midhands wrench with explicit intrinsic-couple capability status."""
    force_n: QuantityField
    moment_nm: QuantityField
    midhands_point_m: tuple[float, float, float]
    left_couple_availability: FieldAvailability
    right_couple_availability: FieldAvailability
    intrinsic_couples_included: bool
    intrinsic_couples_mode: str
def _couple_arr(couple: QuantityField | None, shape: tuple[int, ...], frame: str) -> tuple[np.ndarray, FieldAvailability]:
    """Return couple array when available, or zeros for missing-couple summation."""
    if couple is None or not couple.is_available:
        return np.zeros(shape, dtype=float), FieldAvailability.UNAVAILABLE
    _guard(couple.frame == frame and couple.units == "N*m", "couple frame/units mismatch")
    arr = np.asarray(couple.require_values(), dtype=float)
    _guard(arr.shape == shape, f"couple shape {arr.shape} != {shape}")
    return arr, FieldAvailability.AVAILABLE
def compute_midhands_wrench(
    left_force_n: QuantityField, right_force_n: QuantityField, left_point_m: ArrayLike, right_point_m: ArrayLike, *,
    left_moment_nm: QuantityField | None = None, right_moment_nm: QuantityField | None = None, midhands_point_m: ArrayLike | None = None,
) -> MidhandsWrenchResult:
    """Compute M_mid = M_L + M_R + (p_L - p_mid) x F_L + (p_R - p_mid) x F_R."""
    p_l, p_r = _vec3(left_point_m, "left_point_m"), _vec3(right_point_m, "right_point_m")
    p_mid = _vec3(midhands_point_m, "midhands_point_m") if midhands_point_m is not None else (0.5 * (p_l[0] + p_r[0]), 0.5 * (p_l[1] + p_r[1]), 0.5 * (p_l[2] + p_r[2]))
    frame, unavail, avail = left_force_n.frame, FieldAvailability.UNAVAILABLE, FieldAvailability.AVAILABLE
    if not left_force_n.is_available or not right_force_n.is_available:
        why = left_force_n.reason or right_force_n.reason or "hand force unavailable"
        return MidhandsWrenchResult(QuantityField.unavailable(units="N", frame=frame, reason=why), QuantityField.unavailable(units="N*m", frame=frame, reason=why), p_mid, unavail, unavail, False, "unavailable")
    _guard(right_force_n.frame == frame and left_force_n.units == "N" and right_force_n.units == "N", "hand forces must share frame and 'N'")
    f_l, f_r = np.asarray(left_force_n.require_values(), dtype=float), np.asarray(right_force_n.require_values(), dtype=float)
    _guard(f_l.shape == f_r.shape and f_l.shape[-1] == 3, "hand forces must be (3,) or (T, 3)")
    m_l, l_av = _couple_arr(left_moment_nm, f_l.shape, frame)
    m_r, r_av = _couple_arr(right_moment_nm, f_r.shape, frame)
    m_mid = m_l + m_r + np.cross(np.asarray(p_l) - p_mid, f_l) + np.cross(np.asarray(p_r) - p_mid, f_r)
    ensure(bool(np.all(np.isfinite(m_mid))), "midhands moment must be finite")
    mode = "full" if (l_av is avail and r_av is avail) else ("partial" if (l_av is avail or r_av is avail) else "transfer_only")
    return MidhandsWrenchResult(QuantityField.available(f_l + f_r, units="N", frame=frame, application_point_m=p_mid), QuantityField.available(m_mid, units="N*m", frame=frame, application_point_m=p_mid, capability_note=mode), p_mid, l_av, r_av, l_av is avail or r_av is avail, mode)
@dataclass(frozen=True)
class SuperpositionGateResult:
    """Outcome of the ZTCF + ZVCF linear superposition gate."""
    passed: bool
    fixed_state_contact_verified: bool
    rel_tol: float
    relative_residual: float | None
    absolute_residual: float | None
    reason: str
def evaluate_ztcf_zvcf_superposition(
    total: QuantityField, ztcf: QuantityField, zvcf: QuantityField, *, fixed_state_contact_verified: bool, rel_tol: float = DEFAULT_SUPERPOSITION_RTOL,
) -> SuperpositionGateResult:
    """Evaluate ZTCF + ZVCF = Total linear superposition at fixed (q, qdot, contact)."""
    require(_finite(rel_tol) and float(rel_tol) > 0.0, "rel_tol must be positive")
    tol = float(rel_tol)
    if not fixed_state_contact_verified:
        return SuperpositionGateResult(False, False, tol, None, None, "fixed (q, qdot) and active contact set not verified")
    if not (total.is_available and ztcf.is_available and zvcf.is_available):
        return SuperpositionGateResult(False, True, tol, None, None, "superposition fields contain UNAVAILABLE entries")
    if len({total.units, ztcf.units, zvcf.units}) != 1 or len({total.frame, ztcf.frame, zvcf.frame}) != 1:
        return SuperpositionGateResult(False, True, tol, None, None, "units or frame mismatch")
    a, b, c = np.asarray(total.require_values(), dtype=float), np.asarray(ztcf.require_values(), dtype=float), np.asarray(zvcf.require_values(), dtype=float)
    if not (a.shape == b.shape == c.shape):
        return SuperpositionGateResult(False, True, tol, None, None, "shape mismatch")
    diff = b + c - a
    abs_n = np.linalg.norm(diff, axis=-1) if diff.ndim > 1 else np.linalg.norm(diff)
    scales = np.maximum(np.linalg.norm(a, axis=-1) if a.ndim > 1 else np.linalg.norm(a), 1.0)
    abs_res, rel_res = float(np.max(abs_n)), float(np.max(abs_n / scales))
    ok = rel_res <= tol
    return SuperpositionGateResult(ok, True, tol, rel_res, abs_res, "superposition verified" if ok else f"relative residual {rel_res:.3e} > {tol:.3e}")
def require_ztcf_zvcf_superposition(
    total: QuantityField, ztcf: QuantityField, zvcf: QuantityField, *, fixed_state_contact_verified: bool, rel_tol: float = DEFAULT_SUPERPOSITION_RTOL,
) -> SuperpositionGateResult:
    """Fail-closed ZTCF + ZVCF superposition gate raising SuperpositionGateError."""
    res = evaluate_ztcf_zvcf_superposition(total, ztcf, zvcf, fixed_state_contact_verified=fixed_state_contact_verified, rel_tol=rel_tol)
    if not res.passed:
        raise SuperpositionGateError(res.reason)
    return res
@dataclass(frozen=True)
class GroundWrenchAndCop:
    """Full 6D ground wrench paired with threshold-gated CoP."""
    ground_force_n: QuantityField
    ground_moment_nm: QuantityField
    cop_m: QuantityField
    fz_threshold_n: float
    ground_height_m: float
def compute_ground_cop(
    ground_force_n: QuantityField, ground_moment_nm: QuantityField, *, reference_point_m: ArrayLike = (0.0, 0.0, 0.0), ground_height_m: float = 0.0, fz_threshold_n: float = DEFAULT_COP_FZ_THRESHOLD_N,
) -> GroundWrenchAndCop:
    """Compute CoP via shared center_of_pressure while preserving full 6D ground wrench."""
    require(_finite(fz_threshold_n) and float(fz_threshold_n) >= 0.0 and _finite(ground_height_m), "invalid CoP threshold/height")
    ref, thr, zg, frame = np.asarray(_vec3(reference_point_m, "reference_point_m"), dtype=float), float(fz_threshold_n), float(ground_height_m), ground_force_n.frame
    if not ground_force_n.is_available or not ground_moment_nm.is_available:
        why = ground_force_n.reason or ground_moment_nm.reason or "ground wrench unavailable"
        return GroundWrenchAndCop(ground_force_n, ground_moment_nm, QuantityField.unavailable(units="m", frame=frame, reason=f"CoP UNAVAILABLE: {why}"), thr, zg)
    _guard(ground_moment_nm.frame == frame and ground_force_n.units == "N" and ground_moment_nm.units == "N*m", "ground wrench must share frame and 'N'/'N*m'")
    f_arr, m_arr = np.asarray(ground_force_n.require_values(), dtype=float), np.asarray(ground_moment_nm.require_values(), dtype=float)
    _guard(f_arr.shape == m_arr.shape and f_arr.shape[-1] == 3, "ground wrench must be (3,) or (T, 3)")
    rows_f, rows_m = ((f_arr,), (m_arr,)) if f_arr.ndim == 1 else (f_arr, m_arr)
    cops = [None if (c := center_of_pressure(fr, mr + np.cross(ref, fr), ground_height_m=zg, min_fz=thr)) is None else (float(c[0]), float(c[1]), float(c[2])) for fr, mr in zip(rows_f, rows_m, strict=True)]
    cop_f = QuantityField.unavailable(units="m", frame=frame, reason=f"CoP UNAVAILABLE: F_z < {thr} N") if all(x is None for x in cops) else (QuantityField.available(cops[0], units="m", frame=frame) if f_arr.ndim == 1 and cops[0] is not None else QuantityField.available(tuple(cops), units="m", frame=frame, capability_note=f"fz_threshold_n={thr}"))
    return GroundWrenchAndCop(ground_force_n, ground_moment_nm, cop_f, thr, zg)
@dataclass(frozen=True)
class SimulationOutput:
    """Versioned immutable tensor/series contract for swing report pipelines (#15731)."""
    schema_version: str
    lineage: SimulationLineage
    engine: str
    coordinate_frame: str
    time_s: tuple[float, ...]
    topology: Mapping[str, tuple[str, ...]]
    force_application_points: Mapping[str, tuple[float, float, float]]
    capabilities: CapabilityFlags
    left_hand_force_n: QuantityField
    right_hand_force_n: QuantityField
    left_hand_couple_nm: QuantityField
    right_hand_couple_nm: QuantityField
    midhands_force_n: QuantityField
    midhands_moment_nm: QuantityField
    ground_force_n: QuantityField
    ground_moment_nm: QuantityField
    cop_m: QuantityField
    extra_fields: Mapping[str, QuantityField] = field(default_factory=dict)
    superposition_gate: SuperpositionGateResult | None = None
    def __post_init__(self) -> None:
        """Validate schema version, timebase, topology, application points, and fields."""
        _guard(self.schema_version == SCHEMA_VERSION, f"unsupported schema_version {self.schema_version!r}")
        _non_empty(self.engine, "engine")
        _non_empty(self.coordinate_frame, "coordinate_frame")
        times = _norm_values(self.time_s, allow_2d=False)
        _guard(not any(b <= a for a, b in zip(times, times[1:], strict=False)), "time_s must be strictly increasing")
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "topology", MappingProxyType({_non_empty(k, "topology"): tuple(_non_empty(x, k) for x in v) for k, v in self.topology.items()}))
        object.__setattr__(self, "force_application_points", MappingProxyType({_non_empty(k, "point"): _vec3(v, k) for k, v in self.force_application_points.items()}))
        extra = {_non_empty(k, "extra"): v for k, v in dict(self.extra_fields).items()}
        object.__setattr__(self, "extra_fields", MappingProxyType(extra))
        for fname in (*_CORE, *sorted(extra)):
            qf = getattr(self, fname) if fname in _CORE else extra[fname]
            _guard(isinstance(qf, QuantityField) and qf.frame == self.coordinate_frame, f"{fname} frame mismatch")
            if qf.is_available:
                vals = qf.require_values()
                _guard(not (vals and isinstance(vals[0], tuple) and len(vals) != len(times)), f"{fname} length != time_s")
    def require_field(self, name: str) -> QuantityField:
        """Return the named QuantityField or raise KeyError if unknown."""
        return getattr(self, name) if name in _CORE else self.extra_fields[name]
    def require_field_values(self, name: str) -> tuple[Any, ...]:
        """Fail-closed raw value lookup raising UnavailableFieldError if UNAVAILABLE."""
        return self.require_field(name).require_values()
    def unavailable_fields(self) -> tuple[str, ...]:
        """Return names of all UNAVAILABLE fields in deterministic order."""
        return tuple(n for n in (*_CORE, *sorted(self.extra_fields)) if not self.require_field(n).is_available)
    def to_dict(self) -> dict[str, Any]:
        """Serialize SimulationOutput to a JSON-compatible dictionary."""
        return dict(_to_jsonable(self))
    def to_json(self) -> str:
        """Serialize SimulationOutput to a canonical JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationOutput:
        """Deserialize and validate SimulationOutput from a dictionary."""
        _guard(data.get("schema_version") == SCHEMA_VERSION, f"unsupported schema_version {data.get('schema_version')!r}")
        lin = dict(data["lineage"])
        lin["registry_alignment"] = RegistryAlignment(**{**lin["registry_alignment"], "metric_ids": tuple(lin["registry_alignment"].get("metric_ids", ()))})
        raw_gate = data.get("superposition_gate")
        return cls(
            schema_version=str(data["schema_version"]), lineage=SimulationLineage(**lin), engine=str(data["engine"]), coordinate_frame=str(data["coordinate_frame"]),
            time_s=tuple(float(x) for x in data["time_s"]), topology={str(k): tuple(v) for k, v in data["topology"].items()},
            force_application_points={str(k): _vec3(v, k) for k, v in data["force_application_points"].items()}, capabilities=CapabilityFlags(**data["capabilities"]),
            extra_fields={str(k): QuantityField.from_dict(v) for k, v in data.get("extra_fields", {}).items()},
            superposition_gate=SuperpositionGateResult(**raw_gate) if raw_gate else None, **{f: QuantityField.from_dict(data[f]) for f in _CORE},
        )
    @classmethod
    def from_json(cls, payload: str) -> SimulationOutput:
        """Deserialize and validate SimulationOutput from a JSON string."""
        return cls.from_dict(json.loads(payload))
# fmt: on
