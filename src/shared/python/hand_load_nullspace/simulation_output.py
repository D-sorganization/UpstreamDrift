# fmt: off
# ruff: noqa: E301, E302, E305, E306
"""Versioned SimulationOutput. Unavailable channels carry no payload. CoP threshold is 5 N."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from src.shared.python.biomechanics.ground_reaction import center_of_pressure
from src.shared.python.hand_load_nullspace.solver import _post, _pre, _vec
SCHEMA_VERSION = "simulation-output/1"
DEFAULT_COP_MIN_FZ_N = 5.0
DEFAULT_SUPERPOSITION_RTOL = 1e-4
@dataclass(frozen=True)
class Channel:
    availability: str
    units: str
    values: tuple[float, ...] | None = None
    reason: str = ""
    def __post_init__(self) -> None:
        if self.availability == "UNAVAILABLE":
            if self.values is not None:
                raise ValueError("UNAVAILABLE channel must not carry values (no zero-fill)")
            if not self.reason.strip():
                raise ValueError("UNAVAILABLE channel requires a reason")
            return
        _pre(self.availability == "available", "availability must be available or UNAVAILABLE")
        if self.values is None or not all(math.isfinite(item) for item in self.values):
            raise ValueError("available channel requires finite values")
        if self.reason:
            raise ValueError("available channel must not carry a reason")
def _channel(values, units: str) -> Channel:
    return Channel("available", units, tuple(float(v) for v in np.asarray(values)))
def _missing(reason: str, units: str) -> Channel:
    return Channel("UNAVAILABLE", units, None, reason)
@dataclass(frozen=True)
class SimulationOutput:
    schema_version: str
    frame: str
    time_s: float
    load_kind: str
    replay_badge: str
    net_force_n: Channel
    midhands_moment_nm: Channel
    f_l_n: Channel
    f_r_n: Channel
    m_l_nm: Channel
    m_r_nm: Channel
    ground_force_n: Channel
    ground_moment_nm: Channel
    cop_m: Channel
    match_id: str = ""
    model_definition_hash: str = ""
    subject_profile_hash: str = ""
    match_artifact_hash: str = ""
    cop_wrench_m: Channel | None = None
    cop_pressure_centroid_m: Channel | None = None
    bos_valid: bool = True
    bos_margin_m: float = 0.0
    contact_state: str = "flight"
    cop_segment_id: int = 0
    gate_alpha: float = 1.0
    filter_provenance: str = "raw"
    com_body_m: Channel | None = None
    com_body_club_m: Channel | None = None
    mass_total_kg: Channel | None = None
    momentum_residual_n: Channel | None = None
    def __post_init__(self) -> None:
        _pre(self.schema_version == SCHEMA_VERSION, "schema_version is not supported")
        _pre(bool(self.frame.strip()) and math.isfinite(self.time_s), "frame and time_s are required")
        _pre(self.load_kind in {"total", "ztcf", "zvcf"}, "load_kind is not recognised")
        _pre(self.replay_badge in {"allocation-only", "forward-replayed"}, "replay badge is not recognised")
        for name in ("match_id", "model_definition_hash", "subject_profile_hash", "match_artifact_hash"):
            value = getattr(self, name)
            _pre(isinstance(value, str), f"{name} must be a string")
            if value and value.strip() != value:
                raise ValueError(f"{name} must not have surrounding whitespace")
        _post(
            self.replay_badge != "forward-replayed" or self.f_l_n.availability == "available",
            "forward-replayed requires available hand forces",
        )
def cop_channel(force_n, moment_nm, point_m, *, min_fz_n: float = DEFAULT_COP_MIN_FZ_N) -> Channel:
    force, moment, point = _vec(force_n, "force"), _vec(moment_nm, "moment"), _vec(point_m, "point")
    _pre(math.isfinite(min_fz_n) and min_fz_n >= 0, "min_fz_n must be non-negative")
    about_origin = moment + np.cross(point, force)
    cop = center_of_pressure(force, about_origin, ground_height_m=float(point[2]), min_fz=min_fz_n)
    if cop is None:
        return _missing(f"CoP UNAVAILABLE: F_z < {min_fz_n} N", "m")
    return _channel(cop, "m")
def _hand(values, units: str, reason: str) -> Channel:
    if values is None:
        return _missing(reason or "hand load unavailable", units)
    return _channel(values, units)
def simulation_output(problem, solution, *, frame, time_s, ground=None, cop_min_fz_n=DEFAULT_COP_MIN_FZ_N, lineage=("", "", "", ""), **biomech):
    cop_wrench_m = biomech.get("cop_wrench_m")
    cop_pressure_centroid_m = biomech.get("cop_pressure_centroid_m")
    bos_valid = biomech.get("bos_valid", True)
    bos_margin_m = biomech.get("bos_margin_m", 0.0)
    contact_state = biomech.get("contact_state", "flight")
    cop_segment_id = biomech.get("cop_segment_id", 0)
    gate_alpha = biomech.get("gate_alpha", 1.0)
    filter_provenance = biomech.get("filter_provenance", "raw")
    com_body_m = biomech.get("com_body_m")
    com_body_club_m = biomech.get("com_body_club_m")
    mass_total_kg = biomech.get("mass_total_kg")
    momentum_residual_n = biomech.get("momentum_residual_n")
    _pre(problem.load_kind == solution.load_kind, "problem and solution load_kind differ")
    reason = solution.reason or "hand load unavailable"
    if ground is None:
        ground_f = _missing("ground wrench unavailable", "N")
        ground_m = _missing("ground wrench unavailable", "N*m")
        cop = _missing("CoP UNAVAILABLE: ground wrench unavailable", "m")
    else:
        _pre(len(ground) == 3, "ground wrench needs force, moment, and point")
        ground_force_n, ground_moment_nm, ground_point_m = ground
        ground_f, ground_m = _channel(ground_force_n, "N"), _channel(ground_moment_nm, "N*m")
        cop = cop_channel(ground_force_n, ground_moment_nm, ground_point_m, min_fz_n=cop_min_fz_n)
    match_id, model_def_hash, subj_hash, artifact_hash = lineage
    return SimulationOutput(
        SCHEMA_VERSION, frame, float(time_s), solution.load_kind, solution.replay_badge,
        _hand(solution.net_force_n, "N", reason), _hand(solution.midhands_moment_nm, "N*m", reason),
        _hand(solution.f_l_n, "N", reason), _hand(solution.f_r_n, "N", reason),
        _hand(solution.m_l_nm, "N*m", reason), _hand(solution.m_r_nm, "N*m", reason),
        ground_f, ground_m, cop,
        match_id=match_id,
        model_definition_hash=model_def_hash,
        subject_profile_hash=subj_hash,
        match_artifact_hash=artifact_hash,
        cop_wrench_m=cop_wrench_m or cop,
        cop_pressure_centroid_m=cop_pressure_centroid_m or _missing("pressure centroid unavailable", "m"),
        bos_valid=bos_valid,
        bos_margin_m=float(bos_margin_m),
        contact_state=contact_state,
        cop_segment_id=int(cop_segment_id),
        gate_alpha=float(gate_alpha),
        filter_provenance=filter_provenance,
        com_body_m=com_body_m or _missing("com_body unavailable", "m"),
        com_body_club_m=com_body_club_m or _missing("com_body_club unavailable", "m"),
        mass_total_kg=mass_total_kg or _missing("mass_total unavailable", "kg"),
        momentum_residual_n=momentum_residual_n or _missing("momentum_residual unavailable", "N"),
    )
def superposition_holds(total_f, ztcf_f, zvcf_f, total_m, ztcf_m, zvcf_m, *, rtol=DEFAULT_SUPERPOSITION_RTOL) -> bool:
    _pre(math.isfinite(rtol) and rtol > 0, "rtol must be positive")
    def ok(total, left, right) -> bool:
        whole, a, b = _vec(total, "total"), _vec(left, "ztcf"), _vec(right, "zvcf")
        scale = max(float(np.linalg.norm(whole)), 1.0)
        return float(np.linalg.norm(a + b - whole)) <= rtol * scale
    return ok(total_f, ztcf_f, zvcf_f) and ok(total_m, ztcf_m, zvcf_m)
def zvcf_from_pinned(total_f, total_m, ztcf_f, ztcf_m):
    force = _vec(total_f, "total_f") - _vec(ztcf_f, "ztcf_f")
    moment = _vec(total_m, "total_m") - _vec(ztcf_m, "ztcf_m")
    _post(superposition_holds(total_f, ztcf_f, force, total_m, ztcf_m, moment), "pinned ZVCF must superpose")
    return force, moment
# fmt: on
