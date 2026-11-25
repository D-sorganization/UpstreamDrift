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
def simulation_output(problem, solution, *, frame, time_s, ground_force_n=None, ground_moment_nm=None, ground_point_m=None, cop_min_fz_n=DEFAULT_COP_MIN_FZ_N, match_id="", model_definition_hash="", subject_profile_hash="", match_artifact_hash=""):
    _pre(problem.load_kind == solution.load_kind, "problem and solution load_kind differ")
    reason = solution.reason or "hand load unavailable"
    if ground_force_n is None:
        ground_f = _missing("ground wrench unavailable", "N")
        ground_m = _missing("ground wrench unavailable", "N*m")
        cop = _missing("CoP UNAVAILABLE: ground wrench unavailable", "m")
    else:
        _pre(ground_moment_nm is not None and ground_point_m is not None, "ground wrench needs force, moment, and point")
        ground_f, ground_m = _channel(ground_force_n, "N"), _channel(ground_moment_nm, "N*m")
        cop = cop_channel(ground_force_n, ground_moment_nm, ground_point_m, min_fz_n=cop_min_fz_n)
    return SimulationOutput(
        SCHEMA_VERSION, frame, float(time_s), solution.load_kind, solution.replay_badge,
        _hand(solution.net_force_n, "N", reason), _hand(solution.midhands_moment_nm, "N*m", reason),
        _hand(solution.f_l_n, "N", reason), _hand(solution.f_r_n, "N", reason),
        _hand(solution.m_l_nm, "N*m", reason), _hand(solution.m_r_nm, "N*m", reason),
        ground_f, ground_m, cop, match_id, model_definition_hash, subject_profile_hash, match_artifact_hash,
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
