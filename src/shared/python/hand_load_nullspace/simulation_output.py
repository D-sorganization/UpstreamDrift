# fmt: off
# ruff: noqa: E301, E302, E305, E306
"""Versioned immutable serializable SimulationOutput and DynamicsQuery interface (#15794 B2 / #15731).

Unavailable channels carry no payload. CoP threshold is 5 N. An available channel may carry
NaN only where the quantity is computed as undefined (the ground CoP in flight, #15859);
infinite values and placeholder zero-fills are refused.
"""
from __future__ import annotations
from abc import abstractmethod
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
import numpy as np
from src.shared.python.biomechanics.ground_reaction import center_of_pressure
from src.shared.python.hand_load_nullspace.solver import _post, _pre, _vec

__all__ = [
    "Channel",
    "SimulationOutput",
    "DynamicsQuery",
    "DefaultDynamicsQuery",
    "cop_channel",
    "simulation_output",
    "serialize_simulation_output",
    "deserialize_simulation_output",
    "superposition_holds",
    "zvcf_from_pinned",
    "SCHEMA_VERSION",
    "DEFAULT_COP_MIN_FZ_N",
    "DEFAULT_SUPERPOSITION_RTOL",
]

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
        """Keep UNAVAILABLE value-free with a reason and available channels value-bearing."""
        if self.availability == "UNAVAILABLE":
            if self.values is not None:
                raise ValueError("UNAVAILABLE channel must not carry values (no zero-fill)")
            if not self.reason.strip():
                raise ValueError("UNAVAILABLE channel requires a reason")
            return
        _pre(self.availability == "available", "availability must be available or UNAVAILABLE")
        if self.values is None:
            raise ValueError("available channel requires values (no zero-fill)")
        # NaN marks a quantity computed as undefined (flight CoP, #15859); infinity is a
        # numerical blow-up and is never a measurement, so only the latter is refused here.
        if any(math.isinf(item) for item in self.values):
            raise ValueError("available channel must not carry infinite values")
        if self.reason:
            raise ValueError("available channel must not carry a reason")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the channel, omitting an absent payload and an empty reason."""
        d: dict[str, Any] = {
            "availability": self.availability,
            "units": self.units,
        }
        if self.values is not None:
            d["values"] = list(self.values)
        if self.reason:
            d["reason"] = self.reason
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Channel:
        """Rebuild a channel from its serialized mapping."""
        values_raw = data.get("values")
        values = tuple(float(v) for v in values_raw) if values_raw is not None else None
        return cls(
            availability=str(data["availability"]),
            units=str(data["units"]),
            values=values,
            reason=str(data.get("reason", "")),
        )


def _channel(values, units: str) -> Channel:
    """Wrap a computed numeric payload as an available channel."""
    return Channel("available", units, tuple(float(v) for v in np.asarray(values)))


def _missing(reason: str, units: str) -> Channel:
    """Report a quantity that could not be computed, naming why and carrying nothing."""
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
        """Validate the schema, the enums, lineage strings, scalar finiteness and flight state."""
        _pre(self.schema_version == SCHEMA_VERSION, "schema_version is not supported")
        _pre(bool(self.frame.strip()) and math.isfinite(self.time_s), "frame and time_s are required")
        _pre(self.load_kind in {"total", "ztcf", "zvcf"}, "load_kind is not recognised")
        _pre(self.replay_badge in {"allocation-only", "forward-replayed"}, "replay badge is not recognised")
        for name in ("match_id", "model_definition_hash", "subject_profile_hash", "match_artifact_hash"):
            value = getattr(self, name)
            _pre(isinstance(value, str), f"{name} must be a string")
            if value and value.strip() != value:
                raise ValueError(f"{name} must not have surrounding whitespace")
        _pre(math.isfinite(self.bos_margin_m), "bos_margin_m must be finite")
        _pre(math.isfinite(self.gate_alpha), "gate_alpha must be finite")
        _post(
            self.replay_badge != "forward-replayed" or self.f_l_n.availability == "available",
            "forward-replayed requires available hand forces",
        )
        # Nothing supports the body in flight, so a valid base of support there is a
        # contradiction rather than a reportable state (#15859).
        _post(
            self.contact_state != "flight" or not self.bos_valid,
            "flight requires bos_valid false (undefined)",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize every field, omitting channels that were never populated."""
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "frame": self.frame,
            "time_s": float(self.time_s),
            "load_kind": self.load_kind,
            "replay_badge": self.replay_badge,
            "net_force_n": self.net_force_n.to_dict(),
            "midhands_moment_nm": self.midhands_moment_nm.to_dict(),
            "f_l_n": self.f_l_n.to_dict(),
            "f_r_n": self.f_r_n.to_dict(),
            "m_l_nm": self.m_l_nm.to_dict(),
            "m_r_nm": self.m_r_nm.to_dict(),
            "ground_force_n": self.ground_force_n.to_dict(),
            "ground_moment_nm": self.ground_moment_nm.to_dict(),
            "cop_m": self.cop_m.to_dict(),
            "match_id": self.match_id,
            "model_definition_hash": self.model_definition_hash,
            "subject_profile_hash": self.subject_profile_hash,
            "match_artifact_hash": self.match_artifact_hash,
            "bos_valid": self.bos_valid,
            "bos_margin_m": float(self.bos_margin_m),
            "contact_state": self.contact_state,
            "cop_segment_id": int(self.cop_segment_id),
            "gate_alpha": float(self.gate_alpha),
            "filter_provenance": self.filter_provenance,
        }
        optional_channel_fields = (
            "cop_wrench_m",
            "cop_pressure_centroid_m",
            "com_body_m",
            "com_body_club_m",
            "mass_total_kg",
            "momentum_residual_n",
        )
        for field_name in optional_channel_fields:
            ch = getattr(self, field_name)
            if ch is not None:
                d[field_name] = ch.to_dict()
        return d

    def to_json(self, **kwargs) -> str:
        """Render the dictionary form as a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationOutput:
        """Rebuild an output, deriving an absent bos_valid from the contact state."""
        _pre(data.get("schema_version") == SCHEMA_VERSION, "schema_version is not supported")
        contact_state = str(data.get("contact_state", "flight"))
        def _get_ch(name: str) -> Channel:
            """Read a required channel from the payload."""
            return Channel.from_dict(data[name])

        def _maybe_ch(name: str) -> Channel | None:
            """Read an optional channel, returning None when the payload omits it."""
            raw = data.get(name)
            return Channel.from_dict(raw) if raw is not None else None

        return cls(
            schema_version=str(data["schema_version"]),
            frame=str(data["frame"]),
            time_s=float(data["time_s"]),
            load_kind=str(data["load_kind"]),
            replay_badge=str(data["replay_badge"]),
            net_force_n=_get_ch("net_force_n"),
            midhands_moment_nm=_get_ch("midhands_moment_nm"),
            f_l_n=_get_ch("f_l_n"),
            f_r_n=_get_ch("f_r_n"),
            m_l_nm=_get_ch("m_l_nm"),
            m_r_nm=_get_ch("m_r_nm"),
            ground_force_n=_get_ch("ground_force_n"),
            ground_moment_nm=_get_ch("ground_moment_nm"),
            cop_m=_get_ch("cop_m"),
            match_id=str(data.get("match_id", "")),
            model_definition_hash=str(data.get("model_definition_hash", "")),
            subject_profile_hash=str(data.get("subject_profile_hash", "")),
            match_artifact_hash=str(data.get("match_artifact_hash", "")),
            cop_wrench_m=_maybe_ch("cop_wrench_m"),
            cop_pressure_centroid_m=_maybe_ch("cop_pressure_centroid_m"),
            # A payload without bos_valid must not claim a valid base of support while
            # claiming flight; an explicit true alongside flight fails closed (#15859).
            bos_valid=bool(data.get("bos_valid", contact_state != "flight")),
            bos_margin_m=float(data.get("bos_margin_m", 0.0)),
            contact_state=contact_state,
            cop_segment_id=int(data.get("cop_segment_id", 0)),
            gate_alpha=float(data.get("gate_alpha", 1.0)),
            filter_provenance=str(data.get("filter_provenance", "raw")),
            com_body_m=_maybe_ch("com_body_m"),
            com_body_club_m=_maybe_ch("com_body_club_m"),
            mass_total_kg=_maybe_ch("mass_total_kg"),
            momentum_residual_n=_maybe_ch("momentum_residual_n"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> SimulationOutput:
        """Rebuild an output from its JSON string."""
        return cls.from_dict(json.loads(json_str))


def serialize_simulation_output(output: SimulationOutput) -> str:
    """Serialize SimulationOutput to JSON string."""
    return output.to_json()


def deserialize_simulation_output(json_str: str) -> SimulationOutput:
    """Deserialize SimulationOutput from JSON string."""
    return SimulationOutput.from_json(json_str)


@runtime_checkable
class DynamicsQuery(Protocol):
    """Optional separate interface for live dynamics queries (M(q), C+g, J_midhands)."""

    @abstractmethod
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the generalized inertia matrix M(q)."""

    @abstractmethod
    def bias_forces(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Compute bias forces C(q, qdot) + g(q)."""

    @abstractmethod
    def jacobian_midhands(self, q: np.ndarray) -> np.ndarray:
        """Compute spatial Jacobian J_midhands(q) mapping joint velocities to midhands twist."""


class DefaultDynamicsQuery:
    """Default implementation of DynamicsQuery taking callable providers."""

    def __init__(
        self,
        *,
        mass_matrix_fn: Callable[[np.ndarray], np.ndarray],
        bias_forces_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        jacobian_midhands_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Bind the three provider callables used for live dynamics queries."""
        self._mass_matrix_fn = mass_matrix_fn
        self._bias_forces_fn = bias_forces_fn
        self._jacobian_midhands_fn = jacobian_midhands_fn

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return M(q) from the provider, refusing non-finite state or result."""
        q_arr = np.asarray(q, dtype=float)
        _pre(bool(np.all(np.isfinite(q_arr))), "q must contain finite values")
        M = self._mass_matrix_fn(q_arr)
        _post(bool(np.all(np.isfinite(M))), "M(q) must contain finite values")
        return M

    def bias_forces(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Return C(q,qdot)+g(q) from the provider, refusing non-finite state or result."""
        q_arr = np.asarray(q, dtype=float)
        qdot_arr = np.asarray(qdot, dtype=float)
        _pre(bool(np.all(np.isfinite(q_arr))), "q must contain finite values")
        _pre(bool(np.all(np.isfinite(qdot_arr))), "qdot must contain finite values")
        cg = self._bias_forces_fn(q_arr, qdot_arr)
        _post(bool(np.all(np.isfinite(cg))), "C(q,qdot)+g(q) must contain finite values")
        return cg

    def jacobian_midhands(self, q: np.ndarray) -> np.ndarray:
        """Return J_midhands(q) from the provider, refusing non-finite state or result."""
        q_arr = np.asarray(q, dtype=float)
        _pre(bool(np.all(np.isfinite(q_arr))), "q must contain finite values")
        J = self._jacobian_midhands_fn(q_arr)
        _post(bool(np.all(np.isfinite(J))), "J_midhands must contain finite values")
        return J


def cop_channel(force_n, moment_nm, point_m, *, min_fz_n: float = DEFAULT_COP_MIN_FZ_N) -> Channel:
    """Ground CoP channel, UNAVAILABLE with a reason once Fz falls below the threshold."""
    force, moment, point = _vec(force_n, "force"), _vec(moment_nm, "moment"), _vec(point_m, "point")
    _pre(math.isfinite(min_fz_n) and min_fz_n >= 0, "min_fz_n must be non-negative")
    about_origin = moment + np.cross(point, force)
    cop = center_of_pressure(force, about_origin, ground_height_m=float(point[2]), min_fz=min_fz_n)
    if cop is None:
        return _missing(f"CoP UNAVAILABLE: F_z < {min_fz_n} N", "m")
    return _channel(cop, "m")


def _hand(values, units: str, reason: str) -> Channel:
    """Wrap an optional hand-load vector, reporting the solution reason when absent."""
    if values is None:
        return _missing(reason or "hand load unavailable", units)
    return _channel(values, units)


def simulation_output(problem, solution, *, frame, time_s, ground=None, cop_min_fz_n=DEFAULT_COP_MIN_FZ_N, lineage=("", "", "", ""), **biomech):
    """Assemble a SimulationOutput frame from a solved hand-load problem and biomechanics."""
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
    if contact_state == "flight":
        # No ground contact means no base of support, and a ground CoP derived from a wrench the
        # flight state denies is not a position: publish NaN instead of it (#15859).
        bos_valid = False
        if cop.availability == "available":
            cop = _channel([math.nan, math.nan, math.nan], "m")
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
    """True when the ztcf and zvcf wrenches add back to the total within rtol."""
    _pre(math.isfinite(rtol) and rtol > 0, "rtol must be positive")
    def ok(total, left, right) -> bool:
        """Compare one recovered vector pair against the total at the shared scale."""
        whole, a, b = _vec(total, "total"), _vec(left, "ztcf"), _vec(right, "zvcf")
        scale = max(float(np.linalg.norm(whole)), 1.0)
        return float(np.linalg.norm(a + b - whole)) <= rtol * scale
    return ok(total_f, ztcf_f, zvcf_f) and ok(total_m, ztcf_m, zvcf_m)


def zvcf_from_pinned(total_f, total_m, ztcf_f, ztcf_m):
    """Derive the free contribution by subtracting the pinned ZTCF wrench from the total."""
    force = _vec(total_f, "total_f") - _vec(ztcf_f, "ztcf_f")
    moment = _vec(total_m, "total_m") - _vec(ztcf_m, "ztcf_m")
    _post(superposition_holds(total_f, ztcf_f, force, total_m, ztcf_m, moment), "pinned ZVCF must superpose")
    return force, moment
# fmt: on
