"""Movement-variant generator for the matched swing (SPN-13b, issue #12767).

A :class:`MovementVariant` is a named, versioned perturbation of the
pelvis->thorax rotation series ``R(t)`` (thorax frame in the pelvis frame,
ADR-0055: X anterior, Y cranial, Z right), expressed through one golf-term
registry id (:mod:`src.shared.python.spine.golf_terms`). Definitions live in
``src/config/spine_movement_variants.json``.

Rotation-space operation. With ``R = Rz(FE) Rx(LB) Ry(AR)`` (``isb_spine``
intrinsic ZXY) a variant composes ``R`` with a rotation about the JCS axis of
its term, never adds angles to a re-built matrix:

* ``trunk_extension_jcs`` (FE, proximal pelvis Z axis): ``R' = Rz(a) R``;
* ``x_factor_jcs_axial`` (AR, distal thorax Y axis): ``R' = R Ry(a)``;
* ``trunk_lateral_flexion_jcs`` (LB, floating axis):
  ``R' = R Ry(AR)^T Rx(a) Ry(AR)``.

Because the JCS axes are the axes of the Cardan factors, the declared
coordinate changes by exactly ``a`` and the two other JCS coordinates are
invariant (up to float rounding). ``a = s * delta`` with ``delta`` the change of
the term in degrees and ``s`` its handed sign (ADR-0055 item 4; ``s = 1`` for
extension). ``delta(t)`` is:

* ``scale`` (factor ``k``): ``(k - 1) term(t) w(t)``;
* ``offset`` (``d`` degrees): ``d w(t)``;
* ``shift_peak`` (``m`` samples): ``term(tau(t)) - term(t)`` with ``tau`` the
  monotone piecewise-linear warp ``tau(t_s) = t_s``, ``tau(t_{p+m}) = t_p``,
  ``tau(t_e) = t_e`` (``p`` the peak of the term inside the window).

``w`` is a C1 raised-cosine window over the event times ``(start,
plateau_start, plateau_end, end)``: 0 outside, ``(1 - cos(pi u)) / 2`` on the
ramp-in, 1 on the plateau, ``(1 + cos(pi u)) / 2`` on the ramp-out. Frames
outside the window are returned bit-identical.

Non-trunk segments. ``thorax_relative_hold``: arm and head coordinates are
joint coordinates relative to the thorax and are held, so the arms follow the
thorax; the pelvis and legs are untouched. The club path in ground therefore
changes (the variant is not a re-matched swing).

Gimbal-locked frames (``|LB| = 90 deg``) inside the window raise; the coupling
laws are never touched (variants act on the global rotation only). Units:
degrees for terms and parameters, radians internally, seconds for time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ensure
from src.shared.python.spine._golf_term_impl import (
    check_handedness,
    handed_sign,
    isb_spine_angles_deg,
    rotation_series,
)
from src.shared.python.spine.golf_terms import evaluate, registered_terms

__all__ = [
    "DEFAULT_VARIANTS_PATH",
    "JCS_TERMS",
    "MovementVariant",
    "RangeCheck",
    "RangeViolation",
    "VariantResult",
    "apply_variant",
    "check_coordinate_ranges",
    "load_movement_variants",
    "movement_variant",
    "sample_variant_parameters",
]

DEFAULT_VARIANTS_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "spine_movement_variants.json"
)
JCS_TERMS = ("x_factor_jcs_axial", "trunk_extension_jcs", "trunk_lateral_flexion_jcs")
_ROT = "pelvis_to_thorax_rotation"
_OPERATIONS = ("scale", "offset", "shift_peak")
_POLICIES = ("thorax_relative_hold", "trunk_only")
_PSEUDO_EVENTS = ("start", "end")
_AXIS_COLUMN = {"trunk_extension_jcs": 0, "trunk_lateral_flexion_jcs": 1}
_REQUIRED = (
    "variant_id",
    "version",
    "description",
    "term_id",
    "operation",
    "parameter",
    "window_events",
    "term_bounds_deg",
    "undeclared_tolerance_deg",
    "affected_terms",
    "non_trunk_policy",
    "formula",
)


def _frozen(value: object, dtype: Any = float) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _pair(value: object, name: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a pair of numbers: {exc}") from exc
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ValueError(f"{name} must be finite with lower < upper, got {value}")
    return lo, hi


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


@dataclass(frozen=True)
class MovementVariant:
    """One movement-variant definition (see the module docstring).

    Preconditions (``ValueError``): non-empty text fields; ``term_id`` is one
    of :data:`JCS_TERMS` and appears in ``affected_terms`` (all registered);
    ``operation`` in ``scale``, ``offset``, ``shift_peak``; four window events
    (two for ``shift_peak``); ordered finite bounds containing the default;
    an integer default and bounds when ``parameter_integer``; positive
    ``undeclared_tolerance_deg`` (in each term's own units).
    """

    variant_id: str
    version: str
    description: str
    term_id: str
    operation: str
    parameter_name: str
    parameter_units: str
    parameter_default: float | int
    parameter_bounds: tuple[float, float]
    parameter_integer: bool
    window_events: tuple[str, ...]
    term_bounds_deg: tuple[float, float]
    undeclared_tolerance_deg: float
    affected_terms: tuple[str, ...]
    non_trunk_policy: str
    formula: str

    def __post_init__(self) -> None:
        for name in ("variant_id", "version", "description", "formula"):
            _text(getattr(self, name), name)
        _text(self.parameter_name, "parameter_name")
        if self.term_id not in JCS_TERMS:
            raise ValueError(
                f"{self.variant_id}: term_id {self.term_id!r} is not a JCS term "
                f"({', '.join(JCS_TERMS)})"
            )
        if self.operation not in _OPERATIONS:
            raise ValueError(f"{self.variant_id}: unknown operation {self.operation!r}")
        if self.non_trunk_policy not in _POLICIES:
            raise ValueError(f"{self.variant_id}: bad non_trunk_policy")
        n_events = 2 if self.operation == "shift_peak" else 4
        if len(self.window_events) != n_events:
            raise ValueError(f"{self.variant_id}: needs {n_events} window_events")
        self._check_terms()
        self._check_parameter()
        if not self.undeclared_tolerance_deg > 0.0:
            raise ValueError("undeclared_tolerance_deg must be > 0")

    def _check_terms(self) -> None:
        known = set(registered_terms())
        stray = [t for t in self.affected_terms if t not in known]
        if stray or self.term_id not in self.affected_terms:
            raise ValueError(
                f"{self.variant_id}: affected_terms must be registered and contain "
                f"term_id; unknown {stray}"
            )

    def _check_parameter(self) -> None:
        lo, hi = self.parameter_bounds
        if self.parameter_integer and not all(
            float(v).is_integer() for v in (lo, hi, self.parameter_default)
        ):
            raise ValueError(f"{self.variant_id}: integer parameter needs int values")
        if not lo <= float(self.parameter_default) <= hi:
            raise ValueError(f"{self.variant_id}: default outside parameter bounds")

    @property
    def composition_side(self) -> Literal["left", "right"]:
        """``left`` for FE (pelvis axis, ``C @ R``), else ``right`` (``R @ C``)."""
        return "left" if self.term_id == "trunk_extension_jcs" else "right"


def _variant_from_entry(entry: object) -> MovementVariant:
    if not isinstance(entry, dict):
        raise ValueError("each variant entry must be a JSON object")
    missing = [k for k in _REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"variant {entry.get('variant_id')!r} lacks keys {missing}")
    if entry["term_id"] not in registered_terms():
        raise ValueError(f"unknown golf term {entry['term_id']!r} in variant entry")
    par = entry["parameter"]
    if not isinstance(par, dict):
        raise ValueError("parameter must be a JSON object")
    integer = bool(par.get("integer", False))
    cast = int if integer else float
    return MovementVariant(
        variant_id=entry["variant_id"],
        version=str(entry["version"]),
        description=entry["description"],
        term_id=entry["term_id"],
        operation=entry["operation"],
        parameter_name=par.get("name", ""),
        parameter_units=str(par.get("units", "")),
        parameter_default=cast(par.get("default")),
        parameter_bounds=_pair(par.get("bounds"), "parameter.bounds"),
        parameter_integer=integer,
        window_events=tuple(str(e) for e in entry["window_events"]),
        term_bounds_deg=_pair(entry["term_bounds_deg"], "term_bounds_deg"),
        undeclared_tolerance_deg=float(entry["undeclared_tolerance_deg"]),
        affected_terms=tuple(str(t) for t in entry["affected_terms"]),
        non_trunk_policy=entry["non_trunk_policy"],
        formula=entry["formula"],
    )


def load_movement_variants(
    path: Path | str | None = None,
) -> Mapping[str, MovementVariant]:
    """Load the variant definitions (default :data:`DEFAULT_VARIANTS_PATH`).

    Postcondition: a read-only mapping ``variant_id -> MovementVariant``.

    Raises:
        ValueError: malformed JSON content, unknown terms or operations,
            duplicate ids.
    """
    raw = json.loads(Path(path or DEFAULT_VARIANTS_PATH).read_text(encoding="utf-8"))
    entries = raw.get("variants") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("variant file must hold a non-empty 'variants' list")
    out: dict[str, MovementVariant] = {}
    for entry in entries:
        var = _variant_from_entry(entry)
        if var.variant_id in out:
            raise ValueError(f"duplicate variant id {var.variant_id!r}")
        out[var.variant_id] = var
    return MappingProxyType(out)


def movement_variant(variant_id: str) -> MovementVariant:
    """Return the shipped variant ``variant_id``.

    Raises:
        TypeError: ``variant_id`` is not a ``str``.
        ValueError: unknown id (the message lists the known ids).
    """
    if not isinstance(variant_id, str):
        raise TypeError(f"variant_id must be a str, got {type(variant_id).__name__}")
    variants = load_movement_variants()
    if variant_id not in variants:
        raise ValueError(
            f"unknown movement variant {variant_id!r}; known: {', '.join(variants)}"
        )
    return variants[variant_id]


# ------------------------------------------------------------------ results


@dataclass(frozen=True)
class VariantResult:
    """Outcome of :func:`apply_variant` (arrays read-only).

    ``rotations`` (n, 3, 3) is the varied series and ``corrections`` the
    per-frame rotation composed on ``variant.composition_side``. ``delta_deg``
    is the change of the declared term, ``term_values_deg`` its varied values.
    ``target_values_deg`` at ``target_indices`` are the declared targets and
    ``target_error_deg`` the maximum deviation of the independently
    re-evaluated term from them. ``window`` is ``(start, end)`` sample indices.
    ``term_bounds_ok`` reports (never repairs) ``term_bounds_deg``.
    """

    variant: MovementVariant
    rotations: np.ndarray
    corrections: np.ndarray
    delta_deg: np.ndarray
    term_values_deg: np.ndarray
    target_indices: np.ndarray
    target_values_deg: np.ndarray
    target_error_deg: float
    window: tuple[int, int]
    term_bounds_ok: bool
    distribution_source: str
    provenance: Mapping[str, Any]


# --------------------------------------------------------------- validation


def _check_time(time_s: object, n: int) -> np.ndarray:
    times = np.asarray(time_s, dtype=float)
    if times.shape != (n,) or not np.all(np.isfinite(times)):
        raise ValueError(f"time_s must be a finite array of shape ({n},)")
    if n < 3 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing with >= 3 samples")
    return times


def _event_index(events: Mapping[str, int], name: str, n: int) -> int:
    if name in _PSEUDO_EVENTS:
        return 0 if name == "start" else n - 1
    if name not in events:
        raise ValueError(f"event {name!r} is required by the variant window")
    idx = _integer(events[name], f"events[{name!r}]")
    if not 0 <= idx < n:
        raise ValueError(f"events[{name!r}]={idx} out of range for {n} samples")
    return idx


def _window_indices(
    variant: MovementVariant, events: Mapping[str, int], n: int
) -> tuple[int, ...]:
    if not isinstance(events, Mapping):
        raise TypeError(f"events must be a mapping, got {type(events).__name__}")
    idx = tuple(_event_index(events, e, n) for e in variant.window_events)
    ordered = idx[0] < idx[1] if len(idx) == 2 else idx[0] < idx[1] <= idx[2] < idx[3]
    if not ordered:
        raise ValueError(
            f"window events {variant.window_events} out of order: indices {idx}"
        )
    return idx


def _parameter(variant: MovementVariant, parameter: object) -> float | int:
    if parameter is None:
        return variant.parameter_default
    name = variant.parameter_name
    value: float | int = (
        _integer(parameter, name)
        if variant.parameter_integer
        else _real(parameter, name)
    )
    lo, hi = variant.parameter_bounds
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value} outside bounds [{lo}, {hi}]")
    return value


# ------------------------------------------------------------------ kernels


def _raised_cosine_window(time_s: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """C1 window: 0 outside ``idx[0]..idx[3]``, 1 on ``idx[1]..idx[2]``."""
    t = time_s
    t0, t1, t2, t3 = (t[i] for i in idx)
    w = np.zeros_like(t)
    up = (t > t0) & (t < t1)
    w[up] = 0.5 * (1.0 - np.cos(np.pi * (t[up] - t0) / (t1 - t0)))
    w[(t >= t1) & (t <= t2)] = 1.0
    down = (t > t2) & (t < t3)
    w[down] = 0.5 * (1.0 + np.cos(np.pi * (t[down] - t2) / (t3 - t2)))
    return w


def _shift_delta(
    term: np.ndarray, time_s: np.ndarray, idx: Sequence[int], shift: int
) -> tuple[np.ndarray, int, int]:
    start, end = idx
    peak = start + int(np.argmax(term[start : end + 1]))
    new = peak + shift
    if not (start < peak < end and start < new < end):
        raise ValueError(
            f"shifted peak index {new} (peak {peak}) must lie strictly inside "
            f"the window ({start}, {end})"
        )
    t = time_s
    knots_new = [t[start], t[new], t[end]]
    knots_old = [t[start], t[peak], t[end]]
    seg = slice(start, end + 1)
    tau = np.interp(t[seg], knots_new, knots_old)
    delta = np.zeros_like(term)
    delta[seg] = np.interp(tau, t[seg], term[seg]) - term[seg]
    return delta, peak, new


def _axis_rotations(axis: str, angle_rad: np.ndarray) -> np.ndarray:
    """(n, 3, 3) elementary rotations about body ``axis`` by ``angle_rad``."""
    return Rotation.from_euler(axis, angle_rad[:, None]).as_matrix()


def _corrections(term_id: str, angle_rad: np.ndarray, ar_rad: np.ndarray) -> np.ndarray:
    if term_id == "x_factor_jcs_axial":
        return _axis_rotations("y", angle_rad)
    if term_id == "trunk_extension_jcs":
        return _axis_rotations("z", angle_rad)
    spin = _axis_rotations("y", ar_rad)
    bend = _axis_rotations("x", angle_rad)
    return np.swapaxes(spin, 1, 2) @ bend @ spin


def _check_gimbal(angles_deg: np.ndarray, lo: int, hi: int) -> None:
    window = angles_deg[lo : hi + 1]
    if np.any(~np.isfinite(window)):
        raise ValueError(
            "gimbal lock (|LB| = 90 deg) inside the variant window: FE and AR are "
            "not separable, the variant is undefined there"
        )


def _input_sha256(rots: np.ndarray, times: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(rots, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(times, dtype="<f8").tobytes())
    return digest.hexdigest()


def _term(term_id: str, rots: np.ndarray, hand: str) -> np.ndarray:
    return evaluate(term_id, {_ROT: rots}, handedness=hand)  # type: ignore[arg-type]


def _delta(
    variant: MovementVariant,
    term: np.ndarray,
    times: np.ndarray,
    idx: tuple[int, ...],
    value: float | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(delta_deg, target_indices, target_values_deg, extra)``."""
    if variant.operation == "shift_peak":
        delta, peak, new = _shift_delta(term, times, idx, int(value))
        extra = {
            "peak_index": peak,
            "shifted_peak_index": new,
            "achieved_shift_s": float(times[new] - times[peak]),
        }
        return delta, np.array([new]), np.array([term[peak]]), extra
    w = _raised_cosine_window(times, idx)
    plateau = np.arange(idx[1], idx[2] + 1)
    if variant.operation == "scale":
        delta = (float(value) - 1.0) * np.where(w > 0.0, term, 0.0) * w
        return delta, plateau, float(value) * term[plateau], {}
    return float(value) * w, plateau, term[plateau] + float(value), {}


# --------------------------------------------------------------------- apply


def apply_variant(
    variant: MovementVariant,
    rotations: np.ndarray,
    time_s: np.ndarray,
    events: Mapping[str, int],
    *,
    handedness: Literal["right", "left"],
    parameter: float | int | None = None,
) -> VariantResult:
    """Apply ``variant`` to the pelvis->thorax series ``rotations`` (n, 3, 3).

    ``events`` maps event names to sample indices (``start``/``end`` are
    reserved pseudo-events for the first/last sample). ``parameter`` defaults
    to ``variant.parameter_default`` and is always recorded.

    Preconditions: ``variant`` is a :class:`MovementVariant` (``TypeError``);
    proper rotations, strictly increasing ``time_s`` of matching length, the
    window events present and ordered, ``handedness`` ``right``/``left``, the
    parameter a real (integer for ``shift_peak``, else ``TypeError``) inside its
    bounds, and no gimbal-locked frame in the window (``ValueError``).

    Postconditions: the declared term equals its target at ``target_indices``
    to 1e-6 deg (checked by re-evaluation, ``ensure``); the other JCS terms are
    unchanged by construction; frames outside the window are bit-identical;
    the result records parameters and a ``movement_variant:`` source.
    """
    if not isinstance(variant, MovementVariant):
        raise TypeError(
            f"variant must be a MovementVariant, got {type(variant).__name__}"
        )
    hand = check_handedness(handedness)
    rots = rotation_series(rotations, "rotations")
    times = _check_time(time_s, rots.shape[0])
    idx = _window_indices(variant, events, rots.shape[0])
    value = _parameter(variant, parameter)
    angles = isb_spine_angles_deg(rots, "rotations")
    _check_gimbal(angles, idx[0], idx[-1])
    term = _term(variant.term_id, rots, hand)
    delta, targets, target_values, extra = _delta(variant, term, times, idx, value)
    out, corr = _compose(variant, rots, angles, delta, (idx, hand))
    varied = _term(variant.term_id, out, hand)
    error = float(np.max(np.abs(varied[targets] - target_values)))
    ensure(error <= 1e-6, "variant must reach its target within 1e-6 deg", error)
    lo, hi = variant.term_bounds_deg
    finite = varied[np.isfinite(varied)]
    prov = _provenance(variant, value, hand, idx, _input_sha256(rots, times))
    prov.update(extra)
    return VariantResult(
        variant=variant,
        rotations=_frozen(out),
        corrections=_frozen(corr),
        delta_deg=_frozen(delta),
        term_values_deg=_frozen(varied),
        target_indices=_frozen(targets, dtype=int),
        target_values_deg=_frozen(target_values),
        target_error_deg=error,
        window=(int(idx[0]), int(idx[-1])),
        term_bounds_ok=bool(np.all((finite >= lo) & (finite <= hi))),
        distribution_source=_source(variant, value, hand),
        provenance=MappingProxyType(prov),
    )


def _compose(
    variant: MovementVariant,
    rots: np.ndarray,
    angles_deg: np.ndarray,
    delta_deg: np.ndarray,
    window: tuple[tuple[int, ...], str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(varied rotations, corrections)``; identity outside the window."""
    idx, hand = window
    corr = np.tile(np.eye(3), (rots.shape[0], 1, 1))
    seg = slice(idx[0] + 1, idx[-1])
    sign = 1.0 if variant.term_id == "trunk_extension_jcs" else handed_sign(hand)
    corr[seg] = _corrections(
        variant.term_id,
        np.radians(sign * delta_deg[seg]),
        np.radians(angles_deg[seg, 2]),
    )
    out = rots.copy()
    if variant.composition_side == "left":
        out[seg] = corr[seg] @ rots[seg]
    else:
        out[seg] = rots[seg] @ corr[seg]
    return out, corr


def _source(variant: MovementVariant, value: float | int, hand: str) -> str:
    return (
        f"movement_variant:{variant.variant_id}@{variant.version};"
        f"{variant.term_id}:{variant.operation}({variant.parameter_name}={value});"
        f"handedness={hand}"
    )


def _provenance(
    variant: MovementVariant,
    value: float | int,
    hand: str,
    idx: tuple[int, ...],
    sha: str,
) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "variant_version": variant.version,
        "operation": variant.operation,
        "term_id": variant.term_id,
        "parameter_name": variant.parameter_name,
        "parameter_units": variant.parameter_units,
        "parameter_value": value,
        "handedness": hand,
        "window_events": variant.window_events,
        "window_indices": idx,
        "non_trunk_policy": variant.non_trunk_policy,
        "input_sha256": sha,
    }


# ------------------------------------------------------------------ sampling


def sample_variant_parameters(
    variant: MovementVariant, *, seed: int, count: int
) -> tuple[float | int, ...]:
    """Draw ``count`` parameters uniformly inside ``variant.parameter_bounds``.

    Deterministic under the explicit ``seed`` (``numpy.random.default_rng``;
    no global state). Integer parameters are drawn with both bounds included.

    Raises:
        TypeError: ``variant`` is not a :class:`MovementVariant`, or ``seed`` /
            ``count`` is not an integer.
        ValueError: ``count < 1`` or ``seed < 0``.
    """
    if not isinstance(variant, MovementVariant):
        raise TypeError("variant must be a MovementVariant")
    rng_seed = _integer(seed, "seed")
    n = _integer(count, "count")
    if n < 1 or rng_seed < 0:
        raise ValueError(f"count must be >= 1 and seed >= 0, got {n}, {rng_seed}")
    rng = default_rng(rng_seed)
    lo, hi = variant.parameter_bounds
    if variant.parameter_integer:
        draws = rng.integers(int(lo), int(hi), size=n, endpoint=True)
        return tuple(int(v) for v in draws)
    return tuple(float(v) for v in rng.uniform(lo, hi, size=n))


# --------------------------------------------------------------- range check


@dataclass(frozen=True)
class RangeViolation:
    """One sample outside its coordinate range (radians or metres)."""

    coordinate: str
    frame: int
    value: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RangeCheck:
    """Result of :func:`check_coordinate_ranges` (reported, never repaired)."""

    ok: bool
    violations: tuple[RangeViolation, ...]
    checked: tuple[str, ...]


def check_coordinate_ranges(
    coordinates: Mapping[str, np.ndarray],
    ranges: Mapping[str, tuple[float, float]],
) -> RangeCheck:
    """Check coordinate series against ``(lower, upper)`` ranges (inclusive).

    Typically ``coordinates`` are the level coordinates a driven variant
    produces and ``ranges`` the target model's coordinate ranges.

    Preconditions: both arguments are mappings (``TypeError``); every
    coordinate has a range with finite ``lower < upper`` and its series is a
    1-D finite array (``ValueError``). Postcondition: ``ok`` is true exactly
    when ``violations`` is empty; violations are ordered by coordinate, frame.
    """
    if not isinstance(coordinates, Mapping) or not isinstance(ranges, Mapping):
        raise TypeError("coordinates and ranges must be mappings")
    violations: list[RangeViolation] = []
    for name in sorted(coordinates):
        if name not in ranges:
            raise ValueError(f"no range given for coordinate {name!r}")
        lo, hi = _pair(ranges[name], f"range of {name!r} (lower, upper)")
        values = np.asarray(coordinates[name], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"coordinate {name!r} must be a finite 1-D series")
        for frame in np.flatnonzero((values < lo) | (values > hi)):
            violations.append(
                RangeViolation(name, int(frame), float(values[frame]), lo, hi)
            )
    result = RangeCheck(not violations, tuple(violations), tuple(sorted(coordinates)))
    ensure(result.ok == (not result.violations), "ok iff no violations")
    return result
