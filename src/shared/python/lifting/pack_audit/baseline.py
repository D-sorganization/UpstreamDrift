"""Run the lift-pack audit: inventory, smoke, same-input parity and phases."""

from __future__ import annotations

import logging
import math
import platform
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from . import geometry, phases, poses
from .adapters import create_adapter
from .model import Anthropometry, EngineAdapter, PoseEval
from .names import ENGINES, LIFTS, SIDES
from .packs import PackLocation, locate_pack

SCHEMA = "lift-pack-parity-baseline/v1"
POSITION_TOL_M = 0.02  # cross_engine_position_abs_m in the parity standard
MASS_REL_TOL = 1e-6  # mass_rel in the parity standard

_NON_SEGMENT = ("barbell_", "bench", "ground")
_REFERENCE_POSES = ("zero", "flexion", "frontal", "axial")


def _is_segment(name: str) -> bool:
    return not name.startswith(_NON_SEGMENT)


def _listify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _listify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def _rel_to_pelvis(ev: PoseEval) -> dict[str, np.ndarray]:
    root = ev.positions["pelvis"]
    return {n: p - root for n, p in ev.positions.items()}


def body_com_rel_pelvis(ev: PoseEval, masses: dict[str, float]) -> list[float] | None:
    """Lifter-only CoM (bar and bench removed) relative to the pelvis.

    Uses each excluded link's origin as its centre of mass, which holds for
    every pack's bar and bench (zero mass-centre offset).  ``None`` when no
    lifter mass remains.
    """
    removed = [n for n in ev.positions if n.startswith(("barbell_", "bench"))]
    m_removed = sum(masses.get(n, 0.0) for n in removed)
    m_body = ev.total_mass - m_removed
    if m_body <= 0:
        return None
    moment = ev.total_mass * ev.com - sum(
        masses.get(n, 0.0) * ev.positions[n] for n in removed
    )
    return ((moment / m_body) - ev.positions["pelvis"]).tolist()


def _above(ev: PoseEval, body: str) -> float | None:
    if ev.sole_z is None or body not in ev.positions:
        return None
    return float(ev.positions[body][2] - ev.sole_z)


def _above_mid(ev: PoseEval) -> float | None:
    if ev.sole_z is None:
        return None
    mid = 0.5 * (ev.positions["hand_l"][2] + ev.positions["hand_r"][2])
    return float(mid - ev.sole_z)


def _eval_record(ev: PoseEval, masses: dict[str, float]) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "lifter_com_rel_pelvis_m": body_com_rel_pelvis(ev, masses),
        "summary": geometry.pose_summary(ev.positions),
        "com_rel_pelvis_m": (ev.com - ev.positions["pelvis"]).tolist(),
        "total_mass_kg": ev.total_mass,
        "sole_z_m": ev.sole_z,
        "bar_above_sole_m": _above(ev, "barbell_shaft"),
        "hand_mid_above_sole_m": _above_mid(ev),
        "closure_native": ev.closure,
        "notes": ev.notes,
    }
    return _listify(rec)


def _reference_error(
    pack: PackLocation,
    std: dict[str, Any],
    adapter: EngineAdapter,
    name: str,
    q: dict[str, float],
) -> dict[str, Any]:
    ev = adapter.evaluate(q)
    ref = poses.reference_origins(pack, std, q, adapter.anthro.height_m)
    rel = _rel_to_pelvis(ev)
    common = sorted(set(ref) & set(rel))
    worst, worst_name = 0.0, ""
    for seg in common:
        gap = float(np.max(np.abs(np.asarray(ref[seg]) - rel[seg])))
        if gap > worst:
            worst, worst_name = gap, seg
    return {
        "pose": name,
        "max_abs_m": worst,
        "worst_segment": worst_name,
        "n_compared": len(common),
        "missing_in_model": sorted(set(ref) - set(rel)),
    }


def _phase_records(
    adapter: EngineAdapter, pack: PackLocation, coords: set[str]
) -> list[dict[str, Any]]:
    out = []
    for ph in phases.pack_phases(pack, adapter.lift):
        ev = adapter.evaluate(ph.angles)
        summary = geometry.pose_summary(ev.positions)
        out.append(
            _listify(
                {
                    "name": ph.name,
                    "fraction": ph.fraction,
                    "n_targets": len(ph.raw),
                    "unmapped_keys": list(ph.unmapped_keys),
                    "interpreted_keys": list(ph.interpreted_keys),
                    "bilateral": all(
                        (n.replace("_l_", "_r_") in ph.angles)
                        for n in ph.angles
                        if "_l_" in n
                    ),
                    "summary": summary,
                    "closure_native": ev.closure,
                    "targets": ph.raw,
                    "canonical_targets": ph.angles,
                }
            )
        )
    return out


def audit_pack_lift(
    pack: PackLocation, lift: str, anthro: Anthropometry, pose_sets: dict[str, dict]
) -> dict[str, Any]:
    """Everything the baseline records for one pack and lift."""
    adapter = create_adapter(pack, lift, anthro)
    coords = adapter.coordinates()
    structure = adapter.structure()
    start = adapter.evaluate(None)
    masses = adapter.segment_masses()
    std = poses.load_standard(pack)
    ref_poses = {
        "zero": {},
        **{k: v for k, v in pose_sets.items() if k in _REFERENCE_POSES},
    }
    result: dict[str, Any] = {
        "model_chars": len(adapter.model_text()),
        "structure": structure,
        "coordinates": [c.__dict__ for c in coords],
        "segment_masses_kg": masses,
        "start": {
            **_eval_record(start, masses),
            "pelvis_world_m": start.positions["pelvis"].tolist(),
            "foot_l_world_m": start.positions["foot_l"].tolist(),
            "bar_world_m": start.positions["barbell_shaft"].tolist(),
        },
        "start_contact": adapter.start_contact_force(),
        "smoke": adapter.smoke_step(),
        "same_q": {},
        "reference_fk": [],
    }
    store: dict[str, PoseEval] = {}
    for name, q in pose_sets.items():
        store[name] = adapter.evaluate(q)
        result["same_q"][name] = _eval_record(store[name], masses)
        result["same_q"][name]["rel_pelvis_m"] = {
            k: v.tolist() for k, v in _rel_to_pelvis(store[name]).items()
        }
    if lift != "bench_press":
        for name, q in ref_poses.items():
            result["reference_fk"].append(_reference_error(pack, std, adapter, name, q))
    result["phases"] = _phase_records(adapter, pack, {c.name for c in coords})
    return result


def _pair_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    ra, rb = a["rel_pelvis_m"], b["rel_pelvis_m"]
    seg = [n for n in ra if n in rb and _is_segment(n)]

    def worst(names: Iterable[str]) -> float | None:
        names = [n for n in names if n in ra and n in rb]
        if not names:
            return None
        return max(float(np.max(np.abs(np.subtract(ra[n], rb[n])))) for n in names)

    com = float(
        np.max(np.abs(np.subtract(a["com_rel_pelvis_m"], b["com_rel_pelvis_m"])))
    )
    lifter_a, lifter_b = (
        a.get("lifter_com_rel_pelvis_m"),
        b.get("lifter_com_rel_pelvis_m"),
    )
    lifter = (
        float(np.max(np.abs(np.subtract(lifter_a, lifter_b))))
        if lifter_a is not None and lifter_b is not None
        else None
    )
    return {
        "segments_max_m": worst(seg),
        "hands_max_m": worst(["hand_l", "hand_r"]),
        "feet_max_m": worst(["foot_l", "foot_r"]),
        "bar_centre_max_m": worst(["barbell_shaft"]),
        "com_max_m": com,
        "lifter_com_max_m": lifter,
    }


def compare_engines(per_engine: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pairwise same-q differences (pelvis-relative) and mass differences."""
    engines = [e for e in ENGINES if e in per_engine]
    out: dict[str, Any] = {"poses": {}, "mass": {}}
    pose_names = list(next(iter(per_engine.values()))["same_q"]) if engines else []
    for pose in pose_names:
        pairs = {}
        for i, e1 in enumerate(engines):
            for e2 in engines[i + 1 :]:
                pairs[f"{e1}|{e2}"] = _pair_metrics(
                    per_engine[e1]["same_q"][pose], per_engine[e2]["same_q"][pose]
                )
        out["poses"][pose] = pairs
    masses = {e: per_engine[e]["structure"]["total_mass"] for e in engines}
    out["mass"]["total_kg"] = masses
    lo, hi = min(masses.values()), max(masses.values())
    out["mass"]["spread_kg"] = hi - lo
    return out


def standard_reference(std: dict[str, Any]) -> dict[str, Any]:
    """Reference values from the parity standard the packs claim to follow."""
    limits = {}
    for entry in std["coordinates"]:
        lo, hi = (math.radians(v) for v in entry["limits_deg"])
        for side in SIDES:
            limits[entry["name"].format(side=side)] = [lo, hi]
    anthro = std["anthropometrics"]
    masses = {}
    for seg, spec in anthro["segments"].items():
        mass = spec["mass_frac"] * anthro["body_mass_kg"]
        if spec["bilateral"]:
            for side in SIDES:
                masses[f"{seg}_{side}"] = mass
        else:
            masses[seg] = mass
    phase_counts = {
        lift: int(spec["phase_count"])
        for lift, spec in std["exercises"].items()
        if lift in LIFTS
    }
    return {
        "limits_rad": limits,
        "segment_mass_kg_at_standard_body_mass": masses,
        "standard_body_mass_kg": anthro["body_mass_kg"],
        "phase_count": phase_counts,
        "root": std["root"],
        "barbell": std["barbell"]["mens"],
    }


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod, attr in (("mujoco", "__version__"), ("pinocchio", "__version__")):
        try:
            out[mod] = str(getattr(__import__(mod), attr))
        except ImportError:
            out[mod] = "unavailable"
    try:
        import opensim

        out["opensim"] = str(opensim.GetVersion())
    except (ImportError, AttributeError):
        out["opensim"] = "unavailable"
    try:
        from importlib.metadata import version

        out["pydrake"] = version("drake")
    except ImportError:
        out["pydrake"] = "unavailable"
    return out


def run_baseline(
    anthro: Anthropometry | None = None,
    engines: Iterable[str] = ENGINES,
    lifts: Iterable[str] = LIFTS,
    extra_roots: list | None = None,
) -> dict[str, Any]:
    """Run the full audit and return the JSON-ready receipt (no gaps yet).

    Packs or engines that are not available are recorded under ``deferred``
    with the reason; nothing is substituted for them.
    """
    anthro = anthro or Anthropometry()
    for name in ("drake",):
        logging.getLogger(name).setLevel(logging.ERROR)
    located: dict[str, PackLocation] = {}
    deferred: list[dict[str, str]] = []
    for engine in engines:
        pack = locate_pack(engine, extra_roots)
        if pack is None:
            deferred.append({"engine": engine, "reason": "pack checkout not found"})
        else:
            located[engine] = pack
    if not located:
        raise RuntimeError("no model pack checkout found; clone the *_Models repos")
    first = next(iter(located.values()))
    std = poses.load_standard(first)
    pose_sets: dict[str, dict[str, float]] = {
        "zero": {},
        **poses.standard_poses(first, std),
        **poses.lift_shape_poses(),
    }
    results: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Any] = {}
    for lift in lifts:
        per_engine: dict[str, dict[str, Any]] = {}
        for engine, pack in located.items():
            try:
                per_engine[engine] = audit_pack_lift(pack, lift, anthro, pose_sets)
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                deferred.append(
                    {
                        "engine": engine,
                        "lift": lift,
                        "reason": f"{type(exc).__name__}: {exc}"[:400],
                    }
                )
        results[lift] = per_engine
        if len(per_engine) >= 2:
            comparisons[lift] = compare_engines(per_engine)
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "anthropometry": anthro.__dict__
        | {"bar_total_mass_kg": anthro.bar_total_mass_kg},
        "tolerances": {"position_m": POSITION_TOL_M, "mass_rel": MASS_REL_TOL},
        "standard": {
            "schema": std["schema"],
            "version": std["version"],
            "sha256": std.get("_sha256"),
        },
        "packs": {
            e: {"repo": p.repo, "commit": p.commit, "licence": p.licence}
            for e, p in located.items()
        },
        "environment": _versions(),
        "standard_reference": standard_reference(std),
        "pose_sets": {k: _listify(v) for k, v in pose_sets.items()},
        "results": results,
        "comparisons": comparisons,
        "deferred": deferred,
    }
