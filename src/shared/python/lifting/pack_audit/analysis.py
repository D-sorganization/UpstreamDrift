"""Reduce a baseline receipt to the tables the report and gap rules share.

All functions are pure (receipt in, plain data out) so they can be tested with
synthetic receipts and re-run without any engine installed.
"""

from __future__ import annotations

from typing import Any

from .names import ENGINES, LIFTS

LIMIT_TOL_RAD = 1e-4  # limit_abs_rad in the parity standard
FLOOR_PULLS = ("deadlift", "snatch", "clean_and_jerk")
PLATE_RADIUS_M = 0.225  # bar centre height on the floor (450 mm plates)
GRIP_TOL_M = 0.01


def engines_of(receipt: dict[str, Any], lift: str) -> list[str]:
    """Engines with a result for *lift*, in canonical order."""
    return [e for e in ENGINES if e in receipt["results"].get(lift, {})]


def lifter_mass(masses: dict[str, float]) -> float:
    """Sum of non-bar, non-bench link masses."""
    return sum(m for n, m in masses.items() if not n.startswith(("barbell_", "bench")))


def mass_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Per lift/engine totals: lifter, bar, bench, and worst segment deviation."""
    ref = receipt["standard_reference"]["segment_mass_kg_at_standard_body_mass"]
    scale = (
        receipt["anthropometry"]["body_mass_kg"]
        / receipt["standard_reference"]["standard_body_mass_kg"]
    )
    rows = []
    for lift in LIFTS:
        for engine in engines_of(receipt, lift):
            masses = receipt["results"][lift][engine]["segment_masses_kg"]
            worst, worst_seg = 0.0, ""
            for seg, want in ref.items():
                if seg in masses:
                    rel = abs(masses[seg] - want * scale) / (want * scale)
                    if rel > worst:
                        worst, worst_seg = rel, seg
            rows.append(
                {
                    "lift": lift,
                    "engine": engine,
                    "lifter_kg": lifter_mass(masses),
                    "bar_kg": sum(
                        v for n, v in masses.items() if n.startswith("barbell_")
                    ),
                    "bench_kg": sum(
                        v for n, v in masses.items() if n.startswith("bench")
                    ),
                    "worst_segment_rel": worst,
                    "worst_segment": worst_seg,
                }
            )
    return rows


def limit_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Coordinates whose canonical-angle limits differ from the standard."""
    from .names import canonical_coordinate_name

    ref = receipt["standard_reference"]["limits_rad"]
    rows = []
    for lift in LIFTS:
        for engine in engines_of(receipt, lift):
            bad, checked = [], 0
            for coord in receipt["results"][lift][engine]["coordinates"]:
                canon = canonical_coordinate_name(engine, coord["name"])
                if canon is None or coord["lower"] is None:
                    continue
                offset = coord.get("zero_offset", 0.0)
                lo, hi = coord["lower"] - offset, coord["upper"] - offset
                want = ref[canon]
                dev = max(abs(lo - want[0]), abs(hi - want[1]))
                checked += 1
                if dev > LIMIT_TOL_RAD:
                    bad.append(
                        {"coordinate": canon, "dev_rad": dev, "lo": lo, "hi": hi}
                    )
            rows.append(
                {
                    "lift": lift,
                    "engine": engine,
                    "checked": checked,
                    "off_standard": bad,
                }
            )
    return rows


def grip_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Intended grip half-width vs achievable hand spacing, plus attachment kind."""
    rows = []
    for lift in LIFTS:
        if lift == "squat":
            continue
        for engine in engines_of(receipt, lift):
            res = receipt["results"][lift][engine]
            zero = res["same_q"]["zero"]["summary"]
            start = res["start"]["summary"]
            intended = abs(zero["hand_bar"]["l"]["lateral_m"])
            actual = zero["grip_width_m"] / 2.0
            attach = res["structure"]["bar_hand_attachment"]
            rows.append(
                {
                    "lift": lift,
                    "engine": engine,
                    "attach_l": attach["l"],
                    "attach_r": attach["r"],
                    "intended_half_width_m": intended,
                    "hand_half_spacing_zero_m": actual,
                    "mismatch_m": intended - actual,
                    "start_hand_spacing_m": start["grip_width_m"],
                    "start_closure_native": res["start"]["closure_native"],
                    "start_assembly_notes": [
                        n for n in res["start"]["notes"] if "assembly" in n
                    ],
                }
            )
    return rows


def start_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Pack-default start pose: bar height above the lowest foot contact."""
    rows = []
    for lift in LIFTS:
        for engine in engines_of(receipt, lift):
            res = receipt["results"][lift][engine]
            start = res["start"]
            rows.append(
                {
                    "lift": lift,
                    "engine": engine,
                    "bar_above_sole_m": start["bar_above_sole_m"],
                    "sole_z_m": start["sole_z_m"],
                    "bar_rel_feet_m": start["summary"]["bar_centre_rel_feet_m"],
                    "contact_n": res["start_contact"].get("value_n"),
                    "contact_reason": res["start_contact"].get("reason"),
                    "non_ground_n": res["start_contact"].get(
                        "non_ground_normal_force_n"
                    ),
                }
            )
    return rows


def phase_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Per lift/engine: phase count vs standard, symmetry and mapping defects."""
    want = receipt["standard_reference"]["phase_count"]
    rows = []
    for lift in LIFTS:
        for engine in engines_of(receipt, lift):
            phases = receipt["results"][lift][engine]["phases"]
            limits = {
                c["name"]: (c["lower"], c["upper"], c.get("zero_offset", 0.0))
                for c in receipt["results"][lift][engine]["coordinates"]
            }
            rows.append(
                {
                    "lift": lift,
                    "engine": engine,
                    "count": len(phases),
                    "standard_count": want.get(lift),
                    "names": [p["name"] for p in phases],
                    "unsymmetric": [p["name"] for p in phases if not p["bilateral"]],
                    "unmapped": sorted({k for p in phases for k in p["unmapped_keys"]}),
                    "interpreted": sorted(
                        {k for p in phases for k in p["interpreted_keys"]}
                    ),
                    "out_of_range": _out_of_range(engine, phases, limits),
                }
            )
    return rows


def _out_of_range(
    engine: str,
    phases: list[dict[str, Any]],
    limits: dict[str, tuple[float, float, float]],
) -> list[str]:
    from .names import engine_coordinate_name

    out = []
    for phase in phases:
        for canon, angle in phase["canonical_targets"].items():
            native = engine_coordinate_name(engine, canon)
            if native not in limits or limits[native][0] is None:
                continue
            lo, hi, off = limits[native]
            if angle + off < lo - 1e-9 or angle + off > hi + 1e-9:
                out.append(f"{phase['name']}:{canon}")
    return out


def phase_hand_bar_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Worst hand-to-bar-axis distance along each pack's own phases."""
    rows = []
    for lift in LIFTS:
        for engine in engines_of(receipt, lift):
            worst = 0.0
            where = ""
            for phase in receipt["results"][lift][engine]["phases"]:
                for side in ("l", "r"):
                    dist = phase["summary"]["hand_bar"][side]["axis_distance_m"]
                    if dist > worst:
                        worst, where = dist, f"{phase['name']}/{side}"
            rows.append({"lift": lift, "engine": engine, "worst_m": worst, "at": where})
    return rows


def parity_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Worst pairwise same-q difference per lift, over every pose and engine pair."""
    keys = (
        "segments_max_m",
        "hands_max_m",
        "feet_max_m",
        "bar_centre_max_m",
        "lifter_com_max_m",
    )
    rows = []
    for lift, comp in receipt["comparisons"].items():
        worst: dict[str, tuple[float, str]] = dict.fromkeys(keys, (0.0, ""))
        for pose, pairs in comp["poses"].items():
            for pair, metrics in pairs.items():
                for key in keys:
                    val = metrics.get(key)
                    if val is not None and val > worst[key][0]:
                        worst[key] = (val, f"{pair} @ {pose}")
        rows.append(
            {"lift": lift, "worst": worst, "mass_spread_kg": comp["mass"]["spread_kg"]}
        )
    return rows


def reference_fk_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Worst error of each engine's FK against the standard's reference FK."""
    rows = []
    for lift in LIFTS:
        for engine in engines_of(receipt, lift):
            fk = receipt["results"][lift][engine]["reference_fk"]
            if not fk:
                continue
            worst = max(fk, key=lambda f: f["max_abs_m"])
            rows.append(
                {
                    "lift": lift,
                    "engine": engine,
                    "max_abs_m": worst["max_abs_m"],
                    "pose": worst["pose"],
                    "segment": worst["worst_segment"],
                }
            )
    return rows
