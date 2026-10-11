"""Derive the cross-pack discrepancy list from a baseline receipt.

Each gap is computed from measured rows, never hand-entered, and is mapped to
an existing per-repo issue or to the one filed by this audit. ``lift_story``
names the LIFT-n child of epic #11740 expected to close it.
"""

from __future__ import annotations

from typing import Any

from . import analysis as an

REPO = {
    "mujoco": "MuJoCo_Models",
    "opensim": "OpenSim_Models",
    "drake": "Drake_Models",
    "pinocchio": "Pinocchio_Models",
}

# engine -> issue numbers, per defect class (existing unless marked new).
ISSUES: dict[str, dict[str, list[int]]] = {
    "grip_attachment": {"drake": [365], "pinocchio": [443]},
    "grip_width": {"mujoco": [408], "opensim": [394], "pinocchio": [443]},
    "start_pose": {
        "mujoco": [407],
        "drake": [381, 364],
        "opensim": [416],
        "pinocchio": [450],
    },
    "limits": {"mujoco": [407, 409], "pinocchio": [433]},
    "phase_count": {"mujoco": [401], "drake": [372], "pinocchio": [434]},
    "phase_asymmetric": {"pinocchio": [449]},
    "phase_keys": {"opensim": [414], "drake": [390], "pinocchio": [436]},
    "bench_mass": {
        "drake": [391],
        "mujoco": [428],
        "opensim": [415],
        "pinocchio": [451],
    },
    "contacts": {"mujoco": [427], "opensim": [384]},
    "inertia": {"drake": [391], "opensim": [392], "mujoco": [392]},
    "no_grf": {"drake": [371], "pinocchio": [432]},
}

EXISTING_ONLY = {"grip_attachment", "grip_width", "limits", "phase_count", "no_grf"}


def _gap(
    key: str, title: str, evidence: list[str], engines: list[str], story: str
) -> dict[str, Any]:
    issues = [f"{REPO[e]}#{n}" for e in engines for n in ISSUES.get(key, {}).get(e, [])]
    return {
        "key": key,
        "title": title,
        "engines": engines,
        "evidence": evidence,
        "issues": issues,
        "lift_story": story,
        "new_issue": key not in EXISTING_ONLY,
    }


def derive_gaps(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every measured discrepancy, each mapped to its tracking issues."""
    gaps: list[dict[str, Any]] = []
    gaps += _grip(receipt)
    gaps += _start(receipt)
    gaps += _limits(receipt)
    gaps += _phases(receipt)
    gaps += _mass_contact(receipt)
    gaps += _inertia(receipt)
    return [g for g in gaps if g["engines"]]


def _grip(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = an.grip_rows(receipt)
    no_r = sorted({r["engine"] for r in rows if _unattached(r["attach_r"])})
    wide = sorted({r["engine"] for r in rows if abs(r["mismatch_m"]) > an.GRIP_TOL_M})
    closed = sorted(
        {
            r["engine"]
            for r in rows
            if _closure(r["start_closure_native"]) > an.GRIP_TOL_M
        }
    )
    worst = max(rows, key=lambda r: abs(r["mismatch_m"]), default=None)
    return [
        _gap(
            "grip_attachment",
            "Right hand is not attached to the bar",
            [f"{e}: right-hand attachment is not a bar weld" for e in no_r],
            no_r,
            "LIFT-3",
        ),
        _gap(
            "grip_width",
            "Grip half-width does not match achievable hand spacing",
            [
                f"worst {worst['engine']}/{worst['lift']}: "
                f"{abs(worst['mismatch_m']):.3f} m"
                if worst
                else "none"
            ]
            + [f"{e}: native start closure above {an.GRIP_TOL_M} m" for e in closed],
            sorted(set(wide) | set(closed)),
            "LIFT-3",
        ),
    ]


def _closure(value: Any) -> float:
    """Largest closure residual in a native-closure record (metres)."""
    if not isinstance(value, dict):
        return float(value or 0.0)
    return max(
        (float(v) for k, v in value.items() if not k.startswith("n_")), default=0.0
    )


def _unattached(value: Any) -> bool:
    text = str(value).lower()
    return text.startswith("none") or text in {"null", ""} or "virtual" in text


def _start(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        r
        for r in an.start_rows(receipt)
        if r["lift"] in an.FLOOR_PULLS
        and abs(r["bar_above_sole_m"] - an.PLATE_RADIUS_M) > 0.05
    ]
    engines = sorted({r["engine"] for r in rows})
    ev = [
        f"{r['engine']}/{r['lift']}: bar centre {r['bar_above_sole_m']:.3f} m above "
        f"sole (floor target {an.PLATE_RADIUS_M} m)"
        for r in rows
    ]
    return [
        _gap(
            "start_pose",
            "Floor-pull start pose is not at the floor",
            ev,
            engines,
            "LIFT-4",
        )
    ]


def _limits(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in an.limit_rows(receipt) if r["off_standard"]]
    engines = sorted({r["engine"] for r in rows})
    ev = [
        f"{r['engine']}/{r['lift']}: {len(r['off_standard'])} of {r['checked']} "
        f"limits off standard (max {max(b['dev_rad'] for b in r['off_standard']):.3f} rad)"
        for r in rows
    ]
    return [
        _gap(
            "limits",
            "Joint limits differ from the parity standard",
            ev,
            engines,
            "LIFT-5",
        )
    ]


def _phases(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = an.phase_rows(receipt)
    short = [
        r for r in rows if r["standard_count"] and r["count"] != r["standard_count"]
    ]
    asym = [r for r in rows if r["unsymmetric"]]
    keys = [r for r in rows if r["unmapped"] or r["interpreted"]]
    return [
        _gap(
            "phase_count",
            "Phase count differs from the standard",
            [
                f"{r['engine']}/{r['lift']}: {r['count']} vs {r['standard_count']}"
                for r in short
            ],
            sorted({r["engine"] for r in short}),
            "LIFT-6",
        ),
        _gap(
            "phase_asymmetric",
            "Phase targets are left-side only",
            [
                f"{r['engine']}/{r['lift']}: {len(r['unsymmetric'])} phases"
                for r in asym
            ],
            sorted({r["engine"] for r in asym}),
            "LIFT-6",
        ),
        _gap(
            "phase_keys",
            "Phase keys do not resolve to model coordinates",
            [
                f"{r['engine']}/{r['lift']}: unmapped {r['unmapped']}, "
                f"inferred {r['interpreted'][:3]}"
                for r in keys
            ],
            sorted({r["engine"] for r in keys}),
            "LIFT-6",
        ),
    ]


def _mass_contact(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    bench = [r for r in an.mass_rows(receipt) if r["lift"] == "bench_press"]
    vals = {r["engine"]: r["bench_kg"] for r in bench}
    spread = max(vals.values(), default=0.0) - min(vals.values(), default=0.0)
    out = [
        _gap(
            "bench_mass",
            "Bench mass convention differs across packs",
            [f"{e}: {v:.4g} kg" for e, v in vals.items()] if spread > 1e-3 else [],
            sorted(vals) if spread > 1e-3 else [],
            "LIFT-2",
        )
    ]
    contact = [r for r in an.start_rows(receipt) if (r["non_ground_n"] or 0) > 1.0]
    big = [
        r
        for r in an.start_rows(receipt)
        if r["lift"] == "bench_press" and (r["contact_n"] or 0) > 5000
    ]
    out.append(
        _gap(
            "contacts",
            "Start pose carries unphysical contact force",
            [
                f"{r['engine']}/{r['lift']}: non-ground {r['non_ground_n']:.0f} N"
                for r in contact
            ]
            + [
                f"{r['engine']}/bench_press: ground {r['contact_n']:.0f} N" for r in big
            ],
            sorted({r["engine"] for r in contact} | {r["engine"] for r in big}),
            "LIFT-4",
        )
    )
    none = sorted(
        {r["engine"] for r in an.start_rows(receipt) if r["contact_n"] is None}
    )
    out.append(
        _gap(
            "no_grf",
            "No ground-reaction output available",
            [f"{e}: {receipt_reason(receipt, e)}" for e in none],
            none,
            "LIFT-9",
        )
    )
    return out


def receipt_reason(receipt: dict[str, Any], engine: str) -> str:
    """First recorded reason a contact value is unavailable for *engine*."""
    for lift_res in receipt["results"].values():
        res = lift_res.get(engine)
        if res and res["start_contact"].get("reason"):
            return str(res["start_contact"]["reason"])
    return "unavailable"


def _inertia(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Segments whose principal inertia differs >25 % between any two packs."""
    ev: list[str] = []
    outliers: set[str] = set()
    res = receipt["results"].get("deadlift", {})
    segs = ["torso", "thigh_l", "shank_l", "upperarm_l"]
    for seg in segs:
        vals = {
            e: sorted(r["structure"]["inertia_diag"][seg])
            for e, r in res.items()
            if seg in r["structure"].get("inertia_diag", {})
        }
        if len(vals) < 2:
            continue
        mid = {e: v[1] for e, v in vals.items()}
        ref = sorted(mid.values())[len(mid) // 2]
        for eng, val in mid.items():
            if ref > 0 and abs(val - ref) / ref > 0.25:
                outliers.add(eng)
                ev.append(
                    f"{eng}/{seg}: median principal moment {val:.3f} vs {ref:.3f}"
                )
    return [
        _gap(
            "inertia",
            "Segment inertia differs from the other packs",
            ev,
            sorted(outliers),
            "LIFT-2",
        )
    ]
