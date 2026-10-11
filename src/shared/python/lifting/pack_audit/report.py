"""Render the baseline receipt as the PACK_PARITY_BASELINE Markdown document."""

from __future__ import annotations

import copy
from typing import Any

from . import analysis as an
from .gaps import derive_gaps
from .names import ENGINES, LIFTS

TITLES = {
    "squat": "Back Squat",
    "deadlift": "Deadlift",
    "bench_press": "Bench Press",
    "snatch": "Snatch",
    "clean_and_jerk": "Clean and Jerk",
}
ENGINE_TITLES = {
    "mujoco": "MuJoCo",
    "opensim": "OpenSim",
    "drake": "Drake",
    "pinocchio": "Pinocchio",
}

# Items this audit cannot execute; recorded so nothing is silently skipped.
STANDING_DEFERRALS = [
    {
        "item": "Simscape and MATLAB lift models",
        "reason": "No lift models exist in any repository (epic #11740: Simscape "
        "lift models are missing everywhere); MATLAB R2025b is required if they "
        "are added, and nothing was substituted.",
    },
    {
        "item": "Ground-reaction force at the start pose in Drake and Pinocchio",
        "reason": "Neither pack exposes a contact model (Drake_Models#371, "
        "Pinocchio_Models#432); reported as unavailable, never zero.",
    },
    {
        "item": "MuJoCo phase out-of-range reading",
        "reason": "Pack phase values may be absolute qpos or ref-relative angles; "
        "the audit reads them as canonical angles, so out-of-range flags for "
        "MuJoCo phases are interpretation dependent and are not used as gaps.",
    },
    {
        "item": "Native renders in OpenSim, Drake and Pinocchio",
        "reason": "The packs have no offscreen renderer (MuJoCo_Models#412, "
        "Drake_Models#375, Pinocchio_Models#438); stills are skeleton plots of "
        "each engine's own FK, not native renders.",
    },
    {
        "item": "Physical validation against measured lifts",
        "reason": "No lift mocap or force data exist (epic #11740); this baseline "
        "records software parity only, not scientific qualification.",
    },
]

EPIC_DEFECTS = [
    (
        "OpenSim_Models#394, MuJoCo_Models#408, Pinocchio_Models#443, Drake_Models#365",
        "Barbell grip welds",
        "Confirmed. Right hand unattached in Drake and Pinocchio; MuJoCo has "
        "conflicting welds (0.049 m start closure, deadlift); OpenSim closes only "
        "after assembly moves coordinates.",
    ),
    (
        "Drake_Models#381, #364, MuJoCo_Models#407",
        "Start poses",
        "Confirmed in all four packs, not only Drake (see Start Poses).",
    ),
    (
        "OpenSim_Models#384",
        "Bench default pose contact",
        "Confirmed: 21.7 kN vertical contact force at the bench start pose.",
    ),
    (
        "MuJoCo_Models#409",
        "Bench shoulder ref outside range",
        "Not reproduced at this commit: the reported bench shoulder_flex default "
        "(1.571 rad) lies inside its range [-1.047, 3.142]; recheck against the "
        "issue's ref value.",
    ),
    (
        "MuJoCo_Models#403, #404",
        "IK and controller API",
        "Not exercised by this audit (capability gaps, not parity).",
    ),
    (
        "Repository_Management#2024, #2025",
        "Root joint and named phases",
        "Consistent with measurements: MuJoCo supine via quaternion; phase keys "
        "inconsistent across packs.",
    ),
    (
        "MuJoCo_Models#401, Pinocchio_Models#434",
        "Deadlift phase count",
        "Confirmed (4 vs 5). Drake_Models#372 also holds for all lifts.",
    ),
    (
        "OpenSim_Models#380, MuJoCo_Models#392, Drake_Models#361, Pinocchio_Models#425",
        "Parity roadmaps",
        "Not re-measured.",
    ),
]


def condense(receipt: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky per-pose positions so the committed receipt stays small."""
    out = copy.deepcopy(receipt)
    for lift_res in out["results"].values():
        for res in lift_res.values():
            res.pop("model_chars", None)
            res["same_q"] = {
                name: {
                    "summary": pose["summary"],
                    "closure_native": pose["closure_native"],
                }
                for name, pose in res["same_q"].items()
                if name == "zero"
            }
            for ph in res["phases"]:
                ph.pop("targets", None)
                ph["summary"] = {"hand_bar": ph["summary"]["hand_bar"]}
    return out


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return "-" if v is None else str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _closure_text(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{k} {v:.3g}" for k, v in value.items() if not k.startswith("n_")
        )
    return "-" if value is None else f"{value:.3g}"


def _e(engine: str) -> str:
    return ENGINE_TITLES[engine]


def _setup(receipt: dict[str, Any]) -> str:
    a = receipt["anthropometry"]
    env = receipt["environment"]
    pack_rows = [
        [_e(e), p["repo"], p["commit"][:12], p["licence"]]
        for e, p in receipt["packs"].items()
    ]
    return "\n\n".join(
        [
            f"Reference lifter {a['body_mass_kg']} kg, {a['height_m']} m; "
            f"competition bar (20 kg) plus {a['plate_mass_per_side_kg']} kg per side, "
            f"{a['bar_total_mass_kg']} kg on the bar. Parity standard "
            f"{receipt['standard']['version']} (canonical frame +Z up, +X forward, "
            f"+Y left; cross-engine position tolerance "
            f"{receipt['tolerances']['position_m']} m, mass "
            f"{receipt['tolerances']['mass_rel']} relative).",
            _table(["Engine", "Repository", "Commit", "Licence"], pack_rows),
            "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()) + ".",
        ]
    )


def _inventory(receipt: dict[str, Any]) -> str:
    rows = []
    for lift in LIFTS:
        for e in an.engines_of(receipt, lift):
            res = receipt["results"][lift][e]
            st, sm = res["structure"], res["smoke"]
            att = st["bar_hand_attachment"]
            rows.append(
                [
                    TITLES[lift],
                    _e(e),
                    f"{st.get('nq')}/{st.get('nv')}" if "nq" in st else st.get("n_dof"),
                    st.get("n_bodies"),
                    st.get("root_joint"),
                    st.get("bar_root"),
                    f"L {att['l']}; R {att['r']}",
                    "yes" if sm.get("loaded") and sm.get("stepped") else "NO",
                    len(res["phases"]),
                ]
            )
    head = [
        "Lift",
        "Engine",
        "nq/nv",
        "Bodies",
        "Root",
        "Bar",
        "Hand Attachment",
        "Load+Step",
        "Phases",
    ]
    names = []
    for lift in LIFTS:
        for e in an.engines_of(receipt, lift):
            ph = [p["name"] for p in receipt["results"][lift][e]["phases"]]
            names.append([TITLES[lift], _e(e), ", ".join(ph)])
    return "\n\n".join(
        [
            _table(head, rows),
            "Phase names per pack (start pose is the first phase; catch and "
            "rack poses are the pack's own named phases):",
            _table(["Lift", "Engine", "Phases"], names),
        ]
    )


def _masses(receipt: dict[str, Any]) -> str:
    rows = [
        [
            TITLES[r["lift"]],
            _e(r["engine"]),
            r["lifter_kg"],
            r["bar_kg"],
            r["bench_kg"],
            r["worst_segment_rel"],
            r["worst_segment"],
        ]
        for r in an.mass_rows(receipt)
    ]
    return _table(
        [
            "Lift",
            "Engine",
            "Lifter kg",
            "Bar kg",
            "Bench kg",
            "Worst Segment Rel. Dev.",
            "Segment",
        ],
        rows,
    )


def _parity(receipt: dict[str, Any]) -> str:
    rows = []
    for r in an.parity_rows(receipt):
        w = r["worst"]
        rows.append(
            [TITLES[r["lift"]]]
            + [
                f"{w[k][0]:.4f} ({w[k][1].replace('|', ' vs ')})"
                for k in (
                    "segments_max_m",
                    "hands_max_m",
                    "feet_max_m",
                    "bar_centre_max_m",
                    "lifter_com_max_m",
                )
            ]
            + [r["mass_spread_kg"]]
        )
    fk = [
        [TITLES[r["lift"]], _e(r["engine"]), r["max_abs_m"], r["segment"]]
        for r in an.reference_fk_rows(receipt)
    ]
    worst_fk: dict[str, tuple[float, str]] = {}
    for row in fk:
        worst_fk[row[1]] = max(worst_fk.get(row[1], (0.0, "")), (row[2], row[3]))
    return "\n\n".join(
        [
            "Worst pairwise difference (metres) over all engine pairs and all "
            "same-q poses (position expressed relative to the pelvis; "
            "tolerance 0.02 m). Bar-centre differences come from the packs' "
            "different grip offsets, not from FK.",
            _table(
                [
                    "Lift",
                    "Segments",
                    "Hands",
                    "Feet",
                    "Bar Centre",
                    "Lifter CoM",
                    "Mass Spread kg",
                ],
                rows,
            ),
            "Each engine's FK against the standard's reference FK (all lifts, worst "
            "pose): "
            + "; ".join(f"{e} {v[0]:.2e} m ({v[1]})" for e, v in worst_fk.items())
            + ".",
        ]
    )


def _grips(receipt: dict[str, Any]) -> str:
    rows = [
        [
            TITLES[r["lift"]],
            _e(r["engine"]),
            f"L {r['attach_l']}; R {r['attach_r']}",
            r["intended_half_width_m"],
            r["hand_half_spacing_zero_m"],
            r["mismatch_m"],
            _closure_text(r["start_closure_native"]),
        ]
        for r in an.grip_rows(receipt)
    ]
    return (
        "Closure is the native hand-to-bar residual at the pack start pose (m; OpenSim values are after model assembly, which moves coordinates off their defaults).\n\n"
        + _table(
            [
                "Lift",
                "Engine",
                "Attachment",
                "Intended Half-Width m",
                "Hand Half-Spacing m",
                "Mismatch m",
                "Native Start Closure",
            ],
            rows,
        )
    )


def _starts(receipt: dict[str, Any]) -> str:
    rows = [
        [
            TITLES[r["lift"]],
            _e(r["engine"]),
            r["bar_above_sole_m"],
            r["sole_z_m"],
            r["contact_n"] if r["contact_n"] is not None else "unavailable",
            r["non_ground_n"],
        ]
        for r in an.start_rows(receipt)
    ]
    return _table(
        [
            "Lift",
            "Engine",
            "Bar Centre Above Sole m",
            "Sole z m",
            "Ground Contact N",
            "Non-Ground Contact N",
        ],
        rows,
    )


def _limits(receipt: dict[str, Any]) -> str:
    rows = [
        [
            TITLES[r["lift"]],
            _e(r["engine"]),
            r["checked"],
            len(r["off_standard"]),
            ", ".join(
                f"{b['coordinate']} ({b['dev_rad']:.3f})" for b in r["off_standard"][:6]
            ),
        ]
        for r in an.limit_rows(receipt)
    ]
    return _table(
        ["Lift", "Engine", "Checked", "Off Standard", "Coordinates (rad)"], rows
    )


def _phases(receipt: dict[str, Any]) -> str:
    note = (
        "Hand-bar is the perpendicular distance from a hand origin to the bar axis, "
        "worst over the pack's own phases; a squat bar rides on the back, so about "
        "0.59 m is expected there. Inferred keys are phase keys that had to be "
        "interpreted (side or coordinate guessed) to apply them."
    )
    hb = {(r["lift"], r["engine"]): r for r in an.phase_hand_bar_rows(receipt)}
    rows = []
    for r in an.phase_rows(receipt):
        w = hb[(r["lift"], r["engine"])]
        rows.append(
            [
                TITLES[r["lift"]],
                _e(r["engine"]),
                f"{r['count']} / {r['standard_count']}",
                len(r["unsymmetric"]),
                ", ".join(r["unmapped"]) or "-",
                len(r["interpreted"]),
                f"{w['worst_m']:.3f} ({w['at']})",
            ]
        )
    return (
        note
        + "\n\n"
        + _table(
            [
                "Lift",
                "Engine",
                "Phases / Standard",
                "Left-Only Phases",
                "Unmapped Keys",
                "Inferred Keys",
                "Worst Hand-Bar m (phase/side)",
            ],
            rows,
        )
    )


def _gaps(receipt: dict[str, Any]) -> str:
    rows = []
    for g in derive_gaps(receipt):
        rows.append(
            [
                g["title"],
                ", ".join(_e(e) for e in g["engines"]),
                "; ".join(g["issues"]),
                g["lift_story"],
                "new" if g["new_issue"] else "existing",
            ]
        )
    return _table(["Discrepancy", "Engines", "Issues", "Epic Child", "Issue"], rows)


def render_markdown(receipt: dict[str, Any]) -> str:
    """The complete baseline document for *receipt*."""
    deferred = STANDING_DEFERRALS + [
        {
            "item": f"{d.get('engine')} {d.get('lift', '')}".strip(),
            "reason": d["reason"],
        }
        for d in receipt.get("deferred", [])
    ]
    sections = [
        ("Setup", _setup(receipt)),
        ("Inventory", _inventory(receipt)),
        ("Mass and Segment Totals", _masses(receipt)),
        ("Same-Input Cross-Engine Parity", _parity(receipt)),
        ("Hand and Bar Attachment", _grips(receipt)),
        ("Start Poses", _starts(receipt)),
        ("Joint Limits Versus the Standard", _limits(receipt)),
        ("Phases", _phases(receipt)),
        ("Discrepancies and Tracking Issues", _gaps(receipt)),
        (
            "Epic Defect List Review",
            _table(["Issues", "Defect", "Finding"], [list(r) for r in EPIC_DEFECTS]),
        ),
        ("Deferred", "\n".join(f"- **{d['item']}**: {d['reason']}" for d in deferred)),
    ]
    body = "\n\n".join(f"## {t}\n\n{b}" for t, b in sections)
    return (
        "# Lift Pack Parity Baseline\n\n"
        "Measured baseline of the four lift model packs before any change "
        "(issue #11741, epic #11740). Generated by "
        "`scripts/lifting/run_pack_parity_baseline.py`; do not edit by hand.\n\n"
        + body
        + "\n\n## Reproduction\n\n"
        "```bash\n"
        "git clone --depth 1 https://github.com/D-sorganization/OpenSim_Models ~/Repositories/OpenSim_Models\n"
        "# likewise MuJoCo_Models, Drake_Models, Pinocchio_Models\n"
        "export PYTHONPATH=.:src MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen\n"
        "python3 scripts/lifting/run_pack_parity_baseline.py\n"
        "python3 scripts/lifting/render_pack_stills.py   # optional stills\n"
        "```\n\n"
        "The receipt is `docs/development/lifting/pack_parity_baseline.json`. "
        "`LIFT_PACK_ROOT` overrides the directory searched for the checkouts.\n"
    )
