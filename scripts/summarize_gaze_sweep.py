"""Gaze-weight sweep evidence (OSV-3b, #11729): rows, knees and the default.

    python3 -m scripts.summarize_gaze_sweep SWEEP_ROOT \\
        docs/development/full_body_models/evidence/head_gaze/gaze_weight_sweep.json

``SWEEP_ROOT`` holds one ``<capture>_<weight>/sweep_row.json`` per run of
``scripts.sweep_gaze_weight``. The script writes a compact evidence JSON (one
row per capture and weight, the per-capture knee and the cross-capture
default from ``gaze_sweep.select_default``) and, with ``--plot``, a Pareto
plot of marker RMS against gaze RMS.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching import gaze_sweep as gs

CAPTURE_LABELS = {"driver": "capture-A driver", "iron": "capture-B 7-iron"}


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The reported quantities of one sweep row (no plan or schedule detail)."""
    a2i = row["head_gaze"]["address_to_impact"]
    neck = row["head_gaze"].get("neck_ik_schedule", {})
    return {
        "gaze_weight": float(row["gaze_weight"]),
        "marker_rms_mm": float(row["marker_rms_mm"]),
        "closure_error_max_mm": float(row["closure_error_max_mm"]),
        "face_fit_rms_deg": float(row["face_fit_deg"]["rms_deg"]),
        "theta_gaze_rms_deg": float(a2i["theta_gaze_rms_deg"]),
        "theta_gaze_max_deg": float(a2i["theta_gaze_max_deg"]),
        "eye_translation_range_mm": [float(v) for v in a2i["eye_translation_range_mm"]],
        "head_yaw_pitch_roll_range_deg": [
            float(a2i[f"head_{k}_range_deg"]) for k in ("yaw", "pitch", "roll")
        ],
        "neck_ik_frames_with_clamping": neck.get("frames_with_clamping"),
    }


def load_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Sweep rows under ``root`` grouped by capture, sorted by weight."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*/sweep_row.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.setdefault(row["capture"], []).append(row)
    if not rows:
        raise ValueError(f"no sweep_row.json under {root}")
    for capture, items in rows.items():
        items.sort(key=lambda r: float(r["gaze_weight"]))
        weights = [float(r["gaze_weight"]) for r in items]
        if len(set(weights)) != len(weights):
            raise ValueError(f"duplicate weights for {capture}: {weights}")
    return rows


def summarize(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Evidence document: per-capture rows and knee, cross-capture default."""
    points = {c: [gs.point_from_row(r) for r in items] for c, items in rows.items()}
    captures = {}
    for capture, items in rows.items():
        base = gs.feasible(points[capture])
        captures[capture] = {
            "label": CAPTURE_LABELS.get(capture, capture),
            "rows": [compact_row(r) for r in items],
            "feasible_weights": [p.weight for p in base],
            "knee_weight": gs.select_knee(points[capture]).weight,
        }
    return {
        "issue": "#11729 (OSV-3b)",
        "engine": "mujoco (matching inverse kinematics, reference trajectory q_ref)",
        "rule": (
            "feasible: marker RMS <= (1 + marker_tolerance) x weight-0 RMS and "
            "OSV-10 face-fit RMS <= face_cap_deg; knee: farthest point below the "
            "chord of the feasible Pareto front (marker RMS, theta_gaze RMS); "
            "default: smallest per-capture knee among weights feasible in every "
            "capture"
        ),
        "marker_tolerance": gs.MARKER_TOLERANCE,
        "face_cap_deg": gs.FACE_CAP_DEG,
        "captures": captures,
        "selected_weight": gs.select_default(points),
        "note": (
            "the gaze-regularised head is a soft prior, not measured; qualified "
            "receipts keep --gaze-weight 0 (marker-faithful)"
        ),
    }


def plot(summary: Mapping[str, Any], out_png: Path) -> None:
    """Marker RMS against gaze RMS per capture, weights annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=110)
    for block in summary["captures"].values():
        rows = block["rows"]
        xs = [r["marker_rms_mm"] for r in rows]
        ys = [r["theta_gaze_rms_deg"] for r in rows]
        ax.plot(xs, ys, "o-", label=block["label"])
        for r in rows:
            ax.annotate(
                f"{r['gaze_weight']:g}",
                (r["marker_rms_mm"], r["theta_gaze_rms_deg"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        limit = rows[0]["marker_rms_mm"] * (1.0 + summary["marker_tolerance"])
        ax.axvline(limit, linestyle=":", linewidth=0.8, color=ax.lines[-1].get_color())
    ax.set_xlabel("Marker RMS (mm)")
    ax.set_ylabel("Gaze Error RMS, Address to Impact (deg)")
    ax.set_title(f"Gaze Weight Sweep (Selected Weight {summary['selected_weight']:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--plot", type=Path)
    ns = parser.parse_args(argv)
    summary = summarize(load_rows(ns.root))
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if ns.plot is not None:
        plot(summary, ns.plot)


if __name__ == "__main__":
    main()
