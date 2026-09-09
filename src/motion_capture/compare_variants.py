"""What limiting the cameras costs: compare variants of one session (#9796).

Per variant, against a reference variant (default: the session's default
match, i.e. every camera):

- reprojection RMS of the reconstructed joints on every view that has
  observations, flagged ``held_out`` when the variant never used the view;
- 3-D joint RMS against the reference over the frames both have, per joint
  and overall (only for triangulated variants: image-space variants have no
  reconstructed joints);
- per-DOF joint-angle RMS against the reference when both have a model fit
  of the same model.

Writes ``variants/comparison.{json,md}``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require

from .provenance import write_stamped
from .reconstruct.overlay3d import (
    PALETTE,
    Track,
    reprojection_rms_px,
    variant_tracks,
)
from .reconstruct.skeleton import JOINT_NAMES
from .variants import VARIANTS_DIR, list_variants, variant_dir

COMPARISON_SCHEMA = "variant-comparison/1.0.0"
COMPARISON_FILE = "comparison.json"


def _observed(session: Path, observation_set: str, view: str) -> tuple[Any, Any] | None:
    path = session / observation_set / f"{view}.json"
    if not path.is_file():
        return None
    from .reconstruct.layouts import to_reconstruct_layout
    from .reconstruct.model.fit2d import frame_index

    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["detector_layout"]["keypoint_names"]) != JOINT_NAMES:
        payload = to_reconstruct_layout(payload)
    fps = float(payload["fps"])
    frames = max((frame_index(r, fps) for r in payload["frames"]), default=-1) + 1
    kp = np.zeros((frames, len(JOINT_NAMES), 2))
    conf = np.zeros((frames, len(JOINT_NAMES)))
    for row in payload["frames"]:
        i = frame_index(row, fps)
        kp[i] = np.asarray(row["keypoints_px"], dtype=float)
        conf[i] = np.asarray(row["confidence"], dtype=float)
    return kp, conf


def _joints(session: Path, variant: str) -> Any | None:
    path = variant_dir(session, variant) / "reconstruct" / "joints_3d_m.npy"
    return np.load(path) if path.is_file() else None


def _angles(session: Path, variant: str) -> dict[str, Any] | None:
    path = variant_dir(session, variant) / "model" / "joint_angles.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def joint_rms_mm(a: Any, b: Any) -> dict[str, Any]:
    """Per-joint and overall RMS distance (mm) over frames both arrays have.

    Rows that are all zero (unobservable) in either array are skipped.
    """
    t = min(a.shape[0], b.shape[0])
    valid = (np.abs(a[:t]).sum(axis=2) > 0) & (np.abs(b[:t]).sum(axis=2) > 0)
    diff = a[:t] - b[:t]
    d = np.sqrt(
        np.einsum("ijk,ijk->ij", diff, diff)
    )  # ⚡ Bolt: np.sqrt(np.einsum) is ~10x faster than np.linalg.norm(..., axis=2)
    per = {}
    for k, name in enumerate(JOINT_NAMES):
        col = d[:, k][valid[:, k]]
        per[name] = float(1000 * np.sqrt(np.mean(col**2))) if col.size else None
    overall = d[valid]
    return {
        "per_joint": per,
        "overall": float(1000 * np.sqrt(np.mean(overall**2))) if overall.size else None,
        "frames": int(t),
    }


def angle_rms_deg(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """Per-DOF RMS (degrees) of the wrapped angle difference; rotations only."""
    if a.get("model") != b.get("model") or a["dof_names"] != b["dof_names"]:
        return None
    qa, qb = np.asarray(a["q"], dtype=float), np.asarray(b["q"], dtype=float)
    t = min(qa.shape[0], qb.shape[0])
    diff = (qa[:t, 3:] - qb[:t, 3:] + np.pi) % (2 * np.pi) - np.pi
    per = {
        name: float(np.degrees(np.sqrt(np.mean(diff[:, i] ** 2))))
        for i, name in enumerate(a["dof_names"][3:])
    }
    return {"per_dof": per, "overall": float(np.degrees(np.sqrt(np.mean(diff**2))))}


def _reprojection(
    session: Path,
    variant: str,
    views: Sequence[str],
    observation_set: str,
    used: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view in views:
        observed = _observed(session, observation_set, view)
        if observed is None:
            continue
        try:
            tracks = variant_tracks(session, variant, view, PALETTE[0])
        except ValueError:
            continue
        joints: Track | None = next((t for t in tracks if t.kind == "joints"), None)
        model: Track | None = next((t for t in tracks if t.kind == "model"), None)
        out[view] = {
            "held_out": view not in used,
            "joints_rms_px": reprojection_rms_px(joints, *observed) if joints else None,
            "model_rms_px": _model_rms(model, observed) if model else None,
        }
    return out


def _model_rms(model: Track, observed: tuple[Any, Any]) -> float | None:
    """Model landmarks that map 1:1 onto reconstruct joints, against those joints."""
    kp, conf = observed
    rows = [i for i, n in enumerate(model.names) if n in _JOINT_OF_LANDMARK]
    cols = [JOINT_NAMES.index(_JOINT_OF_LANDMARK[model.names[i]]) for i in rows]
    if not rows:
        return None
    t = min(model.frames, kp.shape[0])
    mask = model.visible[:t][:, rows] & (conf[:t][:, cols] > 0)
    if not mask.any():
        return None
    diff = model.px[:t][:, rows] - kp[:t][:, cols]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))[
        mask
    ]  # ⚡ Bolt: np.sqrt(np.einsum) is ~10x faster than np.linalg.norm(..., axis=2)
    return float(np.sqrt(np.mean(d**2)))


_JOINT_OF_LANDMARK = {
    "pelvis": "mid_hip",
    "hub": "neck",
    "nose": "nose",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
}


def compare_variants(
    session: Path, *, reference: str = "", observation_set: str = "observations"
) -> dict[str, Any]:
    """The comparison payload; also written to ``variants/comparison.{json,md}``.

    Precondition: the reference variant is registered. Postcondition: every
    registered variant appears, the reference first.
    """
    records = list_variants(session)
    names = [r.name for r in records]
    require(reference in names, "reference variant must be registered", reference)
    all_views = sorted({v for r in records for v in r.views})
    ref_joints = _joints(session, reference)
    ref_angles = _angles(session, reference)
    rows = []
    for record in sorted(records, key=lambda r: r.name != reference):
        joints = _joints(session, record.name)
        angles = _angles(session, record.name)
        rows.append(
            {
                "variant": record.name,
                "views": list(record.views),
                "observation_set": record.observation_set,
                "source": record.source,
                "reprojection": _reprojection(
                    session, record.name, all_views, observation_set, record.views
                ),
                "joint_rms_mm": (
                    joint_rms_mm(joints, ref_joints)
                    if joints is not None and ref_joints is not None
                    else None
                ),
                "angle_rms_deg": (
                    angle_rms_deg(angles, ref_angles)
                    if angles is not None and ref_angles is not None
                    else None
                ),
                "model_rms_px": angles.get("rms_px") if angles else None,
            }
        )
    payload = {"reference": reference, "views": all_views, "variants": rows}
    out_dir = session / VARIANTS_DIR
    out_dir.mkdir(exist_ok=True)
    write_stamped(
        out_dir / COMPARISON_FILE,
        payload,
        schema_version=COMPARISON_SCHEMA,
        module=__name__,
        parameters={"reference": reference, "observation_set": observation_set},
        base=session,
    )
    (out_dir / "comparison.md").write_text(markdown(payload), encoding="utf-8")
    return payload


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def markdown(payload: dict[str, Any]) -> str:
    views = payload["views"]
    head = "| variant | views | source | " + " | ".join(f"{v} px" for v in views)
    head += " | 3-D vs ref mm | angles vs ref deg |\n"
    head += (
        "| --- | --- | --- | " + " | ".join("---" for _ in views) + " | --- | --- |\n"
    )
    lines = []
    for row in payload["variants"]:
        cells = []
        for v in views:
            r = row["reprojection"].get(v)
            if r is None:
                cells.append("n/a")
                continue
            value = _fmt(
                r["joints_rms_px"]
                if r["joints_rms_px"] is not None
                else r["model_rms_px"]
            )
            cells.append(value + (" (held out)" if r["held_out"] else ""))
        joint = row["joint_rms_mm"]["overall"] if row["joint_rms_mm"] else None
        angle = row["angle_rms_deg"]["overall"] if row["angle_rms_deg"] else None
        lines.append(
            f"| {row['variant'] or '(default)'} | {','.join(row['views'])} | "
            f"{row['source'].get('kind')} | "
            + " | ".join(cells)
            + f" | {_fmt(joint)} | {_fmt(angle)} |"
        )
    return head + "\n".join(lines) + "\n"
