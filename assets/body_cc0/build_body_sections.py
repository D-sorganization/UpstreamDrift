#!/usr/bin/env python3
"""Measure the cross-sections of the MakeHuman body for the lofts (#15299).

Usage::

    python3 assets/body_cc0/build_body_sections.py PATH/TO/makehuman-checkout [--check]

Writes ``body_sections.json``: per macro mix of ``section_settings.json`` (the
neutral base mesh and a male athletic mix), the sections of the torso, neck,
upper arm, forearm, thigh and shank along their own bone axes, the shoulder
(trapezius slope, deltoid cap) and three head landmarks. No mesh is stored;
the table is numbers only, with the SHA-256 of every source file it came from.

Each row of a segment is ``LOFT_COLUMNS`` then ``SIZE_COLUMNS``: a loft consumes
the first five
(``along`` as a fraction of the segment length, ``width`` and ``depth`` as
fractions of the segment's largest half width and half depth, ``forward`` =
how far the middle of the section sits in front of the axis as a fraction of
the length, and the superellipse exponent); the rest are millimetres at the
stature of the settings. ``depth`` is the mean of the front and back depths.

``--check`` rebuilds in memory and compares with the committed table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makehuman_cc0"))

from mh_rig_io import MakeHumanRig, load_makehuman_rig, sha256_file  # noqa: E402
from mh_sections import (  # noqa: E402
    SegmentTable,
    expand_bones,
    joint_point,
    segment_table,
    skin_volume,
    weighted_vertices,
)
from mh_targets import macro_mix, morphed_rig  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "body_sections.json"
SETTINGS = HERE / "section_settings.json"
LOFT_COLUMNS = ["along", "width", "depth", "forward", "squareness"]
SIZE_COLUMNS = ["half_width_mm", "depth_front_mm", "depth_back_mm", "girth_mm"]
LINE_WIDTH = 80  # Prettier's; a list longer than this stays one item per line


def normalised_rows(table: SegmentTable, mm: float) -> dict[str, Any]:
    """``table`` as loft fractions and millimetres (``mm`` per rig unit)."""
    widths = np.array([s.half_width for s in table.sections])
    depths = np.array([0.5 * (s.depth_front + s.depth_back) for s in table.sections])
    rows = []
    for s, width, depth in zip(table.sections, widths, depths, strict=True):
        forward = 0.5 * (s.depth_front - s.depth_back) / table.length
        loft = [s.along, width / widths.max(), depth / depths.max(), forward]
        sizes = [s.half_width, s.depth_front, s.depth_back, s.girth]
        rows.append(
            [round(float(x), 4) for x in loft]
            + [round(s.squareness, 2)]
            + [round(x * mm, 1) for x in sizes]
        )
    girth = max(s.girth for s in table.sections)
    return {
        "length_mm": round(table.length * mm, 1),
        "half_width_hint_mm": round(float(widths.max()) * mm, 1),
        "half_depth_hint_mm": round(float(depths.max()) * mm, 1),
        "belly_along": round(table.belly_along, 4),
        "max_radius_mm": round(girth * mm / (2.0 * np.pi), 1),
        "rows": rows,
    }


def shoulder_measures(
    rig: MakeHumanRig, shoulder: Mapping[str, Any], frame: np.ndarray, mm: float
) -> dict[str, Any]:
    """Trapezius slope and deltoid cap of the shoulder (rows of ``frame``:
    lateral, up, front).

    The top of the skin is read from ``slope_from`` of the way out from the
    neck base to the shoulder joint, to the acromion over the joint; the slope
    is the straight line that fits it best. The cap is the skin beyond the
    joint, away from the arm.
    """
    neck = joint_point(rig, shoulder["neck_base"])
    joint = joint_point(rig, shoulder["joint"])
    elbow = joint_point(rig, shoulder["arm_end"])
    side = float(np.sign((joint - neck) @ frame[0]))
    run = abs(float((joint - neck) @ frame[0]))
    least = float(shoulder["min_weight"])
    skin = weighted_vertices(rig, expand_bones(rig, shoulder["bones"]), least)
    local = (rig.vertices[skin] - neck) @ frame.T
    window = float(shoulder["window"]) * run
    rows = []
    for share in np.linspace(
        float(shoulder["slope_from"]), 1.0, int(shoulder["steps"])
    ):
        near = np.abs(side * local[:, 0] - share * run) <= window
        if not near.any():
            raise ValueError(f"no shoulder skin at {share:.2f} of the way out")
        rows.append(
            [round(float(share), 4), round(float(-local[near, 1].max()) * mm, 1)]
        )
    out, below = np.asarray(rows).T
    slope = float(np.polyfit(out * run * mm, below, 1)[0])  # least squares
    cap = weighted_vertices(rig, expand_bones(rig, shoulder["cap_bones"]), least)
    away = rig.vertices[cap] - joint
    beyond = away[away @ (joint - elbow) >= 0.0]
    return {
        "neck_base_to_joint_mm": round(run * mm, 1),
        "trapezius_slope_deg": round(float(np.degrees(np.arctan(slope))), 1),
        "top_columns": ["out", "below_neck_base_mm"],
        "top_rows": rows,
        "deltoid_cap_radius_mm": round(
            float(np.median(np.linalg.norm(beyond, axis=1))) * mm, 1
        ),
        "skin_above_joint_mm": round(float((beyond @ frame[1]).max()) * mm, 1),
        "skin_outside_joint_mm": round(float((side * beyond @ frame[0]).max()) * mm, 1),
    }


def head_landmarks(
    rig: MakeHumanRig, head: Mapping[str, Any], frame: np.ndarray, mm: float
) -> dict[str, float]:
    """Menton to vertex, and breadth and length of the head above the ears."""
    body = rig.vertices[np.unique(rig.body_quads)] @ frame.T
    bones = expand_bones(rig, head["jaw_bones"])
    jaw = weighted_vertices(rig, bones, float(head["min_weight"]))
    menton = float((rig.vertices[jaw] @ frame[1]).min())
    vertex = float(body[:, 1].max())
    skull = body[body[:, 1] >= menton + float(head["skull_from"]) * (vertex - menton)]
    return {
        "menton_to_vertex_mm": round((vertex - menton) * mm, 1),
        "breadth_mm": round(float(np.ptp(skull[:, 0])) * mm, 1),
        "length_mm": round(float(np.ptp(skull[:, 2])) * mm, 1),
    }


def measure_mix(rig: MakeHumanRig, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Every table of one (already morphed) rig, at the settings' stature."""
    up, front = (np.asarray(settings[k], float) for k in ("up", "front"))
    frame = np.stack([np.cross(up, front), up, front])
    height = rig.vertices[np.unique(rig.body_quads)] @ up
    stature = float(np.ptp(height))
    mm = 1000.0 * float(settings["stature_m"]) / stature
    shared = {"min_weight": settings["min_weight"]}  # one skin-weight threshold
    segments = {}
    for name, segment in settings["segments"].items():
        cut = {**shared, **segment}
        cut["slice_bones"] = expand_bones(rig, segment["slice_bones"])
        table = segment_table(rig, cut, front, settings["fit"])
        segments[name] = normalised_rows(table, mm)
    return {
        "source_stature_m": round(stature * float(settings["makehuman_unit_m"]), 4),
        "volume_l": round(skin_volume(rig) * (mm / 100.0) ** 3, 1),
        "segments": segments,
        "shoulder": shoulder_measures(
            rig, {**shared, **settings["shoulder"]}, frame, mm
        ),
        "head": head_landmarks(rig, {**shared, **settings["head"]}, frame, mm),
    }


def build(makehuman_root: Path) -> dict[str, Any]:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    rig = load_makehuman_rig(makehuman_root)
    mixes = {}
    for name, macro in settings["mixes"].items():
        mix = macro_mix(macro) if macro is not None else {}
        morphed, targets = morphed_rig(rig, makehuman_root, mix)
        mixes[name] = {"targets": targets, **measure_mix(morphed, settings)}
    sources = [
        {"path": rel, "sha256": sha256_file(makehuman_root / rel)}
        for rel in settings["source_files"]
    ]
    return {
        "schema": "body-sections/1",
        "licence": "CC0-1.0",
        "source_repository": settings["source_repository"],
        "source_commit": settings["source_commit"],
        "sources": sources,
        "settings_sha256": sha256_file(SETTINGS),
        "stature_m": settings["stature_m"],
        "loft_columns": LOFT_COLUMNS,
        "size_columns": SIZE_COLUMNS,
        "mixes": mixes,
    }


def dumps(record: Mapping[str, Any]) -> str:
    """JSON as Prettier prints it: a short list of plain values on one line."""
    text = json.dumps(record, indent=2)

    def flat(match: re.Match[str]) -> str:
        items = ", ".join(part.strip() for part in match[2].split(",\n"))
        line = f"{match[1]}[{items}]"
        return line if len(line) < LINE_WIDTH else match[0]

    pattern = r"(?m)^(.*)\[\n\s+([^\[\]{}]+?)\n\s+\]"
    return re.sub(pattern, flat, text) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("makehuman_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = build(args.makehuman_root)
    if args.check:
        same = json.loads(OUT.read_text(encoding="utf-8")) == json.loads(dumps(record))
        print("body_sections.json matches the MakeHuman source" if same else "MISMATCH")  # noqa: T201
        return 0 if same else 1
    OUT.write_text(dumps(record), encoding="utf-8")
    print(f"wrote {OUT.name}: {', '.join(record['mixes'])}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
