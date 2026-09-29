"""Diagnose a GolfSwing3D input pose (MAT or CSV) without MATLAB.

The GolfSwing3D ``3DModelInputs*.mat`` files store ``Simulink.Parameter``
objects (MCOS opaque types). ``scipy.io`` can identify the file format
but cannot decode the wrapped scalar values. As a workaround this
script accepts EITHER:

  * a ``.mat`` file, in which case it extracts whatever the public
    ``scipy.io.loadmat`` API exposes plus a structural inventory of
    the embedded MCOS workspace; OR
  * a ``.csv`` produced by the Dataset Generator (see
    ``matlab/Scripts/Dataset Generator/``), whose first row contains
    the joint-angle scalars under columns named ``model_<Field>``.

For CSV inputs the script then runs the lightweight Python forward
kinematics evaluator, plots a 3D skeleton, and writes a markdown
report listing per-joint deviations from a reference golfer pose.

Usage
-----
    python3 scripts/diagnose_input_pose.py --input <path> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "shared" / "python"))

from motion_matching.diagnostics.forward_kinematics import (  # noqa: E402
    forward_kinematics,
)
from motion_matching.diagnostics.reference_pose import (  # noqa: E402
    REFERENCE_GOLFER_FIELDS,
    compare_to_reference,
    reference_golfer_setup,
)


def _load_csv_first_row(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    out: dict[str, float] = {}
    for f in REFERENCE_GOLFER_FIELDS:
        col = f"model_{f}"
        if col in header:
            try:
                out[f] = float(row[header.index(col)])
            except (ValueError, IndexError):
                continue
    return out


def _inventory_mat(path: Path) -> dict[str, object]:
    """Return a high-level inventory of a MAT file.

    For v5 MAT files containing only MCOS opaque objects
    (the case for ``3DModelInputs*.mat``), the inventory will
    note that fact rather than try to decode them.
    """
    import scipy.io  # local import keeps CSV-only path lean

    inventory: dict[str, object] = {"path": str(path), "format": "MAT v5"}
    try:
        data = scipy.io.loadmat(path)
    except Exception as exc:  # noqa: BLE001 - inventory must surface any I/O failure
        inventory["error"] = repr(exc)
        return inventory
    fields: list[dict[str, object]] = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        entry: dict[str, object] = {"name": k}
        if hasattr(v, "shape"):
            entry["shape"] = list(v.shape)
            entry["dtype"] = str(v.dtype)
        else:
            entry["type"] = type(v).__name__
        fields.append(entry)
    inventory["fields"] = fields
    fw = data.get("__function_workspace__")
    if fw is not None:
        inventory["mcos_workspace_bytes"] = int(fw.nbytes)
        inventory["note"] = (
            "File stores Simulink.Parameter objects in MCOS subsystem; "
            "scalar Values are not directly decodable in Python. "
            "Run the diagnostic against a Dataset Generator CSV "
            "(model_<Field> columns) to access numeric values."
        )
    return inventory


def _plot_skeleton(angles: dict[str, float], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        return  # plotting is optional
    pose = forward_kinematics(angles)
    ref_pose = forward_kinematics(reference_golfer_setup())

    fig = matplotlib.pyplot.figure(figsize=(10, 6))
    for idx, (label, p) in enumerate(
        [("input pose", pose), ("reference golfer", ref_pose)]
    ):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        edges = [
            ("pelvis", "spine_top"),
            ("spine_top", "torso_top"),
            ("torso_top", "l_shoulder"),
            ("torso_top", "r_shoulder"),
            ("l_shoulder", "l_elbow"),
            ("l_elbow", "l_wrist"),
            ("l_wrist", "l_hand"),
            ("r_shoulder", "r_elbow"),
            ("r_elbow", "r_wrist"),
            ("r_wrist", "r_hand"),
            ("butt", "clubhead"),
        ]
        for a_, b_ in edges:
            xa = [p[a_][0], p[b_][0]]
            ya = [p[a_][1], p[b_][1]]
            za = [p[a_][2], p[b_][2]]
            ax.plot(xa, ya, za, "-o", linewidth=2, markersize=3)
        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        # Roughly equal aspect.
        all_pts = np.array(list(p.points.values()))
        rng = np.ptp(all_pts, axis=0).max() / 2 + 0.1
        ctr = all_pts.mean(axis=0)
        ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
        ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
        ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    matplotlib.pyplot.close(fig)


def _write_report(
    angles: dict[str, float],
    flags: list[dict[str, float | str]],
    out_path: Path,
) -> None:
    lines = ["# Input pose diagnostic\n"]
    lines.append("## Joint-angle inventory\n")
    lines.append("| Field | Value (deg) |")
    lines.append("|---|---:|")
    for f in REFERENCE_GOLFER_FIELDS:
        v = angles.get(f, "—")
        if isinstance(v, float):
            lines.append(f"| `{f}` | {v:+.3f} |")
        else:
            lines.append(f"| `{f}` | {v} |")
    lines.append("\n## Outliers vs reference address ranges\n")
    if not flags:
        lines.append("_None — pose is within plausible address ranges._\n")
    else:
        lines.append("| Field | Value | Range | Deviation |")
        lines.append("|---|---:|:---:|---:|")
        for fl in flags:
            lines.append(
                f"| `{fl['field']}` | {fl['value']:+.3f} | "
                f"[{fl['low']:+.1f}, {fl['high']:+.1f}] | "
                f"{fl['deviation']:+.3f} |"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to .mat or .csv input")
    ap.add_argument("--out-dir", required=True, help="Directory for outputs")
    args = ap.parse_args(argv)

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".mat":
        inventory = _inventory_mat(src)
        (out_dir / "inventory.json").write_text(
            json.dumps(inventory, indent=2), encoding="utf-8"
        )
        sys.stdout.write(json.dumps(inventory, indent=2) + "\n")
        return 0

    if src.suffix.lower() != ".csv":
        sys.stderr.write(f"unsupported input type: {src.suffix}\n")
        return 2

    angles = _load_csv_first_row(src)
    flags = compare_to_reference(angles)
    _plot_skeleton(angles, out_dir / "skeleton.png")
    _write_report(angles, flags, out_dir / "report.md")
    sys.stdout.write(
        f"Loaded {len(angles)} angles from {src.name}; flagged {len(flags)} outliers.\n"
    )
    for fl in flags:
        sys.stdout.write(
            f"  OUTLIER {fl['field']:30s} value={fl['value']:+8.3f} "
            f"range=[{fl['low']:+.1f},{fl['high']:+.1f}] "
            f"deviation={fl['deviation']:+.3f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
