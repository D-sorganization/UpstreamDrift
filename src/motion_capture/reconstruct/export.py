"""Hand the reconstruction to the motion pipeline (#9664).

The fitted joints are written as an OpenSim TRC marker file — the format
``motion_pipeline.sources.TRCAdapter`` already reads — so scaling, inverse
kinematics, engine retargeting and the model-matching tools consume rig
output without a new adapter. A canonical JSON alongside keeps the same
numbers with joint names and units stated explicitly.

TRC layout (tab separated, five header rows, millimetres):
``PathFileType 4 (X/Y/Z) <name>`` / header keys / header values /
``Frame# Time <marker>...`` / ``  X1 Y1 Z1 ...`` then one row per frame.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

from ..provenance import write_stamped

EXPORT_SCHEMA_VERSION = "reconstruction-export/1.0.0"

from .skeleton import JOINT_NAMES

Array = npt.NDArray[np.float64]
M_TO_MM = 1000.0


def trc_text(
    joints_m: Array,
    fps: float,
    *,
    names: Sequence[str] = JOINT_NAMES,
    file_name: str = "reconstruction.trc",
) -> str:
    """The TRC document for ``joints_m`` ``(T, J, 3)`` metres at ``fps``.

    Preconditions: a 3-D array with one name per joint and a positive rate.
    Postcondition: a NaN joint is written as ``nan`` cells; an empty cell
    would shift every column after it in a whitespace-split reader.
    """
    require(joints_m.ndim == 3 and joints_m.shape[2] == 3, "joints must be (T, J, 3)")
    require(joints_m.shape[1] == len(names), "one name per joint")
    require(fps > 0, "fps must be positive", fps)
    frames = joints_m.shape[0]
    rate = f"{fps:g}"
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{file_name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate}\t{rate}\t{frames}\t{len(names)}\tmm\t{rate}\t1\t{frames}",
        "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t",
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(names) + 1)),
        "",
    ]
    for t in range(frames):
        cells = [str(t + 1), f"{t / fps:.6f}"]
        for j in range(len(names)):
            p = joints_m[t, j]
            if np.isfinite(p).all():
                cells.extend(f"{v * M_TO_MM:.3f}" for v in p)
            else:
                cells.extend(["nan", "nan", "nan"])
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def write_trc(joints_m: Array, fps: float, path: Path, **kw: Any) -> Path:
    path.write_text(
        trc_text(joints_m, fps, file_name=path.name, **kw), encoding="utf-8"
    )
    return path


def canonical_payload(
    joints_m: Array,
    fps: float,
    *,
    names: Sequence[str] = JOINT_NAMES,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A self-describing JSON: joint names, metres, one row per frame."""
    require(joints_m.ndim == 3 and joints_m.shape[1] == len(names), "shape/names")
    rows = []
    for t in range(joints_m.shape[0]):
        frame = joints_m[t]
        rows.append(
            {
                "time_s": t / fps,
                "joints_m": [
                    [None if not np.isfinite(v) else float(v) for v in p] for p in frame
                ],
            }
        )
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "units": "m",
        "fps": fps,
        "joint_names": list(names),
        "frames": rows,
        "provenance": provenance or {},
    }


def export_reconstruction(
    reconstruct_dir: Path,
    *,
    trc_path: Path | None = None,
    json_path: Path | None = None,
) -> dict[str, str]:
    """Export ``<reconstruct>/joints_3d_m.npy`` at the fps the take was ingested at.

    Precondition: the directory holds ``joints_3d_m.npy`` and
    ``session_reconstruction.json`` (written by ``reconstruct_session``).
    Returns the files written.
    """
    joints_file = reconstruct_dir / "joints_3d_m.npy"
    summary_file = reconstruct_dir / "session_reconstruction.json"
    require(joints_file.is_file(), "no fitted joints", str(joints_file))
    require(summary_file.is_file(), "no session reconstruction", str(summary_file))
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    fps = float(summary.get("fps") or 0.0)
    if fps <= 0:
        swing = reconstruct_dir / "swing_summary.json"
        if swing.is_file():
            fps = float(json.loads(swing.read_text(encoding="utf-8")).get("fps") or 0)
    require(fps > 0, "session reconstruction does not state the fps")
    joints = np.load(joints_file)
    written: dict[str, str] = {}
    provenance = {
        "reconstruction_file": summary.get("reconstruction_file"),
        "views": summary.get("views"),
        "rms_px": summary.get("rms_px"),
    }
    trc = trc_path or reconstruct_dir / "reconstruction.trc"
    write_trc(joints, fps, trc)
    written["trc"] = str(trc)
    out_json = json_path or reconstruct_dir / "reconstruction_export.json"
    write_stamped(
        out_json,
        canonical_payload(joints, fps, provenance=provenance),
        schema_version=EXPORT_SCHEMA_VERSION,
        module=__name__,
        inputs=[joints_file, summary_file],
        derived_from=[summary_file],
        base=reconstruct_dir.parent,
    )
    written["json"] = str(out_json)
    return written
