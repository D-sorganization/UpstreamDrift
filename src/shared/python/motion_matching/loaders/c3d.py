"""C3D loader for Optical-style mocap files.

Reuses the existing ``C3DDataReader`` from
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py``.
Marker-name discovery is heuristic because the Optical marker set is not
documented in this repo (issue #013 is the verification pass).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.python.core.contracts import postcondition, precondition

from ..club_target import AlignOptions, ClubTarget, SourceProvenance
from ._align import detect_impact_index, resample_target

logger = logging.getLogger(__name__)

_C3D_READER_RELATIVE = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src"
)
BUTT_CANDIDATES: tuple[str, ...] = (
    "BUTT",
    "GRIP",
    "Grip",
    "GripButt",
    "ClubButt",
    "BUTT_END",
    "CLUB_BUTT",
)
HEAD_CANDIDATES: tuple[str, ...] = (
    "CH",
    "ClubHead",
    "CLUBHEAD",
    "Clubhead",
    "HEAD",
    "ClubFace",
    "CLUB_HEAD",
)


def _import_c3d_reader():
    """Side-load the legacy ``c3d_reader`` package by file path."""
    if "_c3d_reader_compat" in sys.modules:
        return sys.modules["_c3d_reader_compat"]
    cwd = Path.cwd()
    candidates = [cwd / _C3D_READER_RELATIVE]
    for parent in cwd.parents:
        candidates.append(parent / _C3D_READER_RELATIVE)
    for base in candidates:
        target = base / "c3d_reader.py"
        if target.is_file():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            spec = importlib.util.spec_from_file_location(
                "_c3d_reader_compat", str(target)
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["_c3d_reader_compat"] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"Could not locate c3d_reader.py (searched relative to {cwd})")


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_marker(labels: list[str], candidates: tuple[str, ...]) -> str | None:
    """First label whose upper-cased form contains any candidate substring."""
    upper = [lbl.upper() for lbl in labels]
    for cand in candidates:
        u = cand.upper()
        for i, lbl in enumerate(upper):
            if u in lbl:
                return labels[i]
    return None


def _marker_xyz(df: pd.DataFrame, marker: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time, xyz)`` for a single marker from a tidy points dataframe."""
    sub = df[df["marker"] == marker].sort_values("frame")
    time = sub["time"].to_numpy(dtype=np.float64)
    xyz = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return time, xyz


@precondition(
    lambda path, opts: Path(path).exists(),
    "C3D file must exist",
)
@precondition(
    lambda path, opts: opts.sample_rate_hz > 0,
    "sample_rate_hz must be > 0",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "load_club_target_c3d must return a ClubTarget",
)
def load_club_target_c3d(path: Path | str, opts: AlignOptions) -> ClubTarget:
    """Load a Optical-style C3D file into a canonical ``ClubTarget``.

    Orientation is reconstructed from the (butt, clubhead) shaft direction
    alone — the C3D file does not carry the 3x3 rotation matrices that the
    Excel sheets do, so the quaternion encodes the swing of an axis-aligned
    shaft (no roll information). Issue #013 will refine this once the Optical
    marker convention is documented.
    """
    path = Path(path)
    reader_mod = _import_c3d_reader()
    reader = reader_mod.C3DDataReader(path)
    metadata = reader.get_metadata()
    df = reader.points_dataframe(include_time=True, target_units="m")
    labels = list(metadata.marker_labels)
    butt_label = _pick_marker(labels, BUTT_CANDIDATES)
    head_label = _pick_marker(labels, HEAD_CANDIDATES)
    if butt_label is None or head_label is None:
        raise ValueError(
            "Could not identify club butt or clubhead markers in C3D file. "
            f"Available labels: {labels}"
        )

    t_butt, butt_raw = _marker_xyz(df, butt_label)
    t_head, head_raw = _marker_xyz(df, head_label)
    if t_butt.shape != t_head.shape:
        raise ValueError("Butt and clubhead marker time vectors disagree on length")
    raw_time = t_butt - float(t_butt[0])
    raw_quat = _shaft_quaternions(butt_raw, head_raw)

    impact_raw = detect_impact_index(raw_time, head_raw)
    sim_time, butt, clubhead, quat, impact_idx = resample_target(
        raw_time, butt_raw, head_raw, raw_quat, impact_raw, opts
    )

    source = SourceProvenance(
        filename=path.name,
        format="c3d",
        subject_id=path.stem,
        trial_id=path.stem,
        sha256=_sha256_of(path),
    )
    logger.info(
        "Loaded ClubTarget from %s: %d output samples (impact=%d)",
        path.name,
        sim_time.shape[0],
        impact_idx,
    )
    return ClubTarget(
        time=sim_time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=int(impact_idx),
        source=source,
    )


def _shaft_quaternions(butt: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Quaternion that rotates ``+z`` onto each shaft direction.

    This is a stand-in until issue #013 documents the Optical marker convention
    for full 3-DOF club orientation.
    """
    n = butt.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        v = head[i] - butt[i]
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            out[i] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        v = v / norm
        dot = float(np.dot(z_axis, v))
        if dot >= 1.0 - 1.0e-12:
            out[i] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        if dot <= -1.0 + 1.0e-12:
            out[i] = np.array([0.0, 1.0, 0.0, 0.0])
            continue
        axis = np.cross(z_axis, v)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot)
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        q = np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])
        if q[0] < 0:
            q = -q
        out[i] = q / np.linalg.norm(q)
    return out
