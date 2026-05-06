"""Wiffle/ProV1 Excel loader.

Reuses ``mocap_data_loader.process_excel_sheet`` from the existing Motion
Capture Plotter rather than re-parsing the workbook.
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
from ._quaternion import rotmat_to_quat

logger = logging.getLogger(__name__)

ALLOWED_SHEETS: frozenset[str] = frozenset(
    {"TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"}
)
INCHES_TO_METERS = 0.0254
_MOCAP_LOADER_RELATIVE = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
    "golf_gui/Motion Capture Plotter/mocap_data_loader.py"
)


def _import_mocap_loader():
    """Side-load the legacy ``mocap_data_loader`` module by file path.

    The Motion Capture Plotter directory has spaces in its name, so it is not
    importable as a regular package. We therefore load it through ``importlib``.
    """
    if "mocap_data_loader" in sys.modules:
        return sys.modules["mocap_data_loader"]
    cwd = Path.cwd()
    candidates = [cwd / _MOCAP_LOADER_RELATIVE]
    for parent in cwd.parents:
        candidates.append(parent / _MOCAP_LOADER_RELATIVE)
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "mocap_data_loader", str(candidate)
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["mocap_data_loader"] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        f"Could not locate mocap_data_loader.py (searched relative to {cwd})"
    )


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_to_quat(df_row: pd.Series) -> np.ndarray:
    """Build a rotation matrix from the 9 club direction-cosine columns."""
    return np.array(
        [
            [df_row["club_Xx"], df_row["club_Yx"], df_row["club_Zx"]],
            [df_row["club_Xy"], df_row["club_Yy"], df_row["club_Zy"]],
            [df_row["club_Xz"], df_row["club_Yz"], df_row["club_Zz"]],
        ],
        dtype=np.float64,
    )


@precondition(
    lambda path, sheet, opts: Path(path).exists(),
    "Excel file must exist",
)
@precondition(
    lambda path, sheet, opts: sheet in ALLOWED_SHEETS,
    f"sheet must be one of {sorted(ALLOWED_SHEETS)}",
)
@precondition(
    lambda path, sheet, opts: opts.sample_rate_hz > 0,
    "sample_rate_hz must be > 0",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "load_club_target_excel must return a ClubTarget",
)
def load_club_target_excel(
    path: Path | str, sheet: str, opts: AlignOptions
) -> ClubTarget:
    """Load a single Wiffle-style sheet into a canonical ``ClubTarget``."""
    path = Path(path)
    mocap = _import_mocap_loader()
    raw = mocap.process_excel_sheet(str(path), sheet)
    if raw is None or len(raw) < 5:
        raise ValueError(f"Sheet {sheet!r} of {path.name} produced no usable frames")

    raw_time_native = raw["time"].to_numpy(dtype=np.float64)
    raw_time = raw_time_native - float(raw_time_native[0])
    raw_butt = raw[["mid_X", "mid_Y", "mid_Z"]].to_numpy(dtype=np.float64)
    raw_clubhead = raw[["club_X", "club_Y", "club_Z"]].to_numpy(dtype=np.float64)
    rotmats = np.empty((raw.shape[0], 3, 3), dtype=np.float64)
    for i in range(raw.shape[0]):
        rotmats[i] = _frame_to_quat(raw.iloc[i])
    raw_quat = rotmat_to_quat(rotmats)

    impact_raw = detect_impact_index(raw_time, raw_clubhead)
    sim_time, butt, clubhead, quat, impact_idx = resample_target(
        raw_time, raw_butt, raw_clubhead, raw_quat, impact_raw, opts
    )

    source = SourceProvenance(
        filename=path.name,
        format="excel",
        subject_id=sheet.split("_")[0],
        trial_id=sheet,
        sha256=_sha256_of(path),
    )
    logger.info(
        "Loaded ClubTarget from %s sheet %s: %d output samples (impact=%d)",
        path.name,
        sheet,
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
