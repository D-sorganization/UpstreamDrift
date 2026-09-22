"""Wiffle/ProV1 Excel loader.

Reuses ``mocap_data_loader.process_excel_sheet`` from the existing Motion
Capture Plotter rather than re-parsing the workbook.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.python.core.contracts import postcondition, precondition

from ..club_target import AlignOptions, ClubTarget, SourceProvenance
from ._align import detect_impact_index, resample_target
from ._quaternion import rotmat_to_quat
from .event_labels import parse_event_marker_cells

logger = logging.getLogger(__name__)

ALLOWED_SHEETS: frozenset[str] = frozenset(
    {"TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"}
)
INCHES_TO_METERS = 0.0254
_MOCAP_LOADER_RELATIVE = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
    "golf_gui/Motion Capture Plotter/mocap_data_loader.py"
)


@dataclass
class ExcelEventMarkers:
    """Event markers extracted from row-1 of Wiffle Excel sheets.

    Attributes:
        A_sample: Sample number (1-based) for Address event
        T_sample: Sample number (1-based) for Top of Backswing event
        I_sample: Sample number (1-based) for Impact event
        F_sample: Sample number (1-based) for Finish event
        CHS_mph: Clubhead speed in mph (NaN if missing)
    """

    A_sample: float = float("nan")
    T_sample: float = float("nan")
    I_sample: float = float("nan")
    F_sample: float = float("nan")
    CHS_mph: float = float("nan")

    def frame_for(self, label: str) -> int | None:
        """Convert 1-based sample number to 0-based frame index."""
        v = getattr(self, f"{label}_sample", float("nan"))
        if v != v:  # NaN check
            return None
        return max(0, int(v) - 1)


def _import_mocap_loader():
    """Side-load the legacy ``mocap_data_loader`` module by file path.

    The Motion Capture Plotter directory has spaces in its name, so it is not
    importable as a regular package. We therefore load it through ``importlib``.
    """
    if "mocap_data_loader" in sys.modules:
        return sys.modules["mocap_data_loader"]
    here = Path(__file__).resolve().parent
    candidates = [here / _MOCAP_LOADER_RELATIVE]
    for parent in here.parents:
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
        f"Could not locate mocap_data_loader.py (searched relative to {here})"
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


def _markers_from_parsed(parsed: dict[str, float]) -> ExcelEventMarkers:
    """Map shared event-label parse results onto ExcelEventMarkers."""
    return ExcelEventMarkers(
        A_sample=parsed.get("A", float("nan")),
        T_sample=parsed.get("T", float("nan")),
        I_sample=parsed.get("I", float("nan")),
        F_sample=parsed.get("F", float("nan")),
        CHS_mph=parsed.get("CHS", float("nan")),
    )


def _extract_event_markers(df: pd.DataFrame) -> ExcelEventMarkers:
    """Extract event markers from row-1 of the Excel sheet.

    The Wiffle/ProV1 format uses row-1 to store event markers in the pattern:
    ``A=<n>`` / ``A <n>`` … ``CHS=<mph>``. Labels are normalized through the
    shared event-label helper so bare and equals-suffixed forms agree.
    """
    if len(df) == 0:
        return ExcelEventMarkers()
    row = [df.iloc[0, c] for c in range(len(df.columns))]
    return _markers_from_parsed(parse_event_marker_cells(row))


def read_excel_event_markers(path: Path | str, sheet: str) -> ExcelEventMarkers:
    """Read only the event markers from row-1 of a Wiffle Excel sheet.

    This is a lightweight function for tools that need event markers
    without loading the full trajectory data. Accepts both ``A=`` and ``A``.
    """
    path = Path(path)
    try:
        row1 = pd.read_excel(path, sheet_name=sheet, header=None, nrows=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read event header: %s", exc)
        return ExcelEventMarkers()
    row = [row1.iat[0, c] for c in range(row1.shape[1])]
    return _markers_from_parsed(parse_event_marker_cells(row))


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
    # mocap_data_loader incorrectly assumes inches (0.0254), but Wiffle data is in cm.
    # We undo the inches conversion and apply the correct cm -> m conversion (0.01).
    correction = 0.01 / 0.0254
    raw_butt = raw[["mid_X", "mid_Y", "mid_Z"]].to_numpy(dtype=np.float64) * correction
    raw_clubhead = (
        raw[["club_X", "club_Y", "club_Z"]].to_numpy(dtype=np.float64) * correction
    )
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
