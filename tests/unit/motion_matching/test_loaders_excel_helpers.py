"""Coverage tests for private helpers in ``loaders.excel``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.shared.python.motion_matching.loaders.excel import (
    ExcelEventMarkers,
    _extract_event_markers,
    _frame_to_quat,
    read_excel_event_markers,
)


def test_event_markers_frame_for_returns_zero_based() -> None:
    """Pin: ``frame_for`` translates 1-based sample to 0-based frame."""
    ev = ExcelEventMarkers(I_sample=10.0)
    assert ev.frame_for("I") == 9


def test_event_markers_frame_for_nan_returns_none() -> None:
    """Pin: NaN sample => None (event missing)."""
    ev = ExcelEventMarkers()  # all NaN
    assert ev.frame_for("I") is None


def test_event_markers_frame_for_clamps_negative() -> None:
    """Pin: a 0-or-negative 1-based sample clamps to frame 0."""
    ev = ExcelEventMarkers(I_sample=0.0)
    assert ev.frame_for("I") == 0


def test_extract_event_markers_parses_row0() -> None:
    """Pin: row-0 ``A=<n>`` pattern is parsed into the matching field."""
    df = pd.DataFrame(
        [
            ["A", 5, "T", 100, "I", 250, "F", 400, "CHS", 95.0, "extra"],
            [None] * 11,
        ]
    )
    ev = _extract_event_markers(df)
    assert ev.A_sample == 5.0
    assert ev.I_sample == 250.0
    assert ev.CHS_mph == 95.0


def test_extract_event_markers_skips_unknown_label() -> None:
    """Pin: cells that aren't recognized labels are skipped."""
    df = pd.DataFrame([["zzz", 9, "I", 100]])
    ev = _extract_event_markers(df)
    assert ev.I_sample == 100.0
    assert np.isnan(ev.A_sample)


def test_extract_event_markers_skips_nan_value() -> None:
    """Pin: a NaN value cell after a recognized label is skipped."""
    df = pd.DataFrame([["I", float("nan")]])
    ev = _extract_event_markers(df)
    assert np.isnan(ev.I_sample)


def test_extract_event_markers_skips_unparseable() -> None:
    """Pin: non-numeric value cells are silently skipped (no raise)."""
    df = pd.DataFrame([["I", "not-a-number"]])
    ev = _extract_event_markers(df)
    assert np.isnan(ev.I_sample)


def test_read_excel_event_markers_missing_file(tmp_path: Path) -> None:
    """Pin: missing file is handled by returning empty markers (warns)."""
    out = read_excel_event_markers(tmp_path / "missing.xlsx", sheet="x")
    assert isinstance(out, ExcelEventMarkers)
    assert np.isnan(out.A_sample)


def test_frame_to_quat_identity() -> None:
    """Pin: identity direction-cosine row gives identity rotation matrix."""
    row = pd.Series(
        {
            "club_Xx": 1.0,
            "club_Yx": 0.0,
            "club_Zx": 0.0,
            "club_Xy": 0.0,
            "club_Yy": 1.0,
            "club_Zy": 0.0,
            "club_Xz": 0.0,
            "club_Yz": 0.0,
            "club_Zz": 1.0,
        }
    )
    rot = _frame_to_quat(row)
    assert np.allclose(rot, np.eye(3))


def test_read_excel_event_markers_synthetic_workbook(tmp_path: Path) -> None:
    """Pin: a real ``A=5 I=100`` row-1 is read correctly."""
    pytest.importorskip("openpyxl")
    p = tmp_path / "wb.xlsx"
    df = pd.DataFrame([["A", 5, "I", 100, "CHS", 90.0]])
    df.to_excel(p, sheet_name="s", header=False, index=False)
    ev = read_excel_event_markers(p, sheet="s")
    assert ev.A_sample == 5.0
    assert ev.I_sample == 100.0
    assert ev.CHS_mph == 90.0
