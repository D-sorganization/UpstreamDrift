"""Cover ``core.py`` xlsx loaders + adapters.

Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.tools.starting_pose_matcher import core
from src.tools.starting_pose_matcher.core import (
    MocapEvents,
    _clubtarget_to_dataframe,
    _safe,
    load_mocap_xlsx,
    read_event_header,
)


pytestmark = pytest.mark.unit


def _build_clubtarget(n: int = 8, *, scale: float = 1.0):
    """Build a synthetic ClubTarget for the dataframe adapter.

    Uses ``scale`` to push the shaft length above 1.4 (which triggers
    the cm/inches re-scaling branch) or below it.
    """
    from src.shared.python.motion_matching.target import ClubTarget, SourceProvenance

    src = SourceProvenance(
        filename="x", format="xlsx", subject_id="s", trial_id="t", sha256="0" * 64
    )
    return ClubTarget(
        time=np.linspace(0.0, 0.3, n),
        butt=np.zeros((n, 3)),
        clubhead=np.tile([scale, 0.0, 0.0], (n, 1)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        impact_idx=min(n - 1, n // 2),
        source=src,
    )


def test_clubtarget_to_dataframe_columns_present():
    df = _clubtarget_to_dataframe(_build_clubtarget(n=4))
    for col in (
        "time",
        "mid_X",
        "mid_Y",
        "mid_Z",
        "club_X",
        "club_Y",
        "club_Z",
        "club_Xx",
        "club_Yy",
        "club_Zz",
    ):
        assert col in df.columns


def test_clubtarget_to_dataframe_rotation_identity():
    df = _clubtarget_to_dataframe(_build_clubtarget(n=3))
    # Identity quaternion -> identity rotation.
    np.testing.assert_allclose(
        df[["club_Xx", "club_Yy", "club_Zz"]].iloc[0].to_numpy(), [1.0, 1.0, 1.0]
    )


def test_clubtarget_to_dataframe_scaling_branch():
    """When the median shaft length is > 1.4 m the adapter re-scales by
    ``CM_TO_M / inches_factor``."""
    target = _build_clubtarget(n=4, scale=2.0)  # shaft length 2.0 m > 1.4
    df = _clubtarget_to_dataframe(target)
    # After re-scaling, club_X is much smaller than the raw 2.0.
    assert abs(float(df["club_X"].iloc[0])) < 1.0


def test_clubtarget_to_dataframe_no_scaling_when_short():
    target = _build_clubtarget(n=4, scale=0.9)  # shaft 0.9 m < 1.4
    df = _clubtarget_to_dataframe(target)
    # No rescaling — value is preserved.
    assert float(df["club_X"].iloc[0]) == pytest.approx(0.9)


def test_safe_handles_index_and_keyerror():
    s = pd.Series([1.0, 2.0, 3.0])
    assert _safe(s, 0) == 1.0
    # Out-of-range -> default.
    assert _safe(s, 99, default=-1.0) == -1.0


def test_safe_handles_nan_and_returns_default():
    s = pd.Series([np.nan, 1.0])
    assert _safe(s, 0, default=42.0) == 42.0


def test_safe_handles_non_numeric_and_returns_default():
    s = pd.Series(["abc", 2.0])
    assert _safe(s, 0, default=7.0) == 7.0


def test_safe_returns_float():
    s = pd.Series([2])
    out = _safe(s, 0)
    assert isinstance(out, float)


def test_load_mocap_xlsx_delegates_to_clubtarget_loader():
    """Patch the canonical loader so we don't need a real xlsx file."""
    target = _build_clubtarget(n=5, scale=0.5)
    with patch.object(core, "load_club_target", return_value=target) as mock_loader:
        df = load_mocap_xlsx("/fake/path.xlsx", "Sheet1")
    mock_loader.assert_called_once()
    assert "time" in df.columns
    assert len(df) == 5


def test_read_event_header_passes_through():
    """Patch the shared loader and verify the header values land."""
    fake = SimpleNamespace(
        A_sample=1.0, T_sample=10.0, I_sample=20.0, F_sample=30.0, CHS_mph=85.0
    )
    with patch.object(core, "read_excel_event_markers", return_value=fake):
        ev = read_event_header("/fake/path.xlsx", "Sheet1")
    assert isinstance(ev, MocapEvents)
    assert ev.A_sample == 1.0
    assert ev.CHS_mph == 85.0
