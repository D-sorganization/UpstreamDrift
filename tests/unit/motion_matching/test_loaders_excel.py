"""Excel loader tests; integration tests rely on the real Wiffle workbook."""

from __future__ import annotations


import pytest
from src.shared.python.motion_matching import (
    AlignOptions,
    load_club_target_excel,
)
from src.shared.python.motion_matching.loaders.excel import (
    ALLOWED_SHEETS,
    INCHES_TO_METERS,
)

from ._fixtures import repo_root

EXCEL_RELATIVE = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
    "golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"
)


def _excel_path():
    p = repo_root() / EXCEL_RELATIVE
    return p if p.is_file() else None


def test_inches_to_metres_constant_matches_legacy_loader() -> None:
    assert pytest.approx(0.0254) == INCHES_TO_METERS


def test_allowed_sheets_set() -> None:
    assert "TW_ProV1" in ALLOWED_SHEETS
    assert "TW_wiffle" in ALLOWED_SHEETS
    assert "GW_wiffle" in ALLOWED_SHEETS
    assert "GW_ProV11" in ALLOWED_SHEETS


def test_excel_rejects_missing_file() -> None:
    from src.shared.python.core.contracts import PreconditionError

    with pytest.raises((FileNotFoundError, ValueError, PreconditionError)):
        load_club_target_excel("does/not/exist.xlsx", "TW_ProV1", AlignOptions())
