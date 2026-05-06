"""Excel loader tests; integration tests rely on the real Wiffle workbook."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from src.shared.python.motion_matching import (
    AlignOptions,
    ClubTarget,
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


def test_excel_rejects_unknown_sheet() -> None:
    p = _excel_path()
    if p is None:
        pytest.skip("Wiffle_ProV1_club_3D_data.xlsx not present")
    from src.shared.python.core.contracts import PreconditionError

    with pytest.raises((ValueError, PreconditionError)):
        load_club_target_excel(p, "NOT_A_SHEET", AlignOptions())


def test_excel_rejects_missing_file() -> None:
    from src.shared.python.core.contracts import PreconditionError

    with pytest.raises((FileNotFoundError, ValueError, PreconditionError)):
        load_club_target_excel("does/not/exist.xlsx", "TW_ProV1", AlignOptions())


@pytest.mark.integration
def test_load_excel_TW_ProV1_succeeds() -> None:
    p = _excel_path()
    if p is None:
        pytest.skip("Wiffle_ProV1_club_3D_data.xlsx not present")
    target = load_club_target_excel(p, "TW_ProV1", AlignOptions())
    assert isinstance(target, ClubTarget)
    assert target.time.shape[0] == target.butt.shape[0]
    assert target.time.shape[0] == target.clubhead.shape[0]
    assert target.time.shape[0] == target.club_quat.shape[0]
    qnorms = np.linalg.norm(target.club_quat, axis=1)
    assert np.all(np.abs(qnorms - 1.0) < 1e-6)
    assert 1 <= target.impact_idx <= target.time.shape[0]


@pytest.mark.integration
def test_load_excel_inches_to_metres() -> None:
    p = _excel_path()
    if p is None:
        pytest.skip("Wiffle_ProV1_club_3D_data.xlsx not present")
    target = load_club_target_excel(p, "TW_ProV1", AlignOptions())
    radii = np.linalg.norm(target.clubhead, axis=1)
    # Plausibility: clubhead radius from world origin must be < 5 m once
    # converted to metres. (If unit conversion were skipped, the inches-scale
    # values would commonly exceed this.)
    assert np.all(radii < 5.0)


@pytest.mark.integration
def test_load_excel_quaternion_sign_canonicalised() -> None:
    p = _excel_path()
    if p is None:
        pytest.skip("Wiffle_ProV1_club_3D_data.xlsx not present")
    target = load_club_target_excel(p, "TW_ProV1", AlignOptions())
    # SLERP over a canonicalised raw series may produce a few negative-w samples
    # near antipodal interpolation segments, but the bulk should be positive.
    assert np.mean(target.club_quat[:, 0] >= 0.0) > 0.8


@pytest.mark.integration
def test_load_excel_sha256_matches_file() -> None:
    p = _excel_path()
    if p is None:
        pytest.skip("Wiffle_ProV1_club_3D_data.xlsx not present")
    target = load_club_target_excel(p, "TW_ProV1", AlignOptions())
    h = hashlib.sha256()
    with p.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    assert target.source.sha256 == h.hexdigest()
