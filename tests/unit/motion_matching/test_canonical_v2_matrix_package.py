"""Tests for canonical-v2 workbook matrix, #15725 package, #15837 cross-check (T2-WP7, #15787)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

from src.shared.python.motion_matching.club_only import canonical_v2_matrix as cm
from src.shared.python.motion_matching.club_only import workbook_identity as wi

pytestmark = pytest.mark.unit
_BASE: dict[str, Any] = dict.fromkeys(cm._KEYS, 0.001) | {
    "grip_closure_max_m": 0.003,
    "torque_headroom_fraction": 0.001,
    "cop_feasible": True,
}


def _inputs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build 4-cell matrix inputs plus tour_baseline across all replay badges."""
    pre, full = "pre-impact-replayed", "full-swing-replayed"
    cells = {
        s: _BASE | {"replay_badge": pre if "wiffle" in s else full}
        for s in wi.CANONICAL_TRIAL_SHEETS
    }
    return cells, _BASE | {"replay_badge": "kinematic-only"}


def _setup_workbooks(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Create temporary Club_Data/Wiffle workbooks with canonical sheets."""
    from src.motion_capture import capture_registry as cr

    paths: dict[str, Path] = {}
    for cid in ("club-workbook-main", "club-workbook-wiffle"):
        wb = openpyxl.Workbook()
        for i, s in enumerate(wi.CANONICAL_TRIAL_SHEETS):
            (wb.active if i == 0 else wb.create_sheet(s)).title = s  # type: ignore[union-attr]
        paths[cid] = root / f"{cid}.xlsx"
        wb.save(paths[cid])
        wb.close()
    monkeypatch.setattr(cr, "resolve_capture", lambda cid, **_: paths[cid])
    monkeypatch.setattr(wi, "verify_workbook_hash", lambda p, h: h)


def test_resolve_matrix_workbook_paths_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail closed when CAPTURE_DATA_DIR is unset or lacks private workbooks."""
    monkeypatch.delenv("CAPTURE_DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="CAPTURE_DATA_DIR"):
        cm.resolve_matrix_workbook_paths()
    monkeypatch.setenv("CAPTURE_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="club-workbook"):
        cm.resolve_matrix_workbook_paths()


def test_matrix_qualification_flagship_binding_and_15725_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve TW/GW x ProV1/Wiffle + Club_Data, bind #15814 TW, verify #15725."""
    _setup_workbooks(monkeypatch, tmp_path)
    cells, base = _inputs()
    q = cm.qualify_canonical_v2_matrix
    b = q(cell_metrics=cells, tour_baseline_metrics=base)
    assert b.schema_version == cm.MATRIX_BUNDLE_SCHEMA and b.governing_issue == 15787
    assert b.qualified and not b.blockers and b.flagship_trial_id == "TW_ProV1"
    assert set(b.workbook_paths) == {"club-workbook-main", "club-workbook-wiffle"}
    assert [t.trial_id for t in b.trials] == list(wi.CANONICAL_TRIAL_SHEETS)
    fb, tgt = b.flagship_model_binding, "full_body_mujoco_tiger_woods"
    assert fb["subject_label"] == "Rob Neal TW" and fb["skin_id"] == "tiger_woods"
    assert fb["target_model_id"] == tgt and fb["skin_issue_ref"] == "#15814"
    by_t = {t.trial_id: t for t in b.trials}
    assert by_t["TW_ProV1"].is_flagship and by_t["GW_ProV11"].ball_type == "ProV1"
    pkg, v = b.comparison_package, ["face_on", "down_the_line"]
    assert pkg.package_ref == "#15725" and len(pkg.package_sha256) == 64
    assert pkg.catalog["render_engine"] == "mujoco" and pkg.catalog["views"] == v
    p = pkg.as_dict()
    assert set(p["arm_summaries"]) == {"Wiffle", "ProV1", "tour_baseline"}
    badges = {"pre-impact-replayed", "full-swing-replayed", "kinematic-only"}
    assert {c.replay_badge for c in pkg.cells} == badges
    assert set(_BASE) | {"replay_badge"} <= set(p["cells"][0])
    again, jd = q(cell_metrics=cells, tour_baseline_metrics=base), json.dumps
    assert again == b and jd(again.as_dict(), sort_keys=True) == jd(
        b.as_dict(), sort_keys=True
    )


def test_15837_shaft_crosscheck_is_non_gating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Assert #15837 shaft-shape cross-check never flips qualification pass/fail."""
    _setup_workbooks(monkeypatch, tmp_path)
    angles, q = np.linspace(-60.0, 60.0, 51), cm.qualify_canonical_v2_matrix
    cc_fn = cm.compute_phase_normalized_shaft_crosscheck
    cc = cc_fn(angles, angles + 120.0)
    assert not cc.affects_pass_criteria and cc.issue_ref == "#15837"
    assert cc.shaft_angle_rmse_deg == pytest.approx(120.0)
    cells, base = _inputs()
    ok = q(cell_metrics=cells, tour_baseline_metrics=base, shaft_crosscheck=cc)
    assert ok.qualified and not ok.blockers
    cc_d = ok.comparison_package.as_dict()["non_gating_cross_check"]
    assert not cc_d["affects_pass_criteria"]
    cells["TW_ProV1"] |= {"cop_feasible": False, "torque_headroom_fraction": -0.05}
    z = cc_fn(angles, angles)
    fail = q(cell_metrics=cells, tour_baseline_metrics=base, shaft_crosscheck=z)
    b_txt = " ".join(fail.blockers)
    assert not fail.qualified and "cop_infeasible" in b_txt
    assert "torque_headroom_exhausted" in b_txt
    with pytest.raises(ValueError, match="pass criteria"):
        cm.ShaftShapeCrossCheck("TW_ProV1", 101, 5.0, 8.0, affects_pass_criteria=True)
