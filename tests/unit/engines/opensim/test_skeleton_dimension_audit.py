"""Unit tests for OpenSim skeleton dimension audit and prescribed replay gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.motion_matching import (
    skeleton_dimension_audit as sda,
)

pytestmark = [pytest.mark.unit]
_ROOT = Path(__file__).resolve().parents[4]
_MODELS = _ROOT / "src/engines/physics_engines/opensim/models"
_SCALED = _MODELS / "golf_humanoid_scaled.osim"
_ANTHRO = _MODELS / "generated/full_body_anthro_driver.osim"


def _fixture() -> dict[str, Any]:
    """Build deterministic reference and near-exact replay trajectory fixtures."""
    z, ang = np.zeros((10, 3)), np.zeros((10, 2))
    lm, rlm = (
        {"pelvis": z, "hand_l": z + 0.5},
        {"pelvis": z + 1e-5, "hand_l": z + 0.5 + 2e-5},
    )
    return {
        "ref_lm": lm,
        "rep_lm": rlm,
        "ref_ang": ang,
        "rep_ang": ang + 2e-4,
        "ref_club": z,
        "rep_club": z + 1e-5,
        "grip": np.full((10, 2), 3e-4),
    }


def _run_replay(d: dict[str, Any], cap: Any = None) -> sda.PrescribedReplayParityReport:
    """Invoke prescribed kinematic replay gate on a fixture dictionary."""
    return sda.evaluate_prescribed_kinematic_replay(
        d["ref_lm"],
        d["rep_lm"],
        d["ref_ang"],
        d["rep_ang"],
        (d["ref_club"], d["rep_club"]),
        d["grip"],
        capture_a_marker_residuals_m=cap,
        tolerances=sda.PrescribedReplayTolerances(),
    )


def test_audit_valid_golf_humanoid_scaled_model() -> None:
    """Valid scaled model reports counts, joints, grips, and reference mass checks."""
    r = sda.audit_osim_skeleton_dimensions(_SCALED)
    assert isinstance(r, sda.OsimSkeletonAuditReport) and r.passed
    assert not r.diagnostic and not r.violations and r.model_name == "golf_humanoid"
    assert (r.body_count, r.coordinate_count, r.muscle_count) == (23, 39, 0)
    assert r.total_mass_kg == pytest.approx(79.69, abs=0.05)
    t = r.body_map["torso"]
    assert isinstance(t, sda.BodyDimensionRecord)
    assert t.mass_kg == pytest.approx(28.2658, abs=1e-4)
    assert (t.parent_joint_name, t.parent_body_name) == ("back", "pelvis")
    assert t.segment_length_m > 0.1 and r.body_map["humerus_r"].segment_length_m > 0.4
    assert len(r.joint_axes["acromial_r"]) == 3 and len(r.joint_axes["elbow_r"]) == 1
    assert "club_grip_offset" in r.grip_frames and "hand_l_grip_offset" in r.grip_frames
    assert "hand_r_grip_offset" in r.grip_frames and "club_head_offset" in r.grip_frames

    ok = sda.audit_osim_skeleton_dimensions(
        _SCALED, reference_masses={"pelvis": 12.4088, "Club": 0.313}
    )
    assert ok.passed and not ok.mass_mismatches
    bad = sda.audit_osim_skeleton_dimensions(
        _SCALED, reference_masses={"pelvis": 14.0, "missing": 1.0}
    )
    assert not bad.passed and "pelvis" in bad.mass_mismatches
    assert {"mass_mismatch:pelvis", "missing_reference_body:missing"}.issubset(
        bad.violations
    )


def test_audit_fails_closed_on_malformed_xml_and_missing_file(tmp_path: Path) -> None:
    """Malformed anthro_driver.osim, broken XML, and missing paths fail closed."""
    rb = sda.audit_osim_skeleton_dimensions(_ANTHRO)
    assert (
        not rb.passed
        and rb.body_count == 0
        and isinstance(rb.diagnostic, sda.OsimXmlDiagnostic)
    )
    assert (rb.diagnostic.line, rb.diagnostic.column) == (
        1,
        69332,
    ) and "xml_parse_error" in rb.violations
    broken = tmp_path / "broken.osim"
    broken.write_text("<OpenSimDocument>\n  <Model>\n    <BodySet\n", encoding="utf-8")
    syn = sda.audit_osim_skeleton_dimensions(broken)
    assert not syn.passed and isinstance(syn.diagnostic, sda.OsimXmlDiagnostic)
    assert syn.diagnostic.line and syn.diagnostic.line >= 3 and syn.diagnostic.column
    assert not sda.audit_osim_skeleton_dimensions(tmp_path / "none.osim").passed


def test_evaluate_prescribed_replay_passes_and_separates_capture_a_rms() -> None:
    """Passing replay satisfies all parity gates while keeping capture-A RMS separate."""
    r = _run_replay(_fixture(), cap=np.full((10, 38), 0.032))
    assert (
        isinstance(r, sda.PrescribedReplayParityReport)
        and r.passed
        and not r.violations
    )
    assert r.max_landmark_error_m <= 1e-4 and r.max_angular_error_rad <= 1e-3
    assert r.max_clubhead_error_m <= 1e-4 and r.max_grip_closure_m <= 1e-3
    assert r.capture_a_marker_fit_rms_m == pytest.approx(0.032, rel=1e-6)


@pytest.mark.parametrize(
    ("key", "val", "expected"),
    [
        (
            "rep_lm",
            {"pelvis": np.zeros((10, 3)), "hand_l": np.ones((10, 3))},
            "landmark_error_exceeded",
        ),
        ("rep_ang", np.full((10, 2), 10.0), "angular_error_exceeded"),
        ("rep_club", np.full((10, 3), 10.0), "clubhead_trajectory_exceeded"),
        ("grip", np.full((10, 2), 1.2e-3), "grip_closure_exceeded"),
        ("rep_club", np.full((10, 3), np.nan), "non_finite_values:clubhead"),
    ],
)
def test_evaluate_prescribed_replay_fails_closed_on_gate_violations(
    key: str, val: Any, expected: str
) -> None:
    """Exceeding any parity gate or supplying non-finite data fails closed."""
    d = _fixture()
    d[key] = val
    r = _run_replay(d)
    assert not r.passed and expected in r.violations
