"""Equipment evidence travels with model output without changing arm constraints."""

import json
from pathlib import Path

import pytest

from src.motion_capture.rig.equipment import model_equipment_context
from src.tools.capture_rig.equipment import save_capture_club
from tests.motion_capture.reconstruct.model.test_session import (
    test_fit_session_model_writes_angles_report_and_landmarks as run_fit_fixture,
)
from tests.tools.capture_rig.test_equipment import capture_root, club, measured_length

pytestmark = pytest.mark.unit


def test_fit_saves_exact_selection_without_substituting_arm_lengths(
    tmp_path: Path,
) -> None:
    root = capture_root(tmp_path)
    item = club().model_copy(update={"overrides": (measured_length(37),)})
    selected = save_capture_club(root, item)
    run_fit_fixture(root)
    path = root / "model" / "fit_report.json"
    before = path.read_bytes()
    report = json.loads(before)
    provenance = report["provenance"]
    context = provenance["parameters"]["equipment"]
    assert context["selection"]["club_revision"] == selected.club_revision
    assert context["eligible_values"]["assembled_length_si"] == pytest.approx(0.9398)
    assert context["club_number"] == "7"
    assert context["constraints_applied"] == []
    assert report["lengths_m"]["forearm"] == 0.26
    assert any(i["path"] == "capture_club.json" for i in provenance["inputs"])
    save_capture_club(root, item.model_copy(update={"notes": "Changed after analysis"}))
    assert path.read_bytes() == before


def test_estimated_length_is_retained_but_withheld_from_default_physics(
    tmp_path: Path,
) -> None:
    root = capture_root(tmp_path)
    claim = measured_length(37).model_copy(update={"status": "estimated"})
    save_capture_club(root, club().model_copy(update={"overrides": (claim,)}))
    context = model_equipment_context(root)
    assert context["eligible_values"]["assembled_length_si"] is None
    assert "estimated" in context["withheld"]["assembled_length_si"]
    assert context["selection"]["club"]["overrides"][0]["value"] == 37
    assert model_equipment_context(tmp_path) is None
