"""Player lens-profile review must stay explicit and preserve all rig views."""

from pathlib import Path
import json

import pytest

from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.tools.capture_rig.calibration_dialog import (
    CalibrationDialog,
    CameraProfilePanel,
)
from src.tools.capture_rig.calibration_profiles import save_profile
from tests.tools.capture_rig.test_calibration_profiles import profile
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_profile_selection_requires_confirmation_and_edits_reset_it(
    tmp_path: Path,
) -> None:
    _app()
    saved = save_profile(tmp_path / "profiles.json", profile(tmp_path))
    binding = CameraBinding(
        view="front", serial="serial-123", mode=CaptureMode(width=1920, height=1200)
    )
    panel = CameraProfilePanel(binding, tmp_path / "profiles.json")
    panel.profiles.setCurrentIndex(1)
    assert panel.assignment().profile.profile_id == saved.profile_id
    assert not panel.assignment().settings_confirmed
    panel.confirmed.setChecked(True)
    assert panel.assignment().settings_confirmed
    panel.fields["zoom"].setText("Changed zoom")
    assert not panel.assignment().settings_confirmed
    panel.close()


def test_empty_profile_cannot_be_applied(tmp_path: Path) -> None:
    _app()
    panel = CameraProfilePanel(
        CameraBinding(view="front", serial="serial-123"), tmp_path / "profiles.json"
    )
    with pytest.raises(ValueError, match="Select"):
        panel.assignment()
    panel.close()


def test_reviewed_dialog_exports_selected_revision_without_overwriting_history(
    tmp_path: Path,
) -> None:
    _app()
    save_profile(tmp_path / "profiles.json", profile(tmp_path))
    before = (tmp_path / "profiles.json").read_bytes()
    binding = CameraBinding(
        view="front", serial="serial-123", mode=CaptureMode(width=1920, height=1200)
    )
    dialog = CalibrationDialog(RigPlan(name="test", cameras=(binding,)), tmp_path)
    dialog.panels[0].profiles.setCurrentIndex(1)
    dialog.panels[0].confirmed.setChecked(True)
    dialog._apply()
    assert dialog.output_path is not None
    payload = json.loads(dialog.output_path.read_text(encoding="utf-8"))
    assert payload["cameras"][0]["camera_id"] == "front"
    assert (tmp_path / "profiles.json").read_bytes() == before
    assert not dialog.recalibrate_requested
    dialog.close()


def test_repeat_exits_without_exporting_or_discarding_history(tmp_path: Path) -> None:
    _app()
    binding = CameraBinding(view="front", serial="serial-123")
    dialog = CalibrationDialog(RigPlan(name="test", cameras=(binding,)), tmp_path)
    dialog._repeat()
    assert dialog.recalibrate_requested
    assert dialog.output_path is None
    assert not (tmp_path / "profiles.json").exists()
    dialog.close()
