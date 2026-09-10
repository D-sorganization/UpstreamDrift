"""The existing video drawing interface can analyze a model-only source."""

from pathlib import Path

import pytest

from src.tools.capture_rig.model_analysis_dialog import ModelAnalysisDialog
from tests.motion_capture.test_reference_registration import sample_motion
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_model_analysis_reuses_drawings_appearance_and_world_references(
    tmp_path: Path,
) -> None:
    _app()
    asset = sample_motion()
    dialog = ModelAnalysisDialog(asset, tmp_path / "analysis")
    dialog.tool.setCurrentText("Line")
    dialog.add_center()
    dialog.appearance.ellipsoids.setChecked(True)
    dialog.geometry_controls.add_plane()
    assert dialog.model_source.dirty
    assert dialog.save()
    assert not dialog.model_source.dirty
    dialog.close()
    reopened = ModelAnalysisDialog(asset, tmp_path / "analysis")
    assert len(reopened.canvas.layer.shapes) == 1
    assert reopened.appearance.ellipsoids.isChecked()
    assert len(reopened.geometry_controls.document.planes) == 1
    reopened.geometry_controls.opacity.setValue(0.6)
    assert reopened.isWindowModified()
    assert reopened.save()
    assert not list(tmp_path.rglob("recordings.json"))
    reopened.close()
