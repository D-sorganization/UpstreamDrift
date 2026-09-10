"""A simulation rollout reaches the existing mapping and model analysis UI."""

from pathlib import Path

import pytest

pytest.importorskip("h5py")

from src.motion_capture.reference.importers import (
    finish_motion_import,
    load_motion_draft,
)
from src.tools.capture_rig.reference_import import ReferenceMappingDialog
from src.tools.capture_rig.model_analysis_dialog import ModelAnalysisDialog
from tests.motion_capture.test_trace_reference_import import club_trace, trace_file
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_trace_mapping_opens_common_analysis(tmp_path: Path):
    _app()
    path = tmp_path / "trace.h5"
    trace_file(path, names='["clubhead"]')
    draft = load_motion_draft(path)
    mapping = ReferenceMappingDialog(draft)
    assert not mapping.units.isEnabled()
    assert mapping.model_identity.text() == "fixture-backend / qualified-model"
    mapping.confirm.setChecked(True)
    asset = finish_motion_import(draft, **mapping.options())
    mapping.close()
    analysis = ModelAnalysisDialog(asset, tmp_path / "analysis")
    analysis.geometry_controls.add_point()
    assert " m" in analysis.readout.value.text()
    assert analysis.save()
    analysis.close()
    assert (tmp_path / "analysis" / "model-analysis.json").is_file()


def test_declared_club_reaches_shared_appearance_controls(tmp_path: Path):
    _app()
    path = tmp_path / "club.h5"
    club_trace(path)
    draft = load_motion_draft(path)
    mapping = ReferenceMappingDialog(draft)
    assert mapping.edges.toPlainText() == "wrist, grip\ngrip, head"
    mapping.confirm.setChecked(True)
    asset = finish_motion_import(draft, **mapping.options())
    mapping.close()
    analysis = ModelAnalysisDialog(asset, tmp_path / "analysis")
    assert analysis.appearance.club.isEnabled()
    analysis.appearance.club.setChecked(False)
    assert not analysis.model_source.recipe.appearance.draw_club
    analysis.appearance.ellipsoids.setChecked(True)
    analysis.appearance.volume_alpha.setValue(0.37)
    analysis.spatial.mirror.setChecked(True)
    analysis.tool.setCurrentText("Line")
    analysis.add_center()
    analysis.geometry_controls.add_plane()
    assert analysis.save()
    analysis.close()
    reopened = ModelAnalysisDialog(asset, tmp_path / "analysis")
    assert not reopened.appearance.club.isChecked()
    assert reopened.appearance.ellipsoids.isChecked()
    assert reopened.appearance.volume_alpha.value() == pytest.approx(0.37)
    assert reopened.spatial.mirror.isChecked()
    assert reopened.model_source.recipe.registration.mirror_lateral
    assert reopened.model_source.recipe.asset.club_edges == ((1, 2),)
    assert len(reopened.canvas.layer.shapes) == 1
    assert len(reopened.geometry_controls.document.planes) == 1
    reopened.close()
