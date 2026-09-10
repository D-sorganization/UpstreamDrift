"""Both analysis surfaces use a live, scene-bound metric readout."""

import pytest

from src.tools.capture_rig.reference_readout import ReferenceReadout
from tests.motion_capture.test_reference_measurements import scene
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_readout_tracks_time_selection_and_scene_identity():
    _app()
    motion, registration, geometry = scene()
    panel = ReferenceReadout()
    panel.set_context(motion, registration, geometry, 0, "scene")
    assert "-5.0000 m" in panel.value.text()
    panel.set_context(motion, registration, geometry, 0.1, "scene")
    assert "Unavailable" in panel.value.text()
    panel.set_context(motion, registration, geometry, 0.2, "scene")
    panel.reference.setCurrentIndex(1)
    assert "8.3066 m" in panel.value.text()
    panel.set_context(motion, registration, geometry, 0, "other")
    assert "scene" in panel.value.text().lower()
    assert "8.3066" not in panel.value.text()
    panel.close()


def test_model_playback_updates_shared_readout(tmp_path):
    from src.tools.capture_rig.model_analysis_dialog import ModelAnalysisDialog

    _app()
    motion, _, _ = scene()
    dialog = ModelAnalysisDialog(motion, tmp_path / "model")
    dialog.geometry_controls.add_plane()
    assert " m" in dialog.readout.value.text()
    dialog.slider.setValue(1)
    assert "Unavailable" in dialog.readout.value.text()
    dialog.save()
    dialog.close()


def test_comparison_playback_updates_shared_readout(tmp_path):
    from src.motion_capture.reference.storage import ReferenceLibrary
    from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
    from tests.motion_capture.rig.test_ingest import _bundle

    _app()
    motion, _, _ = scene()
    library = ReferenceLibrary(tmp_path / "references")
    library.save(motion)
    dialog = ReferenceComparisonDialog(_bundle(tmp_path), "a", library)
    dialog.geometry_controls.add_plane()
    assert " m" in dialog.readout.value.text()
    dialog.slider.setValue(1)
    assert "Unavailable" in dialog.readout.value.text()
    dialog.save()
    dialog.close()
