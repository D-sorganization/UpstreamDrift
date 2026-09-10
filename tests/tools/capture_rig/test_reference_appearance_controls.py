"""Appearance controls edit saved comparison settings without losing other fields."""

import pytest
from pathlib import Path

from src.motion_capture.reference.comparison import ComparisonLayer
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_appearance_controls_preserve_colour_and_other_layer_settings() -> None:
    from src.tools.capture_rig.reference_appearance import MotionAppearanceControls

    app = _app()
    original = ComparisonLayer(opacity=0.6, line_width=5)
    controls = MotionAppearanceControls(original, has_club=True)
    controls.ellipsoids.setChecked(True)
    controls.club.setChecked(False)
    controls.volume_alpha.setValue(0.45)
    result = controls.updated(original)
    assert result.draw_ellipsoids and not result.draw_club
    assert result.ellipsoid_opacity == 0.45
    assert result.opacity == original.opacity and result.line_width == 5
    controls.close()
    app.processEvents()


def test_dialog_saves_appearance_and_flip_and_can_undo(tmp_path: Path) -> None:
    from src.motion_capture.reference.storage import ReferenceLibrary
    from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
    from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion
    from tests.motion_capture.rig.test_ingest import _bundle

    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.motion_appearance.ellipsoids.setChecked(True)
    dialog.undo_change()
    assert not dialog.motion_appearance.ellipsoids.isChecked()
    dialog.motion_appearance.ellipsoids.setChecked(True)
    dialog.motion_appearance.volume_alpha.setValue(0.4)
    dialog.spatial.mirror.setChecked(True)
    assert dialog.spatial.apply_placement()
    assert dialog.save()
    dialog.close()
    restored = ReferenceComparisonDialog(root, "a", library)
    assert restored.motion_appearance.ellipsoids.isChecked()
    assert restored.motion_appearance.volume_alpha.value() == 0.4
    assert restored.spatial.mirror.isChecked()
    restored.close()
    app.processEvents()
