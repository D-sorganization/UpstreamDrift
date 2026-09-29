"""Comparison controls and headless export use the same saved world references."""

from pathlib import Path

import pytest

from src.motion_capture.coaching.geometry_storage import load_geometry
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from src.tools.capture_rig.reference_export import _render_recipe
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_comparison_saves_scene_geometry_and_reopens(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "library")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.geometry_controls.add_plane()
    assert len(dialog.geometry_controls.document.planes) == 1
    assert dialog.save()
    assert len(load_geometry(root).planes) == 1
    context, _ = _render_recipe(
        root,
        "a",
        dialog._current_asset,
        dialog._session.registration,
        dialog._session.layer,
        dialog.reader,
    )
    assert context.geometry == load_geometry(root)
    dialog.close()
    reopened = ReferenceComparisonDialog(root, "a", library)
    assert reopened.geometry_controls.document == load_geometry(root)
    reopened.close()
