"""Drawing edits over a comparison save and export through shared infrastructure."""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.motion_capture.coaching import Drawing, render_layer
from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.rig.edits import CropRect, SessionEdits, ViewEdit, save_edits
from src.tools.capture_rig.comparison_coaching_source import ComparisonCoachingSource
from src.tools.capture_rig.model_frame_source import ModelFrameSource
from src.tools.capture_rig.reference_rendering import ComparisonRenderContext
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion

pytestmark = pytest.mark.unit


def test_comparison_drawings_save_still_and_snapshot_video(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    motion = synthetic_motion()
    virtual = ModelFrameSource.from_motion(motion, size=(64, 48))
    recipe = virtual.recipe
    virtual.close()
    camera = recipe.camera.model_copy(update={"camera_id": "a"}).record()
    crop = CropRect(x=4, y=4, width=40, height=30)
    save_edits(root, SessionEdits(views={"a": ViewEdit(crop=crop)}))
    context = ComparisonRenderContext(
        "a", motion, recipe.registration, ComparisonLayer(), crop=crop
    )
    source = ComparisonCoachingSource(root, context, camera)
    drawings = source.drawings.with_shapes(
        (Drawing(kind="line", start=(0, 24), end=(63, 24)),)
    )
    source.save(drawings)
    expected = source.reader.finalize(
        render_layer(source.reader.read(0), drawings, 0), 0
    )
    still = tmp_path / "comparison.png"
    source.still(drawings, 0, still)
    np.testing.assert_array_equal(cv2.imread(str(still)), expected)
    metadata = json.loads(still.with_suffix(".json").read_text())
    assert metadata["pixel_grid"] == "original_uncropped"
    assert metadata["reference_asset"]["id"] == motion.id
    job = source.export_job(drawings)
    source.save(drawings.with_shapes(()))
    movie = tmp_path / "comparison-video.avi"
    job(movie, lambda: False, lambda done, total: None)
    exported = json.loads(movie.with_suffix(".json").read_text())
    assert len(exported["drawings"]["shapes"]) == 1
    assert exported["output_size_px"] == [40, 30]
    assert cv2.imread(str(still)).shape == (48, 64, 3)
    source.reader.close()
    reopened = ComparisonCoachingSource(root, context, camera)
    assert not reopened.drawings.shapes
    with pytest.raises(ValueError, match="another source"):
        reopened.save(drawings.model_copy(update={"view": "b"}))
    cancelled = tmp_path / "cancelled.avi"
    with pytest.raises(InterruptedError):
        reopened.export_job(reopened.drawings)(
            cancelled, lambda: True, lambda done, total: None
        )
    assert not cancelled.exists()
    (root / "intrinsics.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        reopened.save(reopened.drawings)
    reopened.reader.close()
