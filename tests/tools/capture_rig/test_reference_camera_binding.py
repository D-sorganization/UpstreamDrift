"""Native comparison uses actual calibrated camera records and stale-data checks."""

import json
from pathlib import Path

import pytest

from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from tests.motion_capture.rig.test_ingest import _bundle
from tests.motion_capture.test_reference_registration import two_camera_rig
from tests.tools.capture_rig.test_pane_layout import _app
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_dialog_saves_actual_camera_and_rejects_stale_projection(
    tmp_path: Path,
) -> None:
    app = _app()
    root = _bundle(tmp_path)
    record = two_camera_rig()[0].to_calibration().to_dict()
    record["camera_id"] = "a"
    record["image_size_px"] = [64, 48]
    record["intrinsics"] = {
        "matrix": [[40, 0, 32], [0, 40, 24], [0, 0, 1]],
        "distortion": [0.1, 0, 0, 0, 0],
    }
    path = root / "reconstruct" / "reconstruction.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"cameras": [record]}), encoding="utf-8")
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)
    try:
        assert dialog._camera is not None
        assert "Manual Alignment" in dialog.status_label.text()
        assert dialog.save()
        registration = dialog._session.registration
        assert registration.camera.distortion[0] == 0.1
        assert registration.asset_sha256
        assert not registration.is_calibrated
    finally:
        dialog.close()
        app.processEvents()
    record["intrinsics"]["matrix"][0][0] = 42
    path.write_text(json.dumps({"cameras": [record]}), encoding="utf-8")
    with pytest.raises(ValueError, match="camera changed"):
        ReferenceComparisonDialog(root, "a", library)
