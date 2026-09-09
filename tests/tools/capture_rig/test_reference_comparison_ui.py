"""Deterministic integrated tests for reference comparison workspace, playback, and export (#9866)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.motion_capture.reference.comparison import (
    COMPARISON_EXPORT_SCHEMA,
    COMPARISON_SESSION_SCHEMA,
    ComparisonLayer,
    load_comparison_session,
)
from src.motion_capture.reference.model import (
    ReferenceMotion,
    ReferenceSource,
    ReferenceVideo,
)
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
    TimeMapping,
)
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from src.tools.capture_rig.reference_export import (
    ComparisonVideoExportOptions,
    draw_reference_overlay,
    export_comparison_video,
)
from src.tools.capture_rig.player import VideoReader
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def synthetic_motion() -> ReferenceMotion:
    return ReferenceMotion.model_validate(
        {
            "id": str(uuid4()),
            "title": "Synthetic Tour Benchmark",
            "source": ReferenceSource(
                path="synthetic.c3d", sha256="0" * 64, format="c3d"
            ),
            "source_units": "m",
            "source_axes": ("+X", "+Y", "+Z"),
            "source_names": ("pelvis", "wrist"),
            "joint_names": ("pelvis", "wrist"),
            "edges": ((0, 1),),
            "time_s": (0.0, 0.1, 0.2),
            "points_m": (
                ((0.0, 0.0, 1.0), (0.2, 0.0, 1.0)),
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5)),
                ((0.0, 0.0, 1.0), (-0.2, 0.0, 1.2)),
            ),
        }
    )


def test_reference_comparison_dialog_workflow_and_persistence(tmp_path: Path) -> None:
    app = _app()
    bundle_root = _bundle(tmp_path)
    ref_lib_dir = tmp_path / "references"
    library = ReferenceLibrary(ref_lib_dir)

    motion = synthetic_motion()
    library.save(motion)

    dialog = ReferenceComparisonDialog(bundle_root, "a", library)
    dialog.show()
    app.processEvents()

    assert len(dialog.assets) == 1
    assert dialog.asset_selector.count() == 1
    assert dialog.slider.maximum() > 0

    # Adjust controls
    dialog.visible_check.setChecked(True)
    dialog.opacity_spin.setValue(0.85)
    dialog.offset_spin.setValue(0.05)
    dialog.scale_spin.setValue(1.1)

    # Trigger save
    assert dialog.save() is True

    session_path = bundle_root / "comparisons" / f"a_{motion.id}.json"
    assert session_path.is_file()

    saved_sess = load_comparison_session(session_path)
    assert saved_sess.schema_version == COMPARISON_SESSION_SCHEMA
    assert saved_sess.view == "a"
    assert saved_sess.layer.opacity == 0.85
    assert saved_sess.registration is not None
    assert saved_sess.registration.time_mapping.offset_s == 0.05
    assert saved_sess.registration.transform.scale == 1.1

    dialog.close()


def test_export_comparison_video_and_reproducible_sidecar_parity(
    tmp_path: Path,
) -> None:
    bundle_root = _bundle(tmp_path)
    motion = synthetic_motion()
    out_video = tmp_path / "annotated_comparison.mp4"

    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_test",
        transform=ReferenceTransform(scale=1.0),
        time_mapping=TimeMapping(offset_s=0.0),
        is_calibrated=False,
    )
    layer = ComparisonLayer(colour="#00dcff", opacity=1.0, line_width=2)

    progress_events: list[tuple[int, int]] = []
    sidecar = export_comparison_video(
        bundle_root,
        "a",
        motion,
        reg,
        layer,
        out_video,
        options=ComparisonVideoExportOptions(
            progress=lambda done, total: progress_events.append((done, total)),
        ),
    )

    assert out_video.is_file()
    sidecar_path = out_video.with_suffix(".json")
    assert sidecar_path.is_file()

    assert len(progress_events) > 0
    assert progress_events[-1][0] == progress_events[-1][1]

    # Verify sidecar structure
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_data["schema_version"] == COMPARISON_EXPORT_SCHEMA
    assert sidecar_data["reference"]["id"] == motion.id
    assert sidecar_data["reference"]["is_3d"] is True
    assert sidecar_data["reference"]["is_calibrated"] is False
    assert sidecar_data["reference"]["missing_alignment_evidence"] is True
    assert sidecar_data["reference"]["alignment_status"] == "uncalibrated_3d_projection"

    # Frame parity check: inspect exported video decodability
    with VideoReader(out_video) as reader:
        assert reader.frame_count == sidecar_data["playback"]["frame_count"]
        first_frame = reader.read(0)
        assert first_frame is not None
        assert first_frame.shape[0] > 0
        assert first_frame.shape[1] > 0


def test_export_cancellation(tmp_path: Path) -> None:
    bundle_root = _bundle(tmp_path)
    motion = synthetic_motion()
    out_video = tmp_path / "cancelled_comparison.mp4"

    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_test",
    )
    layer = ComparisonLayer()

    with pytest.raises(InterruptedError):
        export_comparison_video(
            bundle_root,
            "a",
            motion,
            reg,
            layer,
            out_video,
            options=ComparisonVideoExportOptions(
                cancelled=lambda: True,
            ),
        )

    assert not out_video.exists()
    assert not out_video.with_suffix(".json").exists()
