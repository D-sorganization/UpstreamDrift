"""Saved comparison state must survive independent control edits (#9879)."""

from pathlib import Path

import pytest

from src.motion_capture.reference.comparison import (
    MAX_COMPARISON_BYTES,
    ComparisonLayer,
    ComparisonSession,
    comparison_session_path,
    load_comparison_session,
    save_comparison_session,
)
from src.motion_capture.reference.registration import ReferenceRegistration
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def saved_session(root: Path, reference_id: str) -> ComparisonSession:
    return ComparisonSession(
        session_root=str(root),
        view="a",
        reference_id=reference_id,
        reference_kind="motion",
        registration=ReferenceRegistration.model_validate(
            {
                "reference_id": reference_id,
                "calibration_id": "fixture",
                "transform": {"translation_m": (1, 2, 3), "scale": 1.2},
                "time_mapping": {
                    "offset_s": 0.3,
                    "rate_scale": 1.5,
                    "event_anchors": {
                        "reference": {"impact": 1.0},
                        "scene": {"impact": 2.0},
                    },
                },
                "image_transform_2d": ((1, 0, 4), (0, 1, 5), (0, 0, 1)),
                "assumption_labels": ("Manual registration",),
            }
        ),
        layer=ComparisonLayer(opacity=0.4, draw_skeleton=False, draw_joints=False),
    )


def test_layer_update_preserves_saved_display_flags(tmp_path: Path) -> None:
    original = saved_session(tmp_path, synthetic_motion().id)
    changed = original.with_layer(opacity=0.8)
    assert changed.layer.model_dump() == original.layer.model_dump() | {"opacity": 0.8}


@pytest.mark.parametrize("view", ["../outside", "a/b", "a\\b", "C:escape", ""])
def test_comparison_path_rejects_non_filename_views(tmp_path: Path, view: str) -> None:
    with pytest.raises(ValueError):
        comparison_session_path(tmp_path, view, synthetic_motion().id)


def test_comparison_rejects_mismatched_registration(tmp_path: Path) -> None:
    original = saved_session(tmp_path, synthetic_motion().id)
    with pytest.raises(ValueError):
        original.changed(reference_id=synthetic_motion().id)


def test_comparison_path_rejects_non_uuid_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        comparison_session_path(tmp_path, "a", "../../outside")


def test_oversized_comparison_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (MAX_COMPARISON_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        load_comparison_session(path)


def test_control_edits_preserve_registration_and_switch_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard
    )
    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    for _ in range(2):
        library.save(synthetic_motion())
    expected = saved_session(root, library.list()[1].id)
    save_comparison_session(expected, root)
    dialog = ReferenceComparisonDialog(root, "a", library)
    try:
        dialog.asset_selector.setCurrentIndex(1)
        app.processEvents()
        assert dialog._session == expected
        dialog.offset_spin.setValue(0.5)
        assert dialog._session.registration.model_dump() == (
            expected.registration.model_dump()
            | {
                "time_mapping": expected.registration.time_mapping.model_dump()
                | {"offset_s": 0.5}
            }
        )
        dialog.scale_spin.setValue(1.8)
        assert dialog._session.registration.transform.translation_m == (1, 2, 3)
        assert dialog._session.registration.assumption_labels == (
            "Manual registration",
        )
    finally:
        dialog.close()


def test_corrupt_saved_comparison_is_not_silently_replaced(tmp_path: Path) -> None:
    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    asset = synthetic_motion()
    library.save(asset)
    path = comparison_session_path(root, "a", asset.id)
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="[Cc]omparison|JSON"):
        ReferenceComparisonDialog(root, "a", library)
    assert path.read_text(encoding="utf-8") == "{broken"
    app.processEvents()


def test_export_job_ownership_duplicate_guard_and_deferred_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event
    from time import monotonic
    from src.tools.capture_rig import reference_comparison as module

    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    started, release = Event(), Event()
    calls = []

    def export(*args, **kwargs):
        calls.append(args)
        started.set()
        if not release.wait(5):
            raise RuntimeError("Test did not release worker")
        if kwargs["options"].cancelled():
            raise InterruptedError("Comparison export cancelled")

    monkeypatch.setattr(module, "export_comparison_video", export)
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.show()
    try:
        dialog.exporter.start(tmp_path / "comparison.avi")
        assert started.wait(2)
        dialog.exporter.start(tmp_path / "duplicate.avi")
        assert len(calls) == 1
        assert dialog.exporter.busy
        dialog.reject()
        assert dialog.isVisible()
        assert dialog.exporter.busy
        release.set()
        deadline = monotonic() + 5
        while dialog.exporter.busy and monotonic() < deadline:
            app.processEvents()
        assert not dialog.exporter.busy
        assert not dialog.isVisible()
        assert "cancelled" in dialog.status_label.text()
    finally:
        release.set()
        worker = dialog.exporter._worker
        if worker:
            worker.requestInterruption()
            worker.wait(6000)
            app.processEvents()
        dialog.close()


def test_failed_save_does_not_start_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.capture_rig import reference_comparison as module

    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)

    def fail(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(module, "save_comparison_session", fail)
    try:
        dialog.exporter.start(tmp_path / "failed.avi")
        assert not dialog.exporter.busy
        assert dialog.exporter.button.isEnabled()
        assert dialog.status_label.text() == "Disk full"
    finally:
        dialog.close()
        app.processEvents()
