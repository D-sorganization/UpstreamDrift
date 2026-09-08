"""Match panel, provenance tab, variant overlay and the annotate dialog (#9797, #9800, #9803)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication

from src.motion_capture.annotate import AnnotationSet
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.tools.capture_rig import gui
from src.tools.capture_rig.annotate_widget import AnnotateDialog, ImageCanvas
from src.tools.capture_rig.match_panel import (
    IMAGE_SPACE,
    MatchSelection,
    fit_model_args,
    reconstruct_args,
)
from src.tools.capture_rig.provenance_tab import ProvenanceTab, lineage_html
from src.tools.capture_rig.session import load_session
from tests.motion_capture.annotate.test_to_observations import _detector
from tests.tools.capture_rig.test_core import SIZE, _bundle
from tests.tools.capture_rig.test_overlay_render import _reconstructed_bundle

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None  # keep the application alive for the module


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def test_match_selection_and_argument_builders(tmp_path: Path) -> None:
    sel = MatchSelection(
        "pair_fd", ("face_on", "down_line"), "observations", "triangulate"
    )
    argv = reconstruct_args(
        tmp_path,
        sel,
        measurements=("shank=0.42",),
        cameras=tmp_path / "c.json",
        intrinsics=None,
    )
    assert argv[-4:] == ["--views", "face_on,down_line", "--variant", "pair_fd"]
    image = MatchSelection(
        "cam_face", ("face_on",), "observations_openpose", IMAGE_SPACE, ""
    )
    argv = fit_model_args(tmp_path, image, model="golfer", fit_lengths=False)
    assert "--from-views" in argv and argv[argv.index("--from-views") + 1] == "face_on"
    assert argv[argv.index("--observations") + 1] == "observations_openpose"
    with pytest.raises(Exception, match="image-space matches use fit-model"):
        reconstruct_args(
            tmp_path, image, measurements=("a=1",), cameras=None, intrinsics=tmp_path
        )
    with pytest.raises(Exception, match="at least two views"):
        MatchSelection("x", ("a",), "observations", "triangulate")
    with pytest.raises(Exception, match="needs a view"):
        MatchSelection("x", (), "observations", IMAGE_SPACE)
    with pytest.raises(Exception, match="invalid variant name"):
        MatchSelection("bad name", (), "observations", "triangulate")


def test_match_panel_reflects_the_session_and_its_variants(tmp_path: Path) -> None:
    _app()
    root, _ = _reconstructed_bundle(tmp_path)
    widget = gui.CaptureRigWidget()
    widget.capture.session_edit.setText(str(root))
    media = widget.refresh_session()
    assert media is not None and [v.name for v in media.variants] == [""]
    panel = widget.match
    assert set(panel.view_checks) == {"cam_a", "cam_b"}
    assert (
        panel.cameras_combo.count() == 1
        and panel.cameras_combo.itemText(0) == "(default)"
    )
    assert "(default): cam_a · triangulate" in panel.summary.text()
    # All views ticked means "use every camera": no --views argument.
    assert panel.selection().views == ()
    panel.view_checks["cam_b"].setChecked(False)
    panel.name_edit.setText("solo")
    panel.image_radio.setChecked(True)
    sel = panel.selection()
    assert sel.image_space and sel.views == ("cam_a",) and sel.cameras_from == ""
    argv = widget.command_for("fit_model")
    assert "--from-views" in argv and "--variant" in argv
    # Variant overlay: the default variant has joints, so it is offered and draws.
    box = widget.playback.variants
    assert box.selected() == () and set(box._checks) == {""}
    box.select(("",))
    tracks = box.tracks_for("cam_a")
    assert [t.kind for t in tracks] == ["joints"]
    widget.playback.show_frame(3)
    assert "frame 4/12" in widget.playback.status.text()
    widget.shutdown()


def test_provenance_tab_shows_the_lineage_of_a_selected_result(tmp_path: Path) -> None:
    _app()
    root, _ = _reconstructed_bundle(tmp_path)
    tab = ProvenanceTab()
    tab.show_path(root, Path("variants/index.json"))
    text = tab.browser.toPlainText()
    assert "variants/index.json" in text and "variants-index/1.0.0" in text
    tab.show_path(root, Path("model/joint_angles.json"))
    assert "does not exist" in tab.browser.toPlainText()
    assert "<ol>" in lineage_html([])
    widget = gui.CaptureRigWidget()
    widget.capture.session_edit.setText(str(root))
    widget.refresh_session()
    widget.swing_table.fill({"rms_px": 1.0}, root / "variants" / "index.json")
    widget.swing_table.selectRow(0)
    assert widget.results.currentWidget() is widget.provenance
    assert "variants/index.json" in widget.provenance.browser.toPlainText()
    widget.shutdown()


def test_image_canvas_maps_clicks_back_to_image_pixels() -> None:
    _app()
    canvas = ImageCanvas()
    canvas.resize(640, 400)
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    canvas.set_image(image)
    for zoom in (1.0, 2.5):
        canvas.set_zoom(zoom)
        dw, dh = canvas._drawn
        x0 = (canvas.width() - dw) / 2
        y0 = (canvas.height() - dh) / 2
        u = x0 + (100 + 0.5) * dw / SIZE[0]
        v = y0 + (40 + 0.5) * dh / SIZE[1]
        point = canvas.image_point_from_widget(QPoint(int(round(u)), int(round(v))))
        assert point is not None
        assert abs(point[0] - 100) < 0.5 * SIZE[0] / dw + 0.5
        assert abs(point[1] - 40) < 0.5 * SIZE[1] / dh + 0.5
    assert canvas.image_point_from_widget(QPoint(0, 0)) is None or canvas.zoom > 1
    with pytest.raises(Exception, match="zoom"):
        canvas.set_zoom(0)


def test_annotate_dialog_records_clicks_skips_and_saves(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    video = load_session(root).view("cam_a").playable
    assert video is not None
    dialog = AnnotateDialog(
        root, "cam_a", video, joints=("nose", "left_wrist"), stride=5, annotator="t"
    )
    prompt = dialog.guide.prompt()
    assert prompt is not None and (prompt.frame, prompt.joint) == (0, "nose")
    assert "Frame 0 · nose" in dialog.banner.text()
    dialog._on_click(100.0, 40.0)
    dialog.act("skip")
    assert dialog.guide.frame == 5 and dialog.dirty
    dialog.act("next_frame")
    assert dialog.guide.frame == 10
    dialog.act("back")
    assert dialog.guide.frame == 5
    path = dialog.save()
    assert not dialog.dirty
    saved = AnnotationSet.load(path)
    assert saved.point(0, "nose").x_px == 100.0 and saved.is_skipped(0, "left_wrist")
    assert saved.base_set is None
    dialog.reader.close()


def test_annotate_dialog_edit_mode_corrects_detector_points(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    (root / "observations").mkdir()
    base = root / "observations" / "cam_a.json"
    base.write_text(json.dumps(_detector(12)), encoding="utf-8")
    video = load_session(root).view("cam_a").playable
    assert video is not None
    dialog = AnnotateDialog(
        root, "cam_a", video, joints=("nose",), base_set="observations", base_file=base
    )
    assert dialog.base is not None and "detector 0.90" in dialog.banner.text()
    assert "click to correct" in dialog.banner.text()
    dialog._on_click(50.0, 60.0)  # frame 0 corrected
    dialog.act("skip")  # frame 1 rejected
    dialog.act("accept_frame")  # frame 2 kept as detected
    assert dialog.guide.frame == 3
    path = dialog.save()
    saved = AnnotationSet.load(path)
    assert saved.base_set == "observations"
    assert saved.point(0, "nose").x_px == 50.0
    assert saved.is_skipped(1, "nose") and not saved.has_entry(2, "nose")
    dialog.reader.close()
    # The tile opens the same dialog for the player's view and set.
    widget = gui.CaptureRigWidget()
    widget.capture.session_edit.setText(str(root))
    widget.refresh_session()
    built = widget.annotate_dialog()
    assert built is not None and built.store.base_set == "observations"
    assert "annotate" in widget.enabled_actions()
    built.reader.close()
    widget.shutdown()
    assert len(JOINT_NAMES) == 15
