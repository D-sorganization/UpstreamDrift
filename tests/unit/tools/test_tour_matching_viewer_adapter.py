"""Tests for Tour Matching Viewer embed adapter (Step 3)."""

from __future__ import annotations

import os
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.shared.python.launcher_embed import EmbedCapabilities, EmbeddableTool

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_adapter_satisfies_embeddable_tool_protocol() -> None:
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    adapter = _TourMatchingViewerEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)
    assert adapter.tool_id == "tour_matching_viewer"


def test_adapter_embed_capabilities() -> None:
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    adapter = _TourMatchingViewerEmbedAdapter()
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (900, 650)
    assert caps.requires_separate_qapplication is False


def test_adapter_lifecycle(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    adapter = _TourMatchingViewerEmbedAdapter()
    assert adapter.is_dirty() is False

    # Cleanup before handing out widgets is safe
    adapter.cleanup()

    parent = QWidget()
    widget = adapter.create_main_widget(parent)
    try:
        assert isinstance(widget, QWidget)
        assert widget.parent() is parent
    finally:
        adapter.cleanup()
        parent.deleteLater()


def test_import_registers_adapter_in_registry() -> None:
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )
    import src.tools.tour_matching_viewer  # noqa: F401

    assert "tour_matching_viewer" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("tour_matching_viewer")
    assert tool is not None
    assert tool.tool_id == "tour_matching_viewer"


def test_verified_simscape_bundle_uses_retained_model_and_status(
    qapp, tmp_path
) -> None:  # noqa: ANN001
    from pathlib import Path
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    root = Path(__file__).resolve().parents[3]
    manifest = (
        root
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    widget = TourMatchingViewerWidget()
    try:
        widget.load_file(manifest)
        assert widget._engine_name == "simscape"
        assert widget._spec is not None
        assert len(widget._spec["coordinate_order"]) == 27
        assert widget._replay is not None
        assert widget._replay.frame_count == 307

        assert "rejected" in widget._title_label.text().lower()
        assert "unavailable" in widget._title_label.text().lower()
        widget._slider.setValue(306)
        assert "0.850 s" in widget._frame_label.text()
        before = widget._replay
        invalid = tmp_path / "invalid.json"
        invalid.write_text('{"schema_version": "bad"}')
        with pytest.raises(ValueError):
            widget.load_file(invalid)
        assert widget._replay is before
    finally:
        widget.cleanup()
        widget.close()


def test_playback_follows_source_time_and_preserves_camera(qapp) -> None:  # noqa: ANN001
    from pathlib import Path
    from src.shared.python.golf_simulator import MonotonicReplayClock
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    now = [0.0]
    widget = TourMatchingViewerWidget()
    widget._clock = MonotonicReplayClock(time_fn=lambda: now[0])
    manifest = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    try:
        widget.load_file(manifest)
        widget._ax.view_init(elev=32, azim=121)
        widget.toggle_playback()
        now[0] = 0.5
        widget._on_timer_tick()
        assert widget._current_frame == 180
        assert widget._ax.elev == 32
        assert widget._ax.azim == 121
        widget.toggle_playback()
        now[0] = 4.0
        widget._on_timer_tick()
        assert widget._current_frame == 180
        widget._slider.setValue(72)
        widget.toggle_playback()
        now[0] = 4.1
        widget._on_timer_tick()
        assert 107 <= widget._current_frame <= 108
        now[0] = 5.0
        widget._on_timer_tick()
        assert widget._current_frame == 306
        assert not widget._is_playing
        assert not widget._timer.isActive()
        widget.toggle_playback()
        assert widget._current_frame == 0
    finally:
        widget.cleanup()
        widget.close()


def test_speed_controls_update_monotonic_clock(qapp) -> None:  # noqa: ANN001
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget()
    try:
        assert widget._clock.playback_rate == 1.0
        # 0.25x
        widget._speed_combo.setCurrentIndex(0)
        assert widget._clock.playback_rate == 0.25
        # 0.5x
        widget._speed_combo.setCurrentIndex(1)
        assert widget._clock.playback_rate == 0.5
        # 2.0x
        widget._speed_combo.setCurrentIndex(3)
        assert widget._clock.playback_rate == 2.0
    finally:
        widget.cleanup()
        widget.close()


def test_restart_control_resets_clock_and_frame(qapp) -> None:  # noqa: ANN001
    from pathlib import Path
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    manifest = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    widget = TourMatchingViewerWidget()
    try:
        widget.load_file(manifest)
        widget._slider.setValue(150)
        assert widget._current_frame == 150
        assert widget._clock.current_time_s > 0.0

        widget.restart_playback()
        assert widget._current_frame == 0
        assert widget._slider.value() == 0
        assert widget._clock.current_time_s == 0.0
    finally:
        widget.cleanup()
        widget.close()


def test_catalog_combo_discovers_and_loads_saved_run(qapp) -> None:  # noqa: ANN001
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget()
    try:
        # Check combo has discovered simscape-returned102
        assert widget._catalog_combo.count() >= 2
        items = [
            widget._catalog_combo.itemText(i)
            for i in range(widget._catalog_combo.count())
        ]
        assert any("simscape-returned102" in it for it in items)

        # Select index 1 (the registered entry)
        widget._catalog_combo.setCurrentIndex(1)
        assert widget._replay is not None
        assert widget._engine_name == "simscape"
        assert widget._candidate_hash == "simscape-returned102"
        assert widget._report_btn.isEnabled() is True
        assert widget._anim_btn.isEnabled() is True
        assert "Unavailable" in widget._effort_badge.text()
    finally:
        widget.cleanup()
        widget.close()


def test_render_modes_and_error_overlay(qapp) -> None:  # noqa: ANN001
    from pathlib import Path
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    manifest = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    widget = TourMatchingViewerWidget()
    try:
        widget.load_file(manifest)

        # In "Cylinders" mode: Poly3DCollection is created for body segments
        assert widget._render_mode_combo.currentText() == "Cylinders"
        has_poly = any(isinstance(c, Poly3DCollection) for c in widget._ax.collections)
        assert has_poly is True

        # Switch to "Line Skeleton" mode
        widget._render_mode_combo.setCurrentIndex(1)
        assert widget._render_mode_combo.currentText() == "Line Skeleton"
        has_lines = any(isinstance(c, Line3DCollection) for c in widget._ax.collections)
        assert has_lines is True

        # Error overlay check
        assert widget._error_overlay_check.isChecked() is True
        # Uncheck error overlay
        widget._error_overlay_check.setChecked(False)
        widget._error_overlay_check.setChecked(True)
    finally:
        widget.cleanup()
        widget.close()


def test_camera_presets_apply_view_angles(qapp) -> None:  # noqa: ANN001
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget()
    try:
        # Face-On
        widget._camera_combo.setCurrentIndex(1)
        assert widget._ax.elev == 0.0
        assert widget._ax.azim == 0.0

        # Down-the-Line
        widget._camera_combo.setCurrentIndex(2)
        assert widget._ax.elev == 0.0
        assert widget._ax.azim == -90.0

        # Top-Down
        widget._camera_combo.setCurrentIndex(3)
        assert widget._ax.elev == 90.0
        assert widget._ax.azim == 0.0
    finally:
        widget.cleanup()
        widget.close()


def test_inspection_dialogs_instantiate_cleanly(qapp) -> None:  # noqa: ANN001
    from pathlib import Path
    from src.tools.tour_matching_viewer.gui import (
        AnimationInspectorDialog,
        ReportInspectorDialog,
        TourMatchingViewerWidget,
    )

    manifest = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    widget = TourMatchingViewerWidget()
    try:
        widget.load_file(manifest)
        assert widget._report_data is not None
        assert widget._report_path is not None
        assert widget._animation_path is not None

        # Instantiate ReportInspectorDialog
        rep_dlg = ReportInspectorDialog(
            widget._report_data,
            run_id="simscape-returned102",
            report_path=widget._report_path,
        )
        assert "simscape-returned102" in rep_dlg.windowTitle()
        rep_dlg.close()

        # Instantiate AnimationInspectorDialog
        anim_dlg = AnimationInspectorDialog(
            widget._animation_path,
            run_id="simscape-returned102",
        )
        assert "simscape-returned102" in anim_dlg.windowTitle()
        anim_dlg.close()
    finally:
        widget.cleanup()
        widget.close()


def test_counterfactual_wrench_overlay_and_export_provenance(
    qapp,
    tmp_path,
    monkeypatch,  # noqa: ANN001
) -> None:
    import numpy as np
    from PyQt6 import QtWidgets
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from src.shared.python.motion_matching.counterfactual import (
        CutState,
        ForwardZTCFBranch,
    )
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget()
    try:
        assert widget._cf_badge.text() == "CF Wrench: —"
        assert widget._export_btn.isEnabled() is False

        # Build mock ForwardZTCFBranch
        cut = CutState(
            cut_time_s=0.5,
            coordinates={"j1": 0.1},
            rates={"j1": 0.0},
            parent_run_id="run_baseline",
        )
        t = np.array([0.5, 0.51, 0.52])
        branch = ForwardZTCFBranch(
            cut_state=cut,
            time_s=t,
            coordinate_names=("j1",),
            coordinates=np.array([[0.1], [0.12], [0.15]]),
            rates=np.array([[0.0], [2.0], [3.0]]),
            accelerations=np.array([[0.0], [20.0], [10.0]]),
            reaction_wrenches=np.array(
                [
                    [10.0, 0.0, 0.0, 0.0, 0.0, 5.0],
                    [15.0, 0.0, 0.0, 0.0, 0.0, 7.0],
                    [20.0, 0.0, 0.0, 0.0, 0.0, 9.0],
                ]
            ),
            twist=np.zeros((3, 6)),
            branch_id="branch_ztcf_001",
        )

        widget.load_counterfactual_trajectory(branch)
        assert widget._export_btn.isEnabled() is True
        assert "CF Wrench: |F|=10.0 N, |M|=5.0 Nm" in widget._cf_badge.text()

        # By default counterfactual check is unchecked
        assert widget._counterfactual_check.isChecked() is False
        coll_count_before = len(widget._ax.collections)

        # Check counterfactual overlay
        widget._counterfactual_check.setChecked(True)
        coll_count_after = len(widget._ax.collections)
        assert coll_count_after > coll_count_before
        assert any(isinstance(c, Line3DCollection) for c in widget._ax.collections)

        # Test export table with monkeypatched QFileDialog
        export_target = tmp_path / "exported_provenance.csv"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(export_target), "CSV Files (*.csv)"),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            lambda *args, **kwargs: None,
        )
        widget._on_export_table_clicked()
        assert export_target.exists()
        content = export_target.read_text(encoding="utf-8")
        assert "Fx_N" in content
        assert "branch_ztcf_001" in content
    finally:
        widget.cleanup()
        widget.close()
