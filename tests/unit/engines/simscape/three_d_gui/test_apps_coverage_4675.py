"""Coverage tests for apps/ui/* and apps/c3d_viewer.py (issue #4675).

Test-only — no production code changes. Qt-offscreen tab smoke tests
plus the C3DViewerMainWindow drag/drop and dialog plumbing. Pure-data
service tests live in test_apps_coverage_4675_services.py.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ._apps_coverage_helpers import make_model

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


# --------------------------------------------------------------------------
# ui/widgets/mpl_canvas.py
# --------------------------------------------------------------------------


def test_mpl_canvas_basic(qapp):
    from src.apps.ui.widgets.mpl_canvas import MplCanvas  # type: ignore

    c = MplCanvas(None, width=4, height=3, dpi=80)
    ax = c.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    c.clear_axes()
    assert len(c.fig.axes) == 0


# --------------------------------------------------------------------------
# ui/tabs/analog_plot_tab.py
# --------------------------------------------------------------------------


def test_analog_plot_tab(qapp):
    from src.apps.ui.tabs.analog_plot_tab import AnalogPlotTab  # type: ignore

    tab = AnalogPlotTab()
    tab.update_from_model(None)
    tab.update_plot()

    model = make_model(40)
    tab.update_from_model(model)
    assert tab.list_analog.count() > 0
    tab.update_plot()
    tab.list_analog.clearSelection()
    tab.update_plot()


def test_analog_plot_tab_missing_time(qapp):
    """analog_time=None branch."""
    from src.apps.core.models import AnalogData, C3DDataModel  # type: ignore
    from src.apps.ui.tabs.analog_plot_tab import AnalogPlotTab  # type: ignore

    model = C3DDataModel(
        filepath="x.c3d",
        markers={},
        analog={"E": AnalogData(name="E", values=np.zeros(5), unit="V")},
        point_rate=100.0,
        analog_rate=100.0,
        point_time=None,
        analog_time=None,
        metadata={},
        events=[],
    )
    tab = AnalogPlotTab()
    tab.update_from_model(model)
    tab.list_analog.setCurrentRow(0)
    tab.update_plot()


# --------------------------------------------------------------------------
# ui/tabs/marker_plot_tab.py
# --------------------------------------------------------------------------


def test_marker_plot_tab_components(qapp):
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore

    tab = MarkerPlotTab()
    tab.update_from_model(None)
    tab.update_plot()

    model = make_model(40)
    tab.update_from_model(model)
    assert tab.list_markers.count() > 0
    for idx in range(5):
        tab.combo_component.setCurrentIndex(idx)
        tab.update_plot()

    tab.list_markers.clearSelection()
    tab.update_plot()


def test_marker_plot_tab_missing_time(qapp):
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore

    model = C3DDataModel(
        filepath="x.c3d",
        markers={"M": MarkerData(name="M", position=np.zeros((5, 3)))},
        analog={},
        point_rate=100.0,
        analog_rate=0.0,
        point_time=None,
        analog_time=None,
        metadata={},
        events=[],
    )
    tab = MarkerPlotTab()
    tab.update_from_model(model)
    tab.list_markers.setCurrentRow(0)
    tab.update_plot()


# --------------------------------------------------------------------------
# ui/tabs/analysis_tab.py
# --------------------------------------------------------------------------


def test_analysis_tab(qapp):
    from src.apps.ui.tabs.analysis_tab import AnalysisTab  # type: ignore

    tab = AnalysisTab()
    tab.update_from_model(None)
    tab.update_panel()

    model = make_model(30)
    tab.update_from_model(model)
    assert tab.combo_marker_analysis.count() > 0
    tab.button_recompute_stats.click()
    text = tab.text_analysis.toPlainText()
    assert "Marker:" in text


def test_analysis_tab_single_frame(qapp):
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore
    from src.apps.ui.tabs.analysis_tab import AnalysisTab  # type: ignore

    model = C3DDataModel(
        filepath="x.c3d",
        markers={"M": MarkerData(name="M", position=np.zeros((1, 3)))},
        analog={},
        point_rate=100.0,
        analog_rate=0.0,
        point_time=np.array([0.0]),
        analog_time=None,
        metadata={},
        events=[],
    )
    tab = AnalysisTab()
    tab.update_from_model(model)
    tab.update_panel()


def test_analysis_tab_no_time(qapp):
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore
    from src.apps.ui.tabs.analysis_tab import AnalysisTab  # type: ignore

    model = C3DDataModel(
        filepath="x.c3d",
        markers={"M": MarkerData(name="M", position=np.zeros((5, 3)))},
        analog={},
        point_rate=100.0,
        analog_rate=0.0,
        point_time=None,
        analog_time=None,
        metadata={},
        events=[],
    )
    tab = AnalysisTab()
    tab.update_from_model(model)
    tab.update_panel()
    assert "No marker" in tab.text_analysis.toPlainText()


# --------------------------------------------------------------------------
# ui/tabs/force_plot_tab.py
# --------------------------------------------------------------------------


def test_force_plot_tab_full_flow(qapp):
    from src.apps.ui.tabs.force_plot_tab import ForcePlotTab  # type: ignore

    tab = ForcePlotTab()
    tab.update_from_model(None)

    empty = make_model(20, include_analog=False, include_force_plate=False)
    tab.update_from_model(empty)
    assert "No force-plate" in tab.status_label.text()

    model = make_model(40, include_analog=True, include_force_plate=True)
    tab.update_from_model(model)
    assert tab.plate_combo.count() > 0
    for i in range(tab.component_combo.count()):
        tab.component_combo.setCurrentIndex(i)
    tab.show_cop_checkbox.setChecked(False)
    tab.show_cop_checkbox.setChecked(True)


def test_force_plot_tab_no_force_plate_channels(qapp):
    from src.apps.core.models import AnalogData, C3DDataModel  # type: ignore
    from src.apps.ui.tabs.force_plot_tab import ForcePlotTab  # type: ignore

    model = C3DDataModel(
        filepath="x.c3d",
        markers={},
        analog={"EMG1": AnalogData(name="EMG1", values=np.zeros(5), unit="V")},
        point_rate=100.0,
        analog_rate=100.0,
        point_time=np.zeros(5),
        analog_time=np.zeros(5),
        metadata={},
        events=[],
    )
    tab = ForcePlotTab()
    tab.update_from_model(model)
    assert "No force-plate" in tab.status_label.text()


def test_force_plot_tab_cop_no_time(qapp):
    """Force plate data with analog_time=None goes through plot-line branch."""
    from src.apps.core.models import AnalogData, C3DDataModel  # type: ignore
    from src.apps.ui.tabs.force_plot_tab import ForcePlotTab  # type: ignore

    n = 20
    analog = {}
    for axis, base in zip("xyz", (10.0, 20.0, 100.0), strict=True):
        analog[f"F{axis}1"] = AnalogData(
            name=f"F{axis}1",
            values=base + np.arange(n) * 0.1,
            unit="N",
        )
        analog[f"M{axis}1"] = AnalogData(
            name=f"M{axis}1",
            values=np.arange(n) * 0.05,
            unit="N.m",
        )
    model = C3DDataModel(
        filepath="x.c3d",
        markers={},
        analog=analog,
        point_rate=100.0,
        analog_rate=100.0,
        point_time=np.arange(n) / 100.0,
        analog_time=None,
        metadata={},
        events=[],
    )
    tab = ForcePlotTab()
    tab.update_from_model(model)


# --------------------------------------------------------------------------
# ui/tabs/overview_tab.py
# --------------------------------------------------------------------------


def test_overview_tab_full(qapp):
    from src.apps.ui.tabs.overview_tab import OverviewTab  # type: ignore

    tab = OverviewTab()
    tab.update_from_model(None)
    assert tab.label_file.text() == "No file loaded"

    model = make_model(50)
    tab.update_from_model(model)
    assert "Loaded file" in tab.label_file.text()
    assert tab.tree_group_count >= 1
    assert tab.tree_node_count > tab.tree_group_count


def test_overview_tab_no_raw_params_fallback(qapp):
    from src.apps.ui.tabs.overview_tab import OverviewTab  # type: ignore

    model = make_model(10, include_raw_params=False)
    object.__setattr__(model, "metadata", {"k1": "v1", "k2": "v2"})
    tab = OverviewTab()
    tab.update_from_model(model)
    assert tab._prov_form.rowCount() >= 1


def test_overview_helpers_format_value():
    from src.apps.ui.tabs.overview_tab import (  # type: ignore
        _format_value,
        _is_metadata_internal_key,
        _scalar_value,
    )

    assert _is_metadata_internal_key("__type__")
    assert not _is_metadata_internal_key("UNITS")
    assert _scalar_value({"value": 3}) == 3
    assert _scalar_value(np.array(7)) == 7
    arr_one = np.array([5])
    assert _scalar_value(arr_one) == 5
    assert _scalar_value(np.array([1, 2, 3])).shape == (3,)
    assert _scalar_value([42]) == 42

    assert "shape=" in _format_value(np.arange(20))
    s = _format_value(list(range(10)))
    assert "len=" in s
    assert _format_value([1, 2]) == "[1, 2]"
    assert _format_value(b"hi") == "hi"
    assert _format_value("plain") == "plain"
    assert _format_value(42) == "42"


# --------------------------------------------------------------------------
# ui/tabs/segments_tab.py
# --------------------------------------------------------------------------


def test_segments_tab_basic(qapp):
    from src.apps.services.segment_set_io import SegmentSpec  # type: ignore
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    tab = SegmentsTab()
    tab.update_from_model(None)

    model = make_model(20)
    tab.update_from_model(model)
    n0 = len(tab.segments)

    tab.add_segment(SegmentSpec(a="WaistLeft", b="WaistRight", group="custom"))
    assert len(tab.segments) == n0 + 1

    tab.set_segment_visibility(0, False)
    assert tab.segments[0].visible is False
    tab.set_segment_geometry(0, "cylinder")
    assert tab.segments[0].geometry == "cylinder"

    with pytest.raises(ValueError):
        tab.set_segment_geometry(999, "line")
    with pytest.raises(ValueError):
        tab.set_segment_visibility(-1, True)
    with pytest.raises(ValueError):
        tab.set_segment_geometry(0, "bogus")
    with pytest.raises(TypeError):
        tab.add_segment("not-a-spec")  # type: ignore[arg-type]

    tab._on_reset_clicked()
    tab._delete_row(0)
    tab._delete_row(999)

    tab._on_visible_toggled(999, True)
    tab._on_endpoint_changed(999, "a", "x")
    tab._on_geometry_changed(999, "line")
    tab._on_group_changed(999, "g")

    tab._on_endpoint_changed(0, "a", "WaistRight")
    tab._on_geometry_changed(0, "cylinder")
    tab._on_group_changed(0, "   ")
    assert tab.segments[0].group == "auto"

    tab._on_endpoint_changed(0, "b", tab.segments[0].a)


def test_segments_tab_save_load(qapp, tmp_path, monkeypatch):
    from PyQt6 import QtWidgets

    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    tab = SegmentsTab()
    model = make_model(20)
    tab.update_from_model(model)

    save_path = tmp_path / "segs.json"

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(save_path), ""),
    )
    tab._on_save_clicked()
    assert save_path.is_file()
    tab._on_export_clicked()

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(save_path), ""),
    )
    tab._on_load_clicked()

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: ("", ""),
    )
    tab._on_load_clicked()
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: ("", ""),
    )
    tab._on_save_clicked()
    tab._on_export_clicked()

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(bad), ""),
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    tab._on_load_clicked()


def test_segments_tab_add_dialog(qapp, monkeypatch):
    from PyQt6 import QtWidgets

    from src.apps.ui.tabs import segments_tab as st  # type: ignore

    tab = st.SegmentsTab()

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    tab._on_add_clicked()

    model = make_model(20)
    tab.update_from_model(model)

    class _FakeDlg:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def selected_spec(self):
            from src.apps.services.segment_set_io import SegmentSpec  # type: ignore

            return SegmentSpec(a="WaistLeft", b="WaistRight", group="ok")

    monkeypatch.setattr(st, "_AddSegmentDialog", _FakeDlg)
    n0 = len(tab.segments)
    tab._on_add_clicked()
    assert len(tab.segments) == n0 + 1

    class _FakeDlgNone(_FakeDlg):
        def selected_spec(self):
            return None

    monkeypatch.setattr(st, "_AddSegmentDialog", _FakeDlgNone)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    tab._on_add_clicked()

    class _FakeDlgReject(_FakeDlg):
        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(st, "_AddSegmentDialog", _FakeDlgReject)
    tab._on_add_clicked()


def test_add_segment_dialog_selected_spec(qapp):
    from src.apps.ui.tabs.segments_tab import _AddSegmentDialog  # type: ignore

    dlg = _AddSegmentDialog(["A", "B", "C"], "myGroup")
    spec = dlg.selected_spec()
    assert spec is not None
    assert spec.a == "A" and spec.b == "B"

    dlg2 = _AddSegmentDialog(["A"], "")
    assert dlg2.selected_spec() is None


# --------------------------------------------------------------------------
# ui/dialogs/export_markers_dialog.py
# --------------------------------------------------------------------------


def test_export_markers_dialog(qapp, monkeypatch, tmp_path):
    from PyQt6 import QtWidgets

    from src.apps.ui.dialogs.export_markers_dialog import (  # type: ignore
        ExportMarkersDialog,
        _is_club_marker,
    )

    assert _is_club_marker("Marker_1:2:Club")
    assert not _is_club_marker("WaistLeft")

    with pytest.raises(ValueError):
        ExportMarkersDialog(None)  # type: ignore[arg-type]

    model = make_model(20)
    dlg = ExportMarkersDialog(model)
    assert dlg.list_markers.count() == len(model.marker_names())

    dlg.radio_x.setChecked(True)
    assert dlg._selected_components() == ("x",)
    dlg.radio_y.setChecked(True)
    assert dlg._selected_components() == ("y",)
    dlg.radio_z.setChecked(True)
    assert dlg._selected_components() == ("z",)
    dlg.radio_all.setChecked(True)
    assert dlg._selected_components() == ("x", "y", "z")

    dlg.radio_csv.setChecked(True)
    assert dlg._selected_format() == "csv"
    assert dlg._default_extension() == "csv"
    dlg.radio_json.setChecked(True)
    assert dlg._selected_format() == "json"
    dlg.radio_npz.setChecked(True)
    assert dlg._selected_format() == "npz"

    dlg._set_selection(lambda _n: True)
    assert all(
        dlg.list_markers.item(i).isSelected() for i in range(dlg.list_markers.count())
    )
    dlg._set_selection(lambda _n: False)
    assert dlg._selected_markers() == []

    dlg._select_body()

    dlg._set_selection(lambda _n: False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    dlg._on_accept()
    assert dlg._chosen_path is None

    dlg._set_selection(lambda _n: True)
    out = tmp_path / "x.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(out), ""),
    )
    dlg._on_accept()
    assert dlg._chosen_path == str(out)

    params = dlg.export_params()
    assert params is not None
    assert params["path"] == str(out)
    assert params["fmt"] in {"csv", "json", "npz"}

    dlg2 = ExportMarkersDialog(model)
    dlg2._set_selection(lambda _n: True)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: ("", ""),
    )
    dlg2._on_accept()
    assert dlg2.export_params() is None


def test_export_markers_dialog_no_point_time(qapp):
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore
    from src.apps.ui.dialogs.export_markers_dialog import (  # type: ignore
        ExportMarkersDialog,
    )

    model = C3DDataModel(
        filepath="x.c3d",
        markers={"M": MarkerData(name="M", position=np.zeros((3, 3)))},
        analog={},
        point_rate=0.0,
        analog_rate=0.0,
        point_time=None,
        analog_time=None,
        metadata={},
        events=[],
    )
    dlg = ExportMarkersDialog(model)
    assert dlg.spin_end.maximum() == 0


# --------------------------------------------------------------------------
# c3d_viewer.py — main window
# --------------------------------------------------------------------------


def test_c3d_viewer_main_window_basic(qapp, monkeypatch):
    from PyQt6 import QtWidgets

    from src.apps.c3d_viewer import C3DViewerMainWindow  # type: ignore

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "about", lambda *a, **k: None)
    win = C3DViewerMainWindow()
    assert win.windowTitle() == "C3D Motion Analysis Viewer"
    assert win.tabs.count() == 7

    with pytest.raises(ValueError):
        win._update_ui_state(None)  # type: ignore[arg-type]

    win._update_ui_state(True)
    assert win.tabs.isEnabled()
    win._update_ui_state(False)

    model = make_model(20)
    win.model = model
    win._populate_ui_with_model()
    assert win.action_export_markers.isEnabled()

    win._on_load_success(model)
    with pytest.raises(ValueError):
        win._on_load_success(None)  # type: ignore[arg-type]

    win._on_load_failure("oops")
    with pytest.raises(ValueError):
        win._on_load_failure(None)  # type: ignore[arg-type]

    win._on_load_finished()


def test_c3d_viewer_about_and_drag(qapp, monkeypatch):
    from PyQt6 import QtCore, QtGui, QtWidgets

    from src.apps.c3d_viewer import C3DViewerMainWindow  # type: ignore

    win = C3DViewerMainWindow()

    monkeypatch.setattr(QtWidgets.QMessageBox, "about", lambda *a, **k: None)
    win.show_about_dialog()

    with pytest.raises(ValueError):
        win.dragEnterEvent(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        win.dropEvent(None)  # type: ignore[arg-type]

    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile("/tmp/x.c3d")])
    drag_event = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(drag_event)

    mime2 = QtCore.QMimeData()
    mime2.setUrls([QtCore.QUrl.fromLocalFile("/tmp/x.txt")])
    drag_event2 = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.DropAction.CopyAction,
        mime2,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(drag_event2)

    mime3 = QtCore.QMimeData()
    drag_event3 = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.DropAction.CopyAction,
        mime3,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(drag_event3)


def test_c3d_viewer_load_path_security_blocked(qapp, monkeypatch):
    from PyQt6 import QtWidgets

    from src.apps.c3d_viewer import C3DViewerMainWindow  # type: ignore

    win = C3DViewerMainWindow()

    with pytest.raises(ValueError):
        win.load_c3d_file_from_path(None)  # type: ignore[arg-type]

    import shared.python.security.security_utils as sec  # type: ignore

    def _bad(*a, **k):
        raise ValueError("denied")

    monkeypatch.setattr(sec, "validate_path", _bad)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    win.load_c3d_file_from_path("/etc/shadow")


def test_c3d_viewer_open_file_cancel(qapp, monkeypatch):
    from PyQt6 import QtWidgets

    from src.apps.c3d_viewer import C3DViewerMainWindow  # type: ignore

    win = C3DViewerMainWindow()
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: ("", "")
    )
    win.open_c3d_file()


def test_c3d_viewer_export_markers_dialog(qapp, monkeypatch, tmp_path):
    from PyQt6 import QtWidgets

    from src.apps.c3d_viewer import C3DViewerMainWindow  # type: ignore

    win = C3DViewerMainWindow()

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    win._export_markers_dialog()

    model = make_model(20)
    win.model = model

    out_path = tmp_path / "x.csv"

    class _FakeDlg:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def export_params(self):
            return {
                "marker_names": ["WaistLeft"],
                "components": "all",
                "frame_range": None,
                "fmt": "csv",
                "path": str(out_path),
                "include_time": True,
                "include_residual": False,
            }

    from src.apps.ui.dialogs import export_markers_dialog as emd  # type: ignore

    monkeypatch.setattr(emd, "ExportMarkersDialog", _FakeDlg)
    win._export_markers_dialog()
    assert out_path.is_file()

    class _RejDlg(_FakeDlg):
        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(emd, "ExportMarkersDialog", _RejDlg)
    win._export_markers_dialog()

    class _NoneDlg(_FakeDlg):
        def export_params(self):
            return None

    monkeypatch.setattr(emd, "ExportMarkersDialog", _NoneDlg)
    win._export_markers_dialog()

    class _RaisingDlg(_FakeDlg):
        def export_params(self):
            return {
                "marker_names": ["NotAMarker"],
                "components": "all",
                "frame_range": None,
                "fmt": "csv",
                "path": str(out_path),
                "include_time": True,
                "include_residual": False,
            }

    monkeypatch.setattr(emd, "ExportMarkersDialog", _RaisingDlg)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    win._export_markers_dialog()


# --------------------------------------------------------------------------
# ui/tabs/viewer_3d_tab.py — additional CSV export branch
# --------------------------------------------------------------------------


def test_viewer_3d_tab_export_selected_markers_csv(qapp, tmp_path):
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    tab = Viewer3DTab()

    with pytest.raises(ValueError):
        tab.export_selected_markers_csv(str(tmp_path / "x.csv"))
    with pytest.raises(ValueError):
        tab.export_selected_markers_csv("")

    model = make_model(15)
    tab.update_from_model(model)
    out = tmp_path / "trajs.csv"
    tab.export_selected_markers_csv(str(out))
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "frame,time_s" in text
