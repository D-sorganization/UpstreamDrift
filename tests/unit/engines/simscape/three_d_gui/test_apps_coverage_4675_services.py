"""Coverage tests for apps/services and apps/core (issue #4675).

Test-only — no production code changes. Pure-data services and core
dataclasses; no Qt widgets except the loader-thread tests, which need
QApplication for QThread signal delivery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

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
# core/models.py
# --------------------------------------------------------------------------


def test_models_dataclasses_round_trip():
    from src.apps.core.models import (  # type: ignore
        AnalogData,
        C3DDataModel,
        C3DEvent,
        MarkerData,
    )

    pos = np.zeros((3, 3))
    md = MarkerData(name="m", position=pos, residuals=np.zeros(3))
    assert md.name == "m"
    assert md.position.shape == (3, 3)

    ad = AnalogData(name="a", values=np.zeros(3), unit="V")
    assert ad.unit == "V"

    ev = C3DEvent(label="L", time=0.5)
    assert ev.label == "L"
    assert ev.time == 0.5

    model = C3DDataModel(
        filepath="x.c3d",
        markers={"m": md},
        analog={"a": ad},
        events=[ev],
        raw_parameters={"X": 1},
    )
    assert model.marker_names() == ["m"]
    assert model.analog_names() == ["a"]
    assert model.raw_parameters == {"X": 1}


# --------------------------------------------------------------------------
# services/analysis.py
# --------------------------------------------------------------------------


def test_analysis_compute_marker_statistics_basic():
    from src.apps.services.analysis import compute_marker_statistics  # type: ignore

    t = np.linspace(0, 1, 11)
    pos = np.column_stack([t, np.zeros(11), np.zeros(11)])
    stats = compute_marker_statistics(t, pos)
    assert pytest.approx(stats["path_length"], rel=1e-6) == 1.0
    assert stats["max_speed"] > 0
    assert stats["mean_speed"] > 0


def test_analysis_too_few_frames_returns_nans():
    from src.apps.services.analysis import compute_marker_statistics  # type: ignore

    stats = compute_marker_statistics(np.array([0.0]), np.zeros((1, 3)))
    assert np.isnan(stats["path_length"])
    assert np.isnan(stats["max_speed"])
    assert np.isnan(stats["mean_speed"])


def test_analysis_no_time_returns_nans():
    from src.apps.services.analysis import compute_marker_statistics  # type: ignore

    stats = compute_marker_statistics(None, np.zeros((5, 3)))
    assert np.isnan(stats["path_length"])


def test_analysis_zero_dt_yields_nan_speed():
    from src.apps.services.analysis import compute_marker_statistics  # type: ignore

    t = np.array([0.0, 0.0, 0.0, 0.0])
    pos = np.zeros((4, 3))
    stats = compute_marker_statistics(t, pos)
    assert np.isnan(stats["max_speed"])
    assert np.isnan(stats["mean_speed"])


# --------------------------------------------------------------------------
# services/c3d_loader.py
# --------------------------------------------------------------------------


def test_c3d_loader_file_not_found():
    from src.apps.services.c3d_loader import load_c3d_file  # type: ignore

    with pytest.raises(FileNotFoundError):
        load_c3d_file("/nonexistent/path/foo.c3d")


def test_c3d_loader_build_helpers(monkeypatch, tmp_path):
    """Drive load_c3d_file with a stubbed C3DDataReader."""
    import pandas as pd

    from src.apps.services import c3d_loader  # type: ignore

    fake_path = tmp_path / "fake.c3d"
    fake_path.write_bytes(b"\x00")

    class _Meta:
        frame_rate = 100.0
        analog_rate = 1000.0
        frame_count = 5
        marker_count = 2
        units = "m"
        marker_labels = ["A", "B", "C"]  # C is missing -> empty branch
        analog_labels = ["EMG"]
        analog_units = ["V"]

        class _Ev:
            label = "Top"
            time = 0.1

        events = [_Ev()]

    class _Reader:
        def __init__(self, _path):
            pass

        def get_metadata(self):
            return _Meta()

        def points_dataframe(self, include_time=False):
            return pd.DataFrame(
                {
                    "marker": ["A", "A", "B", "B"],
                    "x": [0.0, 1.0, 0.0, 1.0],
                    "y": [0.0, 0.0, 0.0, 0.0],
                    "z": [0.0, 0.0, 0.0, 0.0],
                    "residual": [0.0, 0.0, 0.0, 0.0],
                }
            )

        def analog_dataframe(self, include_time=False):
            return pd.DataFrame(
                {
                    "channel": ["EMG", "EMG"],
                    "value": [0.1, 0.2],
                }
            )

        def _load(self):
            return {"parameters": {"POINT": {"UNITS": "m"}}}

    monkeypatch.setattr(c3d_loader, "C3DDataReader", _Reader)
    model = c3d_loader.load_c3d_file(str(fake_path))
    assert "A" in model.markers
    assert "C" in model.markers  # missing-label branch
    assert model.markers["C"].position.shape == (0, 3)
    assert model.analog["EMG"].unit == "V"
    assert model.point_rate == 100.0
    assert model.events[0].label == "Top"
    assert model.raw_parameters == {"POINT": {"UNITS": "m"}}


def test_c3d_loader_raw_params_failure_branch(monkeypatch, tmp_path):
    import pandas as pd

    from src.apps.services import c3d_loader  # type: ignore

    fake_path = tmp_path / "fake.c3d"
    fake_path.write_bytes(b"\x00")

    class _Meta:
        frame_rate = 0.0  # exercise frame_time None branch
        analog_rate = 0.0
        frame_count = 0
        marker_count = 0
        units = ""
        marker_labels = []
        analog_labels = []
        analog_units = []
        events = None  # falsy -> skip events loop

    class _Reader:
        def __init__(self, _path):
            pass

        def get_metadata(self):
            return _Meta()

        def points_dataframe(self, include_time=False):
            return pd.DataFrame(
                {"marker": [], "x": [], "y": [], "z": [], "residual": []}
            )

        def analog_dataframe(self, include_time=False):
            return pd.DataFrame()

        def _load(self):
            raise KeyError("parameters")

    monkeypatch.setattr(c3d_loader, "C3DDataReader", _Reader)
    model = c3d_loader.load_c3d_file(str(fake_path))
    assert model.raw_parameters is None
    assert model.point_time is None
    assert model.analog == {}


def test_c3d_loader_build_markers_validation():
    from src.apps.services.c3d_loader import _build_markers  # type: ignore

    with pytest.raises(ValueError):
        _build_markers(None, [])


def test_c3d_loader_build_analog_validation():
    from src.apps.services.c3d_loader import _build_analog  # type: ignore

    with pytest.raises(ValueError):
        _build_analog(None, MagicMock())


def test_c3d_loader_build_metadata_ui_validation():
    from src.apps.services.c3d_loader import _build_metadata_ui  # type: ignore

    with pytest.raises(ValueError):
        _build_metadata_ui(None, MagicMock())


def test_c3d_loader_build_metadata_ui_with_events():
    from src.apps.services.c3d_loader import _build_metadata_ui  # type: ignore

    class _Ev:
        label = "Top"
        time = 0.5

    class _Meta:
        frame_rate = 100.0
        analog_rate = 1000.0
        frame_count = 10
        marker_count = 5
        units = "m"
        events = [_Ev()]

    md = _build_metadata_ui("/path/to/file.c3d", _Meta())
    assert md["File"] == "file.c3d"
    assert "Events" in md
    assert "Top" in md["Events"]


# --------------------------------------------------------------------------
# services/loader_thread.py
# --------------------------------------------------------------------------


def test_loader_thread_validates_filepath():
    from src.apps.services.loader_thread import C3DLoaderThread  # type: ignore

    with pytest.raises(ValueError):
        C3DLoaderThread(None)  # type: ignore[arg-type]


def test_loader_thread_emits_failed_for_missing_file(qapp):
    from PyQt6.QtCore import QEventLoop, QTimer

    from src.apps.services.loader_thread import C3DLoaderThread  # type: ignore

    th = C3DLoaderThread("/no/such/file.c3d")
    received = {}

    def on_fail(msg):
        received["msg"] = msg

    th.failed.connect(on_fail)

    loop = QEventLoop()
    th.failed.connect(lambda _: loop.quit())
    th.loaded.connect(lambda _: loop.quit())
    QTimer.singleShot(5000, loop.quit)
    th.start()
    loop.exec()
    th.wait(2000)
    assert "File not found" in received.get("msg", "")


def test_loader_thread_emits_loaded(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QEventLoop, QTimer

    from src.apps.services import loader_thread as lt  # type: ignore

    fake = tmp_path / "ok.c3d"
    fake.write_bytes(b"\x00")
    sentinel = MagicMock(name="model")

    monkeypatch.setattr(lt, "load_c3d_file", lambda _p: sentinel)
    th = lt.C3DLoaderThread(str(fake))
    seen = {}
    th.loaded.connect(lambda m: seen.setdefault("m", m))
    loop = QEventLoop()
    th.loaded.connect(lambda _: loop.quit())
    th.failed.connect(lambda _: loop.quit())
    QTimer.singleShot(5000, loop.quit)
    th.start()
    loop.exec()
    th.wait(2000)
    assert seen.get("m") is sentinel


def test_loader_thread_emits_failed_for_various_exceptions(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QEventLoop, QTimer

    from src.apps.services import loader_thread as lt  # type: ignore

    fake = tmp_path / "ok.c3d"
    fake.write_bytes(b"\x00")

    for exc, fragment in (
        (ImportError("ezc3d"), "Missing dependency"),
        (KeyError("POINT"), "Corrupted"),
        (ValueError("bad data"), "Data inconsistency"),
        (RuntimeError("boom"), "Unexpected error"),
    ):
        captured = {}

        def _raise(_p, e=exc):
            raise e

        monkeypatch.setattr(lt, "load_c3d_file", _raise)
        th = lt.C3DLoaderThread(str(fake))
        th.failed.connect(lambda msg, c=captured: c.setdefault("m", msg))
        loop = QEventLoop()
        th.failed.connect(lambda _, _l=loop: _l.quit())
        QTimer.singleShot(5000, loop.quit)
        th.start()
        loop.exec()
        th.wait(2000)
        assert fragment in captured.get("m", ""), (fragment, captured)


# --------------------------------------------------------------------------
# services/segment_set_io.py
# --------------------------------------------------------------------------


def test_segment_spec_validation_errors():
    from src.apps.services.segment_set_io import SegmentSpec  # type: ignore

    with pytest.raises(ValueError):
        SegmentSpec(a="", b="x")
    with pytest.raises(ValueError):
        SegmentSpec(a="x", b="")
    with pytest.raises(ValueError):
        SegmentSpec(a="x", b="x")
    with pytest.raises(ValueError):
        SegmentSpec(a="x", b="y", geometry="bogus")
    with pytest.raises(ValueError):
        SegmentSpec(a="x", b="y", group="")
    with pytest.raises(TypeError):
        SegmentSpec(a="x", b="y", visible="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SegmentSpec(a="x", b="y", radius=0)
    with pytest.raises(ValueError):
        SegmentSpec(a="x", b="y", radius=True)


def test_segment_set_round_trip(tmp_path):
    from src.apps.services.segment_set_io import (  # type: ignore
        SegmentSet,
        SegmentSpec,
        default_segment_set_path,
        from_dict,
        load_segment_set,
        save_segment_set,
        to_dict,
    )

    segset = SegmentSet(
        segments=(
            SegmentSpec(a="A", b="B", geometry="line", group="g1"),
            SegmentSpec(a="C", b="D", geometry="cylinder", group="g2"),
        )
    )
    out = tmp_path / "sub" / "segs.json"
    save_segment_set(out, segset)
    assert out.is_file()

    reloaded = load_segment_set(out)
    assert reloaded.segments == segset.segments

    payload = to_dict(segset)
    assert payload["schema_version"] == 1
    assert from_dict(payload).segments == segset.segments

    p = default_segment_set_path()
    assert isinstance(p, Path)


def test_segment_set_io_error_paths(tmp_path):
    from src.apps.services.segment_set_io import (  # type: ignore
        from_dict,
        load_segment_set,
        save_segment_set,
        to_dict,
    )

    with pytest.raises(TypeError):
        to_dict({"segments": []})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        from_dict("not a dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        from_dict({"schema_version": 999, "segments": []})
    with pytest.raises(ValueError):
        from_dict({"schema_version": 1, "segments": "not-a-list"})
    with pytest.raises(ValueError):
        from_dict({"schema_version": 1, "segments": ["bad-entry"]})
    with pytest.raises(ValueError):
        save_segment_set(None, MagicMock())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        load_segment_set(None)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        load_segment_set(tmp_path / "missing.json")


# --------------------------------------------------------------------------
# services/marker_export.py
# --------------------------------------------------------------------------


def test_marker_export_validation_errors(tmp_path):
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = make_model(20)
    out = tmp_path / "x.csv"

    with pytest.raises(ValueError):
        export_markers(None, ["WaistLeft"], "x", None, "csv", out)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        export_markers(model, [], "x", None, "csv", out)
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "x", None, "bogus", out)
    with pytest.raises(ValueError):
        export_markers(model, ["NotAMarker"], "x", None, "csv", out)
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "q", None, "csv", out)
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], [], None, "csv", out)
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "x", (5, 100), "csv", out)
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "x", (5, 4), "csv", out)
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "x", (1, 2, 3), "csv", out)  # type: ignore[arg-type]


def test_marker_export_csv_sanitizes_formula(tmp_path):
    """Export a marker whose name starts with '=' to confirm sanitisation."""
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore
    from src.apps.services.marker_export import export_markers  # type: ignore

    danger_name = "=DANGER"
    pos = np.arange(30, dtype=float).reshape(10, 3)
    model = C3DDataModel(
        filepath="syn.c3d",
        markers={danger_name: MarkerData(name=danger_name, position=pos)},
        analog={},
        point_rate=100.0,
        analog_rate=0.0,
        point_time=np.arange(10) / 100.0,
        analog_time=None,
        metadata={"Units (POINT)": "m"},
        events=[],
    )
    out = tmp_path / "danger.csv"
    export_markers(model, [danger_name], "all", None, "csv", out)
    text = out.read_text(encoding="utf-8")
    assert "'=DANGER" in text


def test_marker_export_json_npz(tmp_path):
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = make_model(30, include_force_plate=False, include_analog=False)
    out_json = tmp_path / "m.json"
    export_markers(
        model,
        ["WaistLeft", "WaistRight"],
        "all",
        None,
        "json",
        out_json,
        include_residual=True,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "metadata" in payload and "data" in payload
    assert payload["metadata"]["units"] == "m"

    out_npz = tmp_path / "m.npz"
    export_markers(
        model,
        ["WaistLeft"],
        ("x", "y"),
        (5, 10),
        "npz",
        out_npz,
        include_residual=False,
    )
    with np.load(out_npz, allow_pickle=False) as data:
        assert "WaistLeft" in data.files
        assert data["WaistLeft"].shape == (6, 2)


def test_marker_export_default_frame_range(tmp_path):
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = make_model(15, include_force_plate=False, include_analog=False)
    out = tmp_path / "m.csv"
    export_markers(model, ["WaistLeft"], "all", None, "csv", out)
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 16


def test_marker_export_no_frames_raises(tmp_path):
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = C3DDataModel(
        filepath="x.c3d",
        markers={"M": MarkerData(name="M", position=np.empty((0, 3)))},
        analog={},
        point_rate=0.0,
        analog_rate=0.0,
        point_time=None,
        analog_time=None,
        metadata={},
        events=[],
    )
    with pytest.raises(ValueError):
        export_markers(model, ["M"], "all", None, "csv", tmp_path / "z.csv")
