"""Tests for selective marker export (CSV / JSON / NPZ)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest


def _make_model(n_frames: int = 100):
    from src.apps.core.models import C3DDataModel, MarkerData  # type: ignore

    rng = np.random.default_rng(42)
    names = ["WaistLeft", "WaistRight", "LKneeOut", "RKneeOut", "Marker_1:1:Club"]
    markers = {}
    for i, n in enumerate(names):
        pos = rng.normal(size=(n_frames, 3)) + i
        markers[n] = MarkerData(name=n, position=pos)
    return C3DDataModel(
        filepath="synthetic.c3d",
        markers=markers,
        analog={},
        point_rate=100.0,
        analog_rate=0.0,
        point_time=np.arange(n_frames) / 100.0,
        analog_time=None,
        metadata={"Units (POINT)": "m"},
        events=[],
    )


def test_csv_x_only_subset_frame_range(tmp_path) -> None:
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = _make_model(100)
    out = tmp_path / "out.csv"
    export_markers(
        model,
        marker_names=["WaistLeft", "WaistRight"],
        components=("x",),
        frame_range=(10, 20),
        fmt="csv",
        path=out,
        include_time=True,
        include_residual=False,
    )
    with out.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["frame", "time_s", "marker", "x"]
    # 11 frames * 2 markers = 22 data rows.
    assert len(rows) == 1 + 22
    # Confirm a value matches the source.
    sample = [r for r in rows[1:] if r[0] == "10" and r[2] == "WaistLeft"][0]
    assert pytest.approx(float(sample[3]), rel=1e-6) == float(
        model.markers["WaistLeft"].position[10, 0]
    )


def test_json_metadata_block(tmp_path) -> None:
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = _make_model(50)
    out = tmp_path / "out.json"
    export_markers(
        model,
        marker_names=["WaistLeft"],
        components="all",
        frame_range=None,
        fmt="json",
        path=out,
    )
    with out.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "metadata" in payload and "data" in payload
    assert payload["metadata"]["point_rate_hz"] == pytest.approx(100.0)
    assert payload["metadata"]["frame_range"] == [0, 49]
    assert payload["metadata"]["markers"] == ["WaistLeft"]
    # Synthetic model points to a non-existent file so sha256 should be None.
    assert payload["metadata"]["sha256"] is None
    assert len(payload["data"]) == 50


def test_npz_round_trip(tmp_path) -> None:
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = _make_model(30)
    out = tmp_path / "out.npz"
    export_markers(
        model,
        marker_names=["WaistLeft", "LKneeOut"],
        components="all",
        frame_range=(5, 15),
        fmt="npz",
        path=out,
    )
    with np.load(out, allow_pickle=False) as arr:
        assert "WaistLeft" in arr.files
        assert "LKneeOut" in arr.files
        assert arr["WaistLeft"].shape == (11, 3)
        meta = json.loads(str(arr["_meta"]))
        assert meta["frame_range"] == [5, 15]


def test_invalid_inputs() -> None:
    from src.apps.services.marker_export import export_markers  # type: ignore

    model = _make_model(10)
    with pytest.raises(ValueError):
        export_markers(model, [], "all", None, "csv", "/tmp/x.csv")
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "all", None, "txt", "/tmp/x.txt")
    with pytest.raises(ValueError):
        export_markers(model, ["NopeMarker"], "all", None, "csv", "/tmp/x.csv")
    with pytest.raises(ValueError):
        export_markers(model, ["WaistLeft"], "all", (0, 999), "csv", "/tmp/x.csv")
