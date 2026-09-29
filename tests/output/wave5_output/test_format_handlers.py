"""Tests for output format handlers (CSV/JSON/HDF5/Parquet)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shared.python.data_io._format_handlers import (
    OutputFormat,
    _make_json_serializer,
    dispatch_load,
    dispatch_save,
    save_csv,
    save_hdf5,
    save_json,
    save_parquet,
)
from src.shared.python.data_io.provenance import ProvenanceInfo


@pytest.fixture
def provenance() -> ProvenanceInfo:
    return ProvenanceInfo.capture()


def test_output_format_enum_values():
    assert OutputFormat.CSV.value == "csv"
    assert OutputFormat.JSON.value == "json"
    assert OutputFormat.HDF5.value == "hdf5"
    assert OutputFormat.PARQUET.value == "parquet"
    assert OutputFormat.PICKLE.value == "pickle"


def test_json_serializer_ndarray():
    s = _make_json_serializer()
    assert s(np.array([1, 2, 3])) == [1, 2, 3]


def test_json_serializer_numpy_scalars():
    s = _make_json_serializer()
    assert s(np.int32(5)) == 5.0
    assert s(np.float64(2.5)) == 2.5


def test_json_serializer_datetime():
    s = _make_json_serializer()
    result = s(datetime(2024, 1, 1, 12, 0, 0))
    assert "2024" in result


def test_json_serializer_unsupported_type_raises():
    s = _make_json_serializer()
    with pytest.raises(TypeError, match="not JSON serializable"):
        s(object())


def test_save_csv_dataframe(tmp_path: Path, provenance: ProvenanceInfo):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    p = tmp_path / "out.csv"
    save_csv(df, p, provenance)
    assert p.exists()
    loaded = pd.read_csv(p, comment="#")
    pd.testing.assert_frame_equal(df, loaded)


def test_save_csv_dict(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.csv"
    save_csv({"x": [1, 2], "y": [3, 4]}, p, provenance)
    assert p.exists()
    loaded = pd.read_csv(p, comment="#")
    assert list(loaded.columns) == ["x", "y"]


def test_save_csv_contains_provenance_header(
    tmp_path: Path, provenance: ProvenanceInfo
):
    df = pd.DataFrame({"a": [1]})
    p = tmp_path / "out.csv"
    save_csv(df, p, provenance)
    text = p.read_text()
    # Provenance header lines are comment-prefixed
    assert text.startswith("#") or "#" in text.splitlines()[0]


def test_save_json_dict(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.json"
    save_json({"x": 1, "y": 2}, p, provenance, metadata={"note": "hi"}, engine="mujoco")
    payload = json.loads(p.read_text())
    assert payload["engine"] == "mujoco"
    assert payload["metadata"] == {"note": "hi"}
    assert payload["results"] == {"x": 1, "y": 2}
    assert "provenance" in payload


def test_save_json_no_metadata(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.json"
    save_json([1, 2, 3], p, provenance, metadata=None, engine="drake")
    payload = json.loads(p.read_text())
    assert payload["metadata"] == {}
    assert payload["engine"] == "drake"


def test_save_json_handles_numpy(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.json"
    save_json({"arr": np.array([1, 2, 3])}, p, provenance, None, "mujoco")
    payload = json.loads(p.read_text())
    assert payload["results"]["arr"] == [1, 2, 3]


def test_save_parquet_dataframe(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    p = tmp_path / "out.parquet"
    save_parquet(df, p)
    loaded = pd.read_parquet(p)
    pd.testing.assert_frame_equal(df, loaded)


def test_save_parquet_from_dict(tmp_path: Path):
    p = tmp_path / "out.parquet"
    save_parquet({"a": [1, 2], "b": [3, 4]}, p)
    loaded = pd.read_parquet(p)
    assert set(loaded.columns) == {"a", "b"}


def test_save_hdf5_dataframe(tmp_path: Path):
    pytest.importorskip("tables")
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    p = tmp_path / "out.h5"
    save_hdf5(df, p)
    loaded = pd.read_hdf(p, key="data")
    pd.testing.assert_frame_equal(df, loaded)


def test_save_hdf5_dict(tmp_path: Path):
    pytest.importorskip("tables")
    p = tmp_path / "out.h5"
    save_hdf5({"a": [1, 2], "b": [3, 4]}, p)
    loaded = pd.read_hdf(p, key="data")
    assert isinstance(loaded, pd.DataFrame)


def test_dispatch_save_csv(tmp_path: Path, provenance: ProvenanceInfo):
    df = pd.DataFrame({"a": [1, 2]})
    p = tmp_path / "out.csv"
    dispatch_save(df, p, OutputFormat.CSV, provenance, None, "mujoco")
    assert p.exists()


def test_dispatch_save_json(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.json"
    dispatch_save({"k": "v"}, p, OutputFormat.JSON, provenance, {"m": 1}, "drake")
    assert p.exists()


def test_dispatch_save_parquet(tmp_path: Path, provenance: ProvenanceInfo):
    df = pd.DataFrame({"a": [1]})
    p = tmp_path / "out.parquet"
    dispatch_save(df, p, OutputFormat.PARQUET, provenance, None, "mujoco")
    assert p.exists()


def test_dispatch_save_pickle_rejected(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.pkl"
    with pytest.raises(ValueError, match="Pickle format is disabled"):
        dispatch_save({"x": 1}, p, OutputFormat.PICKLE, provenance, None, "mujoco")


def test_dispatch_load_csv(tmp_path: Path, provenance: ProvenanceInfo):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "out.csv"
    save_csv(df, p, provenance)
    loaded = dispatch_load(p, OutputFormat.CSV)
    pd.testing.assert_frame_equal(df, loaded)


def test_dispatch_load_json(tmp_path: Path, provenance: ProvenanceInfo):
    p = tmp_path / "out.json"
    save_json({"x": 1}, p, provenance, None, "mujoco")
    loaded = dispatch_load(p, OutputFormat.JSON)
    assert loaded == {"x": 1}


def test_dispatch_load_json_raw_when_no_results_key(tmp_path: Path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps([1, 2, 3]))
    loaded = dispatch_load(p, OutputFormat.JSON)
    assert loaded == [1, 2, 3]


def test_dispatch_load_parquet(tmp_path: Path):
    df = pd.DataFrame({"a": [1]})
    p = tmp_path / "out.parquet"
    save_parquet(df, p)
    loaded = dispatch_load(p, OutputFormat.PARQUET)
    pd.testing.assert_frame_equal(df, loaded)


def test_dispatch_load_pickle_rejected(tmp_path: Path):
    p = tmp_path / "x.pkl"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="Pickle format is disabled"):
        dispatch_load(p, OutputFormat.PICKLE)


def test_dispatch_load_hdf5(tmp_path: Path):
    pytest.importorskip("tables")
    df = pd.DataFrame({"a": [1, 2]})
    p = tmp_path / "out.h5"
    save_hdf5(df, p)
    loaded = dispatch_load(p, OutputFormat.HDF5)
    pd.testing.assert_frame_equal(df, loaded)
