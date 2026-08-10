"""Tests for the DeepLabCut adapter.

Golden fixtures are built in-test. The CSV fixture is written by pandas
``to_csv`` (3 header rows). The HDF5 fixtures replicate — via ``h5py`` —
the exact on-disk layouts pandas ``to_hdf`` produces for a DLC frame in
both the "fixed" and "table" formats (PyTables is not a dependency of
this repo, so ``to_hdf`` itself is unavailable in CI; the replicated
layouts were verified byte-structure-identical against genuine
``pandas.DataFrame.to_hdf`` output).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.shared.python.motion_pipeline.sources import detect_format, load_any
from src.shared.python.motion_pipeline.sources.csv_adapter import CSVAdapter
from src.shared.python.motion_pipeline.sources.deeplabcut_adapter import (
    DeepLabCutAdapter,
)

pytestmark = pytest.mark.unit

_SCORER = "DLC_resnet50_golfJan1shuffle1_50000"
_BODYPARTS = ["clubhead", "hosel", "ball"]
_COORDS = ["x", "y", "likelihood"]


def _golden_data(n_frames: int = 4) -> np.ndarray:
    """Deterministic (n_frames, n_bodyparts, 3) x/y/likelihood array."""
    data = np.empty((n_frames, len(_BODYPARTS), 3), dtype=float)
    for i in range(n_frames):
        for b in range(len(_BODYPARTS)):
            data[i, b] = [10.0 * b + i, 100.0 + 5.0 * b - i, 0.5 + 0.1 * b]
    return data


def _golden_df(data: np.ndarray) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [[_SCORER], _BODYPARTS, _COORDS], names=["scorer", "bodyparts", "coords"]
    )
    return pd.DataFrame(data.reshape(data.shape[0], -1), columns=columns)


def _write_csv(path: Path, data: np.ndarray) -> Path:
    _golden_df(data).to_csv(path)
    return path


def _write_h5_fixed(path: Path, data: np.ndarray) -> Path:
    """Replicate the pandas 'fixed' to_hdf layout for a DLC frame."""
    n_frames = data.shape[0]
    flat = data.reshape(n_frames, -1)
    n_cols = flat.shape[1]
    label0 = np.zeros(n_cols, dtype=np.int8)
    label1 = np.repeat(np.arange(len(_BODYPARTS), dtype=np.int8), len(_COORDS))
    label2 = np.tile(np.arange(len(_COORDS), dtype=np.int8), len(_BODYPARTS))
    with h5py.File(path, "w") as f:
        g = f.create_group("df_with_missing")
        g.attrs["pandas_type"] = np.bytes_(b"frame")
        g.attrs["pandas_version"] = np.bytes_(b"0.15.2")
        g.attrs["axis0_variety"] = np.bytes_(b"multi")
        g.attrs["axis1_variety"] = np.bytes_(b"regular")
        g.attrs["axis0_nlevels"] = 3
        g.attrs["block0_items_variety"] = np.bytes_(b"multi")
        g.attrs["block0_items_nlevels"] = 3
        g.attrs["encoding"] = np.bytes_(b"UTF-8")
        g.attrs["errors"] = np.bytes_(b"strict")
        g.attrs["nblocks"] = 1
        g.attrs["ndim"] = 2
        for key in ("axis0", "block0_items"):
            g.create_dataset(f"{key}_level0", data=np.array([_SCORER], dtype="S"))
            g.create_dataset(f"{key}_level1", data=np.array(_BODYPARTS, dtype="S"))
            g.create_dataset(f"{key}_level2", data=np.array(_COORDS, dtype="S"))
            g.create_dataset(f"{key}_label0", data=label0)
            g.create_dataset(f"{key}_label1", data=label1)
            g.create_dataset(f"{key}_label2", data=label2)
        axis1 = g.create_dataset("axis1", data=np.arange(n_frames, dtype=np.int64))
        axis1.attrs["kind"] = np.bytes_(b"integer")
        g.create_dataset("block0_values", data=flat)
    return path


def _write_h5_table(path: Path, data: np.ndarray) -> Path:
    """Replicate the pandas 'table' (format="table") to_hdf layout."""
    n_frames = data.shape[0]
    flat = data.reshape(n_frames, -1)
    columns = [(_SCORER, bp, c) for bp in _BODYPARTS for c in _COORDS]
    dtype = np.dtype([("index", "<i8"), ("values_block_0", "<f8", (len(columns),))])
    rows = np.empty(n_frames, dtype=dtype)
    rows["index"] = np.arange(n_frames)
    rows["values_block_0"] = flat
    with h5py.File(path, "w") as f:
        g = f.create_group("df_with_missing")
        g.attrs["pandas_type"] = np.bytes_(b"frame_table")
        g.attrs["pandas_version"] = np.bytes_(b"0.15.2")
        g.attrs["table_type"] = np.bytes_(b"appendable_frame")
        g.attrs["non_index_axes"] = np.bytes_(pickle.dumps([(1, columns)], protocol=0))
        g.create_dataset("table", data=rows)
    return path


# ----------------------------------------------------------------------
# Routing / detection


def test_detect_format_routes_h5_fixed(tmp_path: Path) -> None:
    p = _write_h5_fixed(tmp_path / "videoDLC.h5", _golden_data())
    assert DeepLabCutAdapter.supports(p) is True
    assert detect_format(p) is DeepLabCutAdapter


def test_detect_format_routes_h5_table(tmp_path: Path) -> None:
    p = _write_h5_table(tmp_path / "videoDLC.h5", _golden_data())
    assert DeepLabCutAdapter.supports(p) is True
    assert detect_format(p) is DeepLabCutAdapter


def test_detect_format_routes_dlc_csv(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "videoDLC.csv", _golden_data())
    assert DeepLabCutAdapter.supports(p) is True
    assert detect_format(p) is DeepLabCutAdapter


def test_generic_csv_adapter_still_claims_plain_csv(tmp_path: Path) -> None:
    """Regression: the DLC adapter must not steal plain trajectory CSVs."""
    p = tmp_path / "plain.csv"
    p.write_text(
        "frame,timestamp,x_hip,y_hip,z_hip\n0,0.000,0.0,1.0,0.5\n1,0.033,0.1,1.0,0.5\n"
    )
    assert DeepLabCutAdapter.supports(p) is False
    assert CSVAdapter.supports(p) is True
    assert detect_format(p) is CSVAdapter


def test_generic_csv_adapter_does_not_claim_dlc_csv(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "videoDLC.csv", _golden_data())
    assert CSVAdapter.supports(p) is False


# ----------------------------------------------------------------------
# Loading


@pytest.mark.parametrize("writer", [_write_h5_fixed, _write_h5_table, _write_csv])
def test_load_any_preserves_custom_keypoints(tmp_path: Path, writer) -> None:
    data = _golden_data()
    p = writer(tmp_path / f"videoDLC{'.csv' if writer is _write_csv else '.h5'}", data)
    seq = load_any(p)  # runs load_checked postconditions
    assert seq.num_frames == 4
    for frame in seq.frames:
        assert frame.schema_name == "custom"
        assert [kp.name for kp in frame.keypoints] == _BODYPARTS
    # likelihood -> confidence, coordinates intact
    kp = seq.frames[1].keypoints[2]  # frame 1, "ball"
    assert kp.x == pytest.approx(data[1, 2, 0])
    assert kp.y == pytest.approx(data[1, 2, 1])
    assert kp.confidence == pytest.approx(data[1, 2, 2])
    assert kp.z is None  # DLC is 2D


def test_synthesized_timestamps_at_default_fps(tmp_path: Path) -> None:
    p = _write_h5_fixed(tmp_path / "videoDLC.h5", _golden_data())
    seq = DeepLabCutAdapter().load_checked(p)
    timestamps = [f.timestamp for f in seq.frames]
    assert timestamps == pytest.approx([i / 30.0 for i in range(4)])
    assert all(b > a for a, b in zip(timestamps, timestamps[1:], strict=False))


def test_synthesized_timestamps_at_configured_fps(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "videoDLC.csv", _golden_data())
    seq = DeepLabCutAdapter(fps=120.0).load_checked(p)
    timestamps = [f.timestamp for f in seq.frames]
    assert timestamps == pytest.approx([i / 120.0 for i in range(4)])


def test_load_checked_postconditions_pass(tmp_path: Path) -> None:
    p = _write_h5_table(tmp_path / "videoDLC.h5", _golden_data())
    seq = DeepLabCutAdapter().load_checked(p)  # would raise AdapterContractError
    assert seq.num_frames == 4
    assert seq.metadata["scorer"] == _SCORER
    assert seq.metadata["bodyparts"] == _BODYPARTS


def test_metadata(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "videoDLC.csv", _golden_data())
    md = DeepLabCutAdapter(fps=240.0).metadata(p)
    assert md.format_name == "deeplabcut"
    assert md.fps == 240.0
    assert md.frame_count == 4
    assert md.unit_system == "pixels"
    assert md.keypoint_schema == "custom"


def test_likelihood_clamped_to_unit_interval(tmp_path: Path) -> None:
    data = _golden_data()
    data[0, 0, 2] = 1.5  # out-of-range likelihood
    p = _write_csv(tmp_path / "videoDLC.csv", data)
    seq = DeepLabCutAdapter().load_checked(p)
    assert seq.frames[0].keypoints[0].confidence == 1.0


def test_nan_keypoints_dropped_not_fatal(tmp_path: Path) -> None:
    data = _golden_data()
    data[2, 1, 0] = np.nan  # hosel x missing in frame 2
    p = _write_csv(tmp_path / "videoDLC.csv", data)
    seq = DeepLabCutAdapter().load_checked(p)
    assert seq.num_frames == 4
    assert [kp.name for kp in seq.frames[2].keypoints] == ["clubhead", "ball"]


def test_all_nan_file_raises_clear_error(tmp_path: Path) -> None:
    data = np.full((3, len(_BODYPARTS), 3), np.nan)
    p = _write_csv(tmp_path / "videoDLC.csv", data)
    with pytest.raises(ValueError, match="no usable frames"):
        DeepLabCutAdapter().load(p)


# ----------------------------------------------------------------------
# Malformed inputs -> descriptive errors, not pandas/h5py stack traces


def test_garbage_h5_not_supported_and_load_raises(tmp_path: Path) -> None:
    p = tmp_path / "garbage.h5"
    p.write_text("this is not an HDF5 file")
    assert DeepLabCutAdapter.supports(p) is False
    with pytest.raises(ValueError, match="Malformed DeepLabCut HDF5"):
        DeepLabCutAdapter().load(p)


def test_h5_without_dlc_frame_raises(tmp_path: Path) -> None:
    p = tmp_path / "other.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("random", data=np.arange(4))
    assert DeepLabCutAdapter.supports(p) is False
    with pytest.raises(ValueError, match="no DeepLabCut-style"):
        DeepLabCutAdapter().load(p)


def test_csv_missing_likelihood_column_raises(tmp_path: Path) -> None:
    columns = pd.MultiIndex.from_product([[_SCORER], _BODYPARTS, ["x", "y"]])
    df = pd.DataFrame(np.ones((2, 6)), columns=columns)
    # Force the DLC header names so the sniffer claims the file
    df.columns.names = ["scorer", "bodyparts", "coords"]
    p = tmp_path / "videoDLC.csv"
    df.to_csv(p)
    assert DeepLabCutAdapter.supports(p) is True
    with pytest.raises(ValueError, match="missing the 'likelihood' column"):
        DeepLabCutAdapter().load(p)


def test_multi_animal_csv_rejected_descriptively(tmp_path: Path) -> None:
    p = tmp_path / "multiDLC.csv"
    p.write_text(
        "scorer,s,s,s\n"
        "individuals,a1,a1,a1\n"
        "bodyparts,clubhead,clubhead,clubhead\n"
        "coords,x,y,likelihood\n"
        "0,1.0,2.0,0.9\n"
    )
    with pytest.raises(ValueError, match="multi-animal"):
        DeepLabCutAdapter().load(p)


def test_non_numeric_csv_cells_raise(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "videoDLC.csv", _golden_data())
    text = p.read_text().replace("100.0", "oops", 1)
    p.write_text(text)
    with pytest.raises(ValueError, match="non-numeric"):
        DeepLabCutAdapter().load(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DeepLabCutAdapter().load(tmp_path / "nope.h5")


def test_invalid_fps_rejected() -> None:
    with pytest.raises(ValueError, match="fps"):
        DeepLabCutAdapter(fps=0.0)
    with pytest.raises(ValueError, match="fps"):
        DeepLabCutAdapter(fps=float("nan"))
