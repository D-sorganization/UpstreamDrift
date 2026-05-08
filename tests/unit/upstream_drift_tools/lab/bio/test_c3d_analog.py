"""Unit tests for ``_c3d_analog`` analog/force-plate processing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from src.shared.python.upstream_drift_tools.lab.bio._c3d_analog import (
    build_analog_dataframe,
    build_force_plate_dataframe,
    build_plate_dataframe,
    detect_force_plate_channels,
    force_plate_columns,
)
from src.shared.python.upstream_drift_tools.lab.bio._c3d_models import C3DMetadata

ANALOG_LOGGER = "src.shared.python.upstream_drift_tools.lab.bio._c3d_analog"


def _meta(
    analog_labels: list[str],
    analog_rate: float | None = 1000.0,
) -> C3DMetadata:
    return C3DMetadata(
        marker_labels=["M1"],
        frame_count=1,
        frame_rate=100.0,
        units="m",
        analog_labels=analog_labels,
        analog_units=[""] * len(analog_labels),
        analog_rate=analog_rate,
        events=[],
    )


# ----- build_analog_dataframe -----------------------------------------------


def test_build_analog_dataframe_empty() -> None:
    c3d_data = {"data": {"analogs": np.zeros((1, 0, 5))}}
    md = _meta([])
    df = build_analog_dataframe(c3d_data, md, include_time=True)
    assert df.empty
    assert list(df.columns) == ["sample", "time", "channel", "value"]


def test_build_analog_dataframe_no_time() -> None:
    c3d_data = {"data": {"analogs": np.zeros((1, 0, 5))}}
    md = _meta([])
    df = build_analog_dataframe(c3d_data, md, include_time=False)
    assert list(df.columns) == ["sample", "channel", "value"]


def test_build_analog_dataframe_synth_labels_when_unlabeled() -> None:
    # 2 channels, 3 frames, 1 subframe each
    arr = np.arange(6, dtype=float).reshape(1, 2, 3)
    c3d_data = {"data": {"analogs": arr}}
    md = _meta([], analog_rate=1000.0)
    df = build_analog_dataframe(c3d_data, md, include_time=True)
    assert set(df["channel"].unique()) == {"Analog_1", "Analog_2"}
    assert "time" in df.columns
    assert df.shape[0] == 6


def test_build_analog_dataframe_no_rate_omits_time() -> None:
    arr = np.arange(2, dtype=float).reshape(1, 1, 2)
    c3d_data = {"data": {"analogs": arr}}
    md = _meta(["A1"], analog_rate=None)
    df = build_analog_dataframe(c3d_data, md, include_time=True)
    assert "time" not in df.columns


# ----- detect_force_plate_channels ------------------------------------------


def test_detect_standard_channels() -> None:
    labels = ["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"]
    plates = detect_force_plate_channels(labels)
    assert set(plates.keys()) == {1}
    assert plates[1] == {
        "fx": "Fx1",
        "fy": "Fy1",
        "fz": "Fz1",
        "mx": "Mx1",
        "my": "My1",
        "mz": "Mz1",
    }


def test_detect_force_dot_prefix() -> None:
    labels = [
        "Force.Fx1",
        "Force.Fy1",
        "Force.Fz1",
        "Force.Mx1",
        "Force.My1",
        "Force.Mz1",
    ]
    plates = detect_force_plate_channels(labels)
    assert plates[1]["fx"] == "Force.Fx1"


def test_detect_vicon_style() -> None:
    labels = ["FP2_Fx", "FP2_Fy", "FP2_Fz", "FP2_Mx", "FP2_My", "FP2_Mz"]
    plates = detect_force_plate_channels(labels)
    assert 2 in plates
    assert plates[2]["fz"] == "FP2_Fz"


def test_detect_two_plates() -> None:
    labels = [
        "Fx1",
        "Fy1",
        "Fz1",
        "Mx1",
        "My1",
        "Mz1",
        "Fx2",
        "Fy2",
        "Fz2",
        "Mx2",
        "My2",
        "Mz2",
    ]
    plates = detect_force_plate_channels(labels)
    assert set(plates.keys()) == {1, 2}


def test_detect_ignores_unrelated_channels() -> None:
    labels = ["EMG1", "BodyMass", "PowerSupply"]
    assert detect_force_plate_channels(labels) == {}


# ----- force_plate_columns --------------------------------------------------


def test_force_plate_columns_all_options() -> None:
    cols = force_plate_columns(include_time=True, compute_cop=True)
    assert cols == [
        "sample",
        "time",
        "plate",
        "fx",
        "fy",
        "fz",
        "mx",
        "my",
        "mz",
        "cop_x",
        "cop_y",
        "cop_z",
    ]


def test_force_plate_columns_minimal() -> None:
    assert force_plate_columns(False, False) == [
        "sample",
        "plate",
        "fx",
        "fy",
        "fz",
        "mx",
        "my",
        "mz",
    ]


# ----- build_plate_dataframe ------------------------------------------------


def _wide(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": np.arange(n),
            "Fx1": np.full(n, 1.0),
            "Fy1": np.full(n, 2.0),
            "Fz1": np.array([0.0, 0.0, 100.0, 100.0])[:n],
            "Mx1": np.full(n, 5.0),
            "My1": np.full(n, -10.0),
            "Mz1": np.full(n, 0.5),
        }
    )


def test_build_plate_dataframe_missing_channel_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wide = _wide()
    channels = {"fx": "Fx1", "fy": "Fy1", "fz": "Fz1", "mx": "Mx1", "my": "My1"}
    with caplog.at_level(logging.WARNING, logger=ANALOG_LOGGER):
        out = build_plate_dataframe(
            1,
            channels,
            {"fx", "fy", "fz", "mx", "my", "mz"},
            wide,
            compute_cop=True,
            ground_height=0.0,
        )
    assert out is None
    assert any("missing channels" in r.getMessage() for r in caplog.records)


def test_build_plate_dataframe_cop() -> None:
    wide = _wide()
    channels = {
        "fx": "Fx1",
        "fy": "Fy1",
        "fz": "Fz1",
        "mx": "Mx1",
        "my": "My1",
        "mz": "Mz1",
    }
    out = build_plate_dataframe(
        1,
        channels,
        {"fx", "fy", "fz", "mx", "my", "mz"},
        wide,
        compute_cop=True,
        ground_height=0.05,
    )
    assert out is not None
    # First two rows: |fz|=0 < threshold -> NaN; last two rows: 100 N -> valid
    assert np.isnan(out["cop_x"].iloc[0])
    assert np.isnan(out["cop_z"].iloc[0])
    assert out["cop_x"].iloc[2] == pytest.approx(10.0 / 100.0)
    assert out["cop_y"].iloc[2] == pytest.approx(5.0 / 100.0)
    assert out["cop_z"].iloc[2] == pytest.approx(0.05)


def test_build_plate_dataframe_no_cop() -> None:
    wide = _wide()
    channels = {
        "fx": "Fx1",
        "fy": "Fy1",
        "fz": "Fz1",
        "mx": "Mx1",
        "my": "My1",
        "mz": "Mz1",
    }
    out = build_plate_dataframe(
        1,
        channels,
        {"fx", "fy", "fz", "mx", "my", "mz"},
        wide,
        compute_cop=False,
        ground_height=0.0,
    )
    assert out is not None
    assert "cop_x" not in out.columns


# ----- build_force_plate_dataframe ------------------------------------------


def _analog_long_for_plate(samples: int = 4) -> pd.DataFrame:
    """Build a long analog DataFrame for one plate's worth of channels."""
    rows = []
    fz_values = np.array([0.0, 0.0, 100.0, 100.0])[:samples]
    channel_values = {
        "Fx1": np.full(samples, 1.0),
        "Fy1": np.full(samples, 2.0),
        "Fz1": fz_values,
        "Mx1": np.full(samples, 5.0),
        "My1": np.full(samples, -10.0),
        "Mz1": np.full(samples, 0.5),
    }
    for s in range(samples):
        for ch, vals in channel_values.items():
            rows.append({"sample": s, "channel": ch, "value": vals[s]})
    return pd.DataFrame(rows)


def test_build_force_plate_no_plates(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=ANALOG_LOGGER):
        df = build_force_plate_dataframe(
            {},
            pd.DataFrame({"sample": [], "channel": [], "value": []}),
            1000.0,
            "x.c3d",
            None,
            True,
            True,
            0.0,
        )
    assert df.empty
    assert any("No force plate channels" in r.getMessage() for r in caplog.records)


def test_build_force_plate_unknown_plate_number() -> None:
    plates = {
        1: {
            "fx": "Fx1",
            "fy": "Fy1",
            "fz": "Fz1",
            "mx": "Mx1",
            "my": "My1",
            "mz": "Mz1",
        }
    }
    long_df = _analog_long_for_plate()
    with pytest.raises(ValueError, match="Force plate 5 not found"):
        build_force_plate_dataframe(
            plates, long_df, 1000.0, "x.c3d", 5, True, True, 0.0
        )


def test_build_force_plate_full_pipeline_with_time_and_cop() -> None:
    plates = {
        1: {
            "fx": "Fx1",
            "fy": "Fy1",
            "fz": "Fz1",
            "mx": "Mx1",
            "my": "My1",
            "mz": "Mz1",
        }
    }
    long_df = _analog_long_for_plate()
    df = build_force_plate_dataframe(
        plates, long_df, 1000.0, "x.c3d", None, True, True, 0.05
    )
    assert "time" in df.columns
    assert "cop_x" in df.columns
    assert df["plate"].unique().tolist() == [1]
    assert df.shape[0] == 4


def test_build_force_plate_explicit_plate_filter() -> None:
    plates = {
        1: {
            "fx": "Fx1",
            "fy": "Fy1",
            "fz": "Fz1",
            "mx": "Mx1",
            "my": "My1",
            "mz": "Mz1",
        },
        2: {
            "fx": "Fx1",
            "fy": "Fy1",
            "fz": "Fz1",
            "mx": "Mx1",
            "my": "My1",
            "mz": "Mz1",
        },  # share channels for test
    }
    long_df = _analog_long_for_plate()
    df = build_force_plate_dataframe(
        plates, long_df, 1000.0, "x.c3d", 2, False, False, 0.0
    )
    assert set(df["plate"].unique()) == {2}
    assert "time" not in df.columns
    assert "cop_x" not in df.columns


def test_build_force_plate_all_skipped_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # plate channels missing required keys -> all skipped
    plates = {1: {"fx": "Fx1"}}  # incomplete
    long_df = _analog_long_for_plate()
    with caplog.at_level(logging.WARNING, logger=ANALOG_LOGGER):
        df = build_force_plate_dataframe(
            plates, long_df, 1000.0, "x.c3d", None, True, True, 0.0
        )
    assert df.empty
    assert list(df.columns) == force_plate_columns(True, True)


def test_build_force_plate_no_rate_omits_time() -> None:
    plates = {
        1: {
            "fx": "Fx1",
            "fy": "Fy1",
            "fz": "Fz1",
            "mx": "Mx1",
            "my": "My1",
            "mz": "Mz1",
        }
    }
    long_df = _analog_long_for_plate()
    df = build_force_plate_dataframe(
        plates, long_df, None, "x.c3d", None, True, False, 0.0
    )
    assert "time" not in df.columns
