"""Tests for the MachineLearning <-> canonical ClubTarget adapter."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.loaders._machinelearning_compat import (
    CLUBFACE_POSITION,
    CLUBLOGS_POSITION,
    DEFAULT_SHAFT_LENGTH_M,
    to_canonical_target_from_clubface,
    to_canonical_target_from_clublogs,
    to_machinelearning_clubface,
    to_machinelearning_clublogs,
)

from ._fixtures import make_target, repo_root


def _silence_lossy() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module=".*ml.*")


def test_clublogs_round_trip() -> None:
    """``target -> clublogs DF -> target`` is bit-equal on numerical fields."""
    target = make_target(n=301)
    df = to_machinelearning_clublogs(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recovered = to_canonical_target_from_clublogs(df)

    assert isinstance(recovered, ClubTarget)
    assert recovered.time.shape == target.time.shape
    np.testing.assert_allclose(recovered.time, target.time, atol=1e-12)
    np.testing.assert_allclose(recovered.clubhead, target.clubhead, atol=1e-12)
    assert recovered.impact_idx == target.impact_idx


def test_clubface_round_trip_lossy() -> None:
    """``target -> clubface DF -> target`` preserves clubhead positions; butt /
    quat are reconstructed and not necessarily equal to the original."""
    target = make_target(n=301)
    df = to_machinelearning_clubface(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recovered = to_canonical_target_from_clubface(df)

    np.testing.assert_allclose(recovered.clubhead, target.clubhead, atol=1e-12)
    np.testing.assert_allclose(recovered.time, target.time, atol=1e-12)
    # Documented loss: butt and quat are reconstructions, not the originals.
    assert not np.allclose(recovered.butt, target.butt, atol=1e-3)
    # Quaternions remain unit-norm regardless.
    np.testing.assert_allclose(
        np.linalg.norm(recovered.club_quat, axis=1),
        np.ones(recovered.club_quat.shape[0]),
        atol=1e-9,
    )


def test_to_machinelearning_emits_correct_columns() -> None:
    target = make_target(n=51)
    cf = to_machinelearning_clubface(target)
    cl = to_machinelearning_clublogs(target)
    for c in CLUBFACE_POSITION:
        assert c in cf.columns
    for c in CLUBLOGS_POSITION:
        assert c in cl.columns
    assert "time" in cf.columns and "time" in cl.columns


def test_invalid_columns_raise_clear_error() -> None:
    bad = pd.DataFrame({"foo": [1.0, 2.0], "bar": [3.0, 4.0]})
    with (
        pytest.raises(ValueError, match="missing required columns"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        to_canonical_target_from_clubface(bad)
    with (
        pytest.raises(ValueError, match="missing required columns"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        to_canonical_target_from_clublogs(bad)


def test_lossy_warning_emitted() -> None:
    target = make_target(n=51)
    df = to_machinelearning_clublogs(target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        to_canonical_target_from_clublogs(df)
    assert any(
        issubclass(w.category, UserWarning) and "lossy" in str(w.message)
        for w in caught
    )


def test_too_short_dataframe_rejected() -> None:
    cols = {c: [1.0] for c in CLUBLOGS_POSITION}
    df = pd.DataFrame(cols)
    with pytest.raises(ValueError, match=">=2 rows"), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        to_canonical_target_from_clublogs(df)


def test_butt_uses_default_shaft_length() -> None:
    """When velocity is finite, butt = clubhead - shaft * v_hat."""
    target = make_target(n=101)
    df = to_machinelearning_clublogs(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recovered = to_canonical_target_from_clublogs(df)
    # Centre sample has well-defined velocity; check geometry there.
    i = 50
    delta = recovered.butt[i] - recovered.clubhead[i]
    assert abs(np.linalg.norm(delta) - DEFAULT_SHAFT_LENGTH_M) < 1e-9


@pytest.mark.integration
def test_to_canonical_from_clubface_against_wiffle_fixture() -> None:
    """Check that the prepare_club_target_trajectory output, fed through the
    adapter, agrees on positions with the canonical excel loader within 1e-9."""
    pytest.importorskip("openpyxl")
    excel_rel = (
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
        "golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"
    )
    xlsx = repo_root() / excel_rel
    if not xlsx.is_file():
        pytest.skip("Wiffle xlsx fixture not present")

    import importlib.util
    import sys

    prep_path = (
        repo_root()
        / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/"
        "prepare_club_target_trajectory.py"
    )
    if not prep_path.is_file():
        pytest.skip("prepare_club_target_trajectory.py not present")
    spec = importlib.util.spec_from_file_location("_prep_ctt", str(prep_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_prep_ctt"] = mod
    spec.loader.exec_module(mod)

    raw = pd.read_excel(xlsx, sheet_name="TW_ProV1", header=None)
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Sample #")]
    if len(header_row) != 1:
        pytest.skip("Wiffle workbook layout differs from expected")
    start = int(header_row[0]) + 1
    data = raw.iloc[start:, :26].copy()
    data.columns = [
        "sample",
        "time",
        "midhand_x",
        "midhand_y",
        "midhand_z",
        "midhand_xx",
        "midhand_xy",
        "midhand_xz",
        "midhand_yx",
        "midhand_yy",
        "midhand_yz",
        "midhand_zx",
        "midhand_zy",
        "midhand_zz",
        "clubface_x",
        "clubface_y",
        "clubface_z",
        "clubface_xx",
        "clubface_xy",
        "clubface_xz",
        "clubface_yx",
        "clubface_yy",
        "clubface_yz",
        "clubface_zx",
        "clubface_zy",
        "clubface_zz",
    ]
    numeric = (
        data.apply(pd.to_numeric, errors="coerce")
        .dropna(subset=["sample", "time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    time = numeric["time"].to_numpy(dtype=np.float64)
    pos_in = numeric[["clubface_x", "clubface_y", "clubface_z"]].to_numpy(
        dtype=np.float64
    )
    # Workbook is in inches with a non-zero origin; convert to metres and
    # centre so the canonical ClubTarget validators (|r| < 5 m) accept it.
    pos_m = pos_in * 0.0254
    pos = pos_m - pos_m.mean(axis=0)
    # Keep only the central swing window so |r| stays well below 5 m.
    inside = np.linalg.norm(pos, axis=1) < 3.5
    if inside.sum() < 50:
        pytest.skip("Wiffle workbook does not produce a usable centred window")
    pos = pos[inside]
    time = time[inside] - time[inside][0]
    vel = np.gradient(pos, time, axis=0, edge_order=1)
    acc = np.gradient(vel, time, axis=0, edge_order=1)
    df = pd.DataFrame(
        {
            "time": time,
            "clubface_x": pos[:, 0],
            "clubface_y": pos[:, 1],
            "clubface_z": pos[:, 2],
            "clubface_vx": vel[:, 0],
            "clubface_vy": vel[:, 1],
            "clubface_vz": vel[:, 2],
            "clubface_ax": acc[:, 0],
            "clubface_ay": acc[:, 1],
            "clubface_az": acc[:, 2],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        target = to_canonical_target_from_clubface(df)
    np.testing.assert_allclose(target.clubhead, pos, atol=1e-9)
