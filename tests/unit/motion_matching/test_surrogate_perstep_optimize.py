"""Tests for surrogate perstep optimization logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.shared.python.motion_matching.surrogate.perstep.optimize import (
    _desired_club_targets,
    _desired_quaternions,
    _interpolate_reference,
    _read_state_reference,
    build_argument_parser,
    find_quaternion_columns,
    quaternion_orientation_term,
    quaternion_orientation_term_numpy,
    resolve_cost_config,
    total_work_numpy,
    total_work_regularizer,
)


@pytest.mark.unit
def test_resolve_cost_config():
    """Test cost config resolution."""
    c = resolve_cost_config("position")
    assert c.mode == "position"
    assert c.regularizer_kind == "effort_smoothness"

    c = resolve_cost_config("full")
    assert c.mode == "full"
    assert c.regularizer_kind == "total_work"

    c = resolve_cost_config("position_orientation", regularizer_kind="total_work")
    assert c.mode == "position_orientation"
    assert c.regularizer_kind == "total_work"

    with pytest.raises(ValueError):
        resolve_cost_config("invalid_mode")

    with pytest.raises(ValueError):
        resolve_cost_config("position", regularizer_kind="invalid")


@pytest.mark.unit
def test_find_quaternion_columns():
    """Test finding quaternion columns in dataframe."""
    df1 = pd.DataFrame(
        {"club_quat_w": [], "club_quat_x": [], "club_quat_y": [], "club_quat_z": []}
    )
    assert find_quaternion_columns(df1) == (
        "club_quat_w",
        "club_quat_x",
        "club_quat_y",
        "club_quat_z",
    )

    df2 = pd.DataFrame(
        {"clubface_qw": [], "clubface_qx": [], "clubface_qy": [], "clubface_qz": []}
    )
    assert find_quaternion_columns(df2) == (
        "clubface_qw",
        "clubface_qx",
        "clubface_qy",
        "clubface_qz",
    )

    df_missing = pd.DataFrame({"club_quat_w": [], "club_quat_x": []})
    assert find_quaternion_columns(df_missing) is None


@pytest.mark.unit
def test_quaternion_orientation_term():
    """Test orientation term."""
    q_sim = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    q_meas = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    loss = quaternion_orientation_term(q_sim, q_meas)
    assert loss.item() < 1e-6  # q and -q should both have 0 loss

    # Perpendicular quaternions
    q_meas2 = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    loss2 = quaternion_orientation_term(q_sim, q_meas2)
    assert loss2.item() > 0.0

    with pytest.raises(ValueError):
        quaternion_orientation_term(torch.ones(2, 4), torch.ones(3, 4))

    with pytest.raises(ValueError):
        quaternion_orientation_term(torch.ones(2, 3), torch.ones(2, 3))


@pytest.mark.unit
def test_quaternion_orientation_term_numpy():
    """Test numpy orientation term."""
    q_sim = np.array([[1.0, 0.0, 0.0, 0.0]])
    q_meas = np.array([[-1.0, 0.0, 0.0, 0.0]])
    loss = quaternion_orientation_term_numpy(q_sim, q_meas)
    assert loss < 1e-6


@pytest.mark.unit
def test_total_work_regularizer():
    """Test total work regularizer."""
    tau = torch.ones((10, 3))
    omega = torch.ones((10, 3))
    time = torch.linspace(0.0, 1.0, 10)

    work = total_work_regularizer(tau, omega, time)
    assert work.item() > 0.0

    with pytest.raises(ValueError):
        total_work_regularizer(tau, torch.ones(10, 2), time)

    with pytest.raises(ValueError):
        total_work_regularizer(tau, omega, torch.linspace(0, 1, 11))


@pytest.mark.unit
def test_total_work_numpy():
    """Test total work numpy implementation."""
    tau = np.ones((10, 3))
    omega = np.ones((10, 3))
    time = np.linspace(0.0, 1.0, 10)

    work = total_work_numpy(tau, omega, time)
    assert work > 0.0


@pytest.mark.unit
def test_read_state_reference(tmp_path: Path):
    """Test reading state reference."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "ref.csv"
    df.to_csv(csv_path, index=False)

    res = _read_state_reference(csv_path, ["a"])
    assert list(res.columns) == ["a", "b"]

    with pytest.raises(ValueError):
        _read_state_reference(csv_path, ["c"])


@pytest.mark.unit
def test_interpolate_reference():
    """Test reference interpolation."""
    df1 = pd.DataFrame({"a": [1.0]})
    res1 = _interpolate_reference(df1, np.array([0.0, 0.5, 1.0]), ["a"])
    assert res1.shape == (3, 1)
    assert np.allclose(res1, 1.0)

    df2 = pd.DataFrame({"time": [0.0, 1.0], "a": [0.0, 2.0]})
    res2 = _interpolate_reference(df2, np.array([0.0, 0.5, 1.0]), ["a"])
    assert res2.shape == (3, 1)
    assert np.allclose(res2[:, 0], [0.0, 1.0, 2.0])


@pytest.mark.unit
def test_desired_club_targets():
    """Test desired club targets extraction."""
    df = pd.DataFrame(
        {"clubface_x": [1.0, 2.0], "ClubLogs_CHGlobalPosition_2": [3.0, 4.0]}
    )
    targets, indices = _desired_club_targets(
        df, ["ClubLogs_CHGlobalPosition_1", "ClubLogs_CHGlobalPosition_2"]
    )
    assert len(indices) == 2
    assert targets.shape == (2, 2)
    assert np.allclose(targets[:, 0], [3.0, 4.0])

    with pytest.raises(ValueError):
        _desired_club_targets(
            pd.DataFrame({"unknown": [1]}), ["ClubLogs_CHGlobalPosition_1"]
        )


@pytest.mark.unit
def test_desired_quaternions():
    """Test desired quaternions extraction."""
    df = pd.DataFrame(
        {
            "club_quat_w": [1.0],
            "club_quat_x": [0.0],
            "club_quat_y": [0.0],
            "club_quat_z": [0.0],
        }
    )
    res = _desired_quaternions(df)
    assert res is not None
    assert res.shape == (1, 4)

    assert _desired_quaternions(pd.DataFrame({"a": [1]})) is None


@pytest.mark.unit
def test_argparse():
    """Test argument parsing."""
    parser = build_argument_parser()
    args = parser.parse_args(
        ["--desired-club-csv", "d.csv", "--reference-body-csv", "r.csv"]
    )
    assert str(args.desired_club_csv) == "d.csv"
    assert str(args.reference_body_csv) == "r.csv"
    assert args.cost_mode == "position"
