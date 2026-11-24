"""Flex sweep and CLI output on a synthetic grip motion (#15684)."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

from src.shared.python.club_fitting.__main__ import build_parser, write_outputs  # noqa: E402
from src.shared.python.club_fitting.flex_sweep import run_flex_sweep  # noqa: E402
from src.shared.python.club_fitting.grip_driven_shaft import HeadGeometry  # noqa: E402
from src.shared.python.club_fitting.grip_motion import (  # noqa: E402
    ClubPoseTrack,
    load_settings,
)
from src.shared.python.motion_matching import club_models as cm  # noqa: E402

pytestmark = pytest.mark.unit

HEAD = HeadGeometry(np.array([0.0, 0.0, cm.HEAD_HEIGHT_M]), np.array([0.0, 0.0, 1.0]))
FAST = dataclasses.replace(
    load_settings(),
    lead_in_s=0.06,
    post_impact_s=0.005,
    integration_dt_s=2e-4,
    record_every=1,
)


def track() -> ClubPoseTrack:
    t = np.arange(-0.6, 0.05, 0.002)
    rot = Rotation.from_rotvec(np.outer(12.0 * t, [0.0, 0.0, 1.0]))
    rot = rot * Rotation.from_euler("z", -90.0, degrees=True)
    grip = np.stack([0 * t, 0 * t, 1.0 + 0 * t], axis=1)
    return ClubPoseTrack(t, grip, rot.as_matrix())


@pytest.fixture(scope="module")
def rows():
    stub = lambda run: {"ball_speed_mph": 100.0 if run is None else 99.0}  # noqa: E731
    return run_flex_sweep(
        track(), cm.DRIVER, HEAD, 0.0, ["Ladies", "X-Stiff"], FAST, shot=stub
    )


@pytest.mark.slow
def test_rows_follow_flex_order_and_deflection(rows) -> None:
    assert [r.flex for r in rows] == ["Ladies", "X-Stiff"]
    assert rows[0].peak_tip_deflection_mm > rows[1].peak_tip_deflection_mm > 0.0
    assert rows[0].flex_cpm < rows[1].flex_cpm
    assert rows[0].shot == {"ball_speed_mph": 99.0}
    for r in rows:
        assert r.peak_head_speed_mph >= r.impact_head_speed_mph * 0.99
        assert r.rigid_impact_head_speed_mph == pytest.approx(
            rows[0].rigid_impact_head_speed_mph
        )


@pytest.mark.slow
def test_outputs_round_trip(rows, tmp_path) -> None:
    json_path, csv_path = write_outputs(rows, tmp_path / "out")
    data = json.loads(json_path.read_text())["rows"]
    assert data[0]["flex"] == "Ladies" and data[0]["shot_ball_speed_mph"] == 99.0
    with csv_path.open() as handle:
        assert [r["flex"] for r in csv.DictReader(handle)] == ["Ladies", "X-Stiff"]


def test_flex_contracts() -> None:
    with pytest.raises(ValueError, match="subset"):
        run_flex_sweep(track(), cm.DRIVER, HEAD, 0.0, ["Soft-ish"], FAST)
    with pytest.raises(ValueError, match="subset"):
        run_flex_sweep(track(), cm.DRIVER, HEAD, 0.0, [], FAST)
    with pytest.raises(ValueError, match="no rows"):
        write_outputs([], Path("unused"))


def test_cli_parser_defaults() -> None:
    args = build_parser().parse_args(["--bundle", "b.npz", "--out", "o"])
    assert args.club == "driver" and args.flexes is None
