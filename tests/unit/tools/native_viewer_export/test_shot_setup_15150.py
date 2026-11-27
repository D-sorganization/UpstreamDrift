"""Opt-in shot setup of the native export: club roll and ball position (#15150)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tools.native_viewer_export import cli, shot_setup
from src.tools.native_viewer_export.ball import SOURCE_ADDRESS, attach_ball
from src.tools.native_viewer_export.core import ExportSettings
from tests.unit.tools.native_viewer_export.test_export_ball import _fixture_swing

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

BUNDLE_ARGS = ["--bundle", "b.npz", "--out", "out", "--swing", "driver"]


def _roll_of(swing: Any) -> float:
    from src.shared.python.model_appearance.club_assembly import assembly_from_spec

    return float(assembly_from_spec(json.loads(swing.bundle.spec_bytes)).face_roll_deg)


def _faces(swing: Any) -> Any:
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    model = ClubPoseModel(swing.bundle.spec_bytes)
    coords = dict(
        zip(swing.bundle.coordinate_order, map(float, swing.q[3]), strict=True)
    )
    return model, model.face_world(coords), coords


def _address_ball(swing: Any, **changes: Any) -> Any:
    settings = dataclasses.replace(
        ExportSettings(impact_time_s=swing.time_s(swing.n_states - 1)), **changes
    )
    return attach_ball(swing, settings)


def test_defaults_change_nothing() -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing("driver", returning=True)
    assert shot_setup.apply_club_roll(swing, 0.0) is swing
    plain = _address_ball(swing)
    zero = _address_ball(swing, club_roll_in_hands_deg=0.0, ball_position_shift_m=0.0)
    assert plain == zero
    assert plain[1] is not None and "ball_position_shift_m" not in plain[1]
    settings = ExportSettings()
    assert settings.club_roll_in_hands_deg == 0.0
    assert settings.ball_position_shift_m == 0.0


def test_roll_adds_to_the_face_roll_and_records_the_offset() -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing("driver", returning=True)
    rolled = shot_setup.apply_club_roll(swing, -4.0)
    club = json.loads(rolled.bundle.spec_bytes)["club"]
    assert club["roll_in_hands_deg"] == -4.0
    assert _roll_of(rolled) == pytest.approx(_roll_of(swing) - 4.0)
    assert club["face_roll_deg"] == pytest.approx(_roll_of(swing) - 4.0)
    assert json.loads(swing.bundle.spec_bytes) != json.loads(rolled.bundle.spec_bytes)
    assert rolled.bundle.coordinate_order == swing.bundle.coordinate_order
    assert np.array_equal(rolled.q, swing.q)  # no joint angle changes


@pytest.mark.parametrize("offset", [-4.0, 2.5])
def test_roll_turns_the_face_about_the_shaft(offset: float) -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing("driver", returning=True)
    model0, face0, coords = _faces(swing)
    rolled = shot_setup.apply_club_roll(swing, offset)
    model1, face1, _ = _faces(rolled)
    before = np.column_stack([face0.toe, face0.normal, face0.grip])
    after = np.column_stack([face1.toe, face1.normal, face1.grip])
    turn = after @ before.T  # rigid world rotation of the head
    angle = math.degrees(math.acos(np.clip((np.trace(turn) - 1.0) / 2.0, -1, 1)))
    assert angle == pytest.approx(abs(offset), abs=1e-6)
    axis = np.array(
        [turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]]
    )
    axis /= np.linalg.norm(axis)
    shaft = model0.body_world(coords)[:3, 1]
    assert abs(float(axis @ shaft)) == pytest.approx(1.0, abs=1e-6)


def test_ball_shift_moves_the_nominal_ball_along_the_target_line() -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing("driver", returning=True)
    from src.tools.native_viewer_export.ball import address_ball

    nominal = address_ball(swing, 0.02)
    shifted = shot_setup.shift_ball(swing, nominal, 0.03)
    move = np.asarray(shifted.position_m) - np.asarray(nominal.position_m)
    target = shot_setup.target_direction(swing)
    assert target[2] == 0.0 and np.linalg.norm(target) == pytest.approx(1.0)
    assert move == pytest.approx(0.03 * target, abs=1e-12)
    assert shifted.ground_height_m == nominal.ground_height_m
    assert shot_setup.shift_ball(swing, nominal, 0.0) is nominal


def test_attach_ball_records_the_shift_and_places_by_strike() -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing("driver", returning=True)
    placed, record = _address_ball(swing, ball_position_shift_m=0.02)
    assert placed.ball is not None and record is not None
    assert record["ball_position_shift_m"] == 0.02
    assert "first few centimetres" in record["position_note"]
    assert placed.ball.source in (SOURCE_ADDRESS, "strike_geometry")


def test_a_ball_already_on_the_swing_is_not_shifted() -> None:
    pytest.importorskip("mujoco")
    swing = _fixture_swing("driver", returning=True)
    placed, _ = _address_ball(swing)
    own = dataclasses.replace(swing, ball=placed.ball)
    kept, record = _address_ball(own, ball_position_shift_m=0.05)
    assert kept.ball == placed.ball
    assert record is not None and "ball_position_shift_m" not in record


@pytest.mark.parametrize("value", [21.0, -20.5, float("nan"), float("inf")])
def test_roll_out_of_range_or_not_finite_raises(value: float) -> None:
    with pytest.raises(ValueError, match="club_roll_in_hands_deg"):
        ExportSettings(club_roll_in_hands_deg=value)


@pytest.mark.parametrize("value", [0.151, -0.2, float("nan"), float("-inf")])
def test_shift_out_of_range_or_not_finite_raises(value: float) -> None:
    with pytest.raises(ValueError, match="ball_position_shift_m"):
        ExportSettings(ball_position_shift_m=value)


@pytest.mark.parametrize("value", ["3", None, True])
def test_non_numbers_raise_type_error(value: Any) -> None:
    with pytest.raises(TypeError):
        ExportSettings(club_roll_in_hands_deg=value)
    with pytest.raises(TypeError):
        ExportSettings(ball_position_shift_m=value)


def test_limits_come_from_the_settings_file() -> None:
    raw = json.loads(
        (Path(shot_setup.__file__).parent / "shot_setup_settings.json").read_text()
    )
    assert raw["max_abs_roll_deg"] == 20 and raw["max_abs_ball_shift_m"] == 0.15
    assert set(raw["sources"]) == set(raw) - {"sources"}


def test_cli_parses_both_flags() -> None:
    parsed = cli.build_parser().parse_args(BUNDLE_ARGS)
    settings = cli.build_settings(parsed)
    assert settings.club_roll_in_hands_deg == 0.0
    assert settings.ball_position_shift_m == 0.0
    flags = ["--club-roll-in-hands", "-4", "--ball-position-shift", "0.03"]
    parsed = cli.build_parser().parse_args([*BUNDLE_ARGS, *flags])
    settings = cli.build_settings(parsed)
    assert settings.club_roll_in_hands_deg == -4.0
    assert settings.ball_position_shift_m == 0.03
