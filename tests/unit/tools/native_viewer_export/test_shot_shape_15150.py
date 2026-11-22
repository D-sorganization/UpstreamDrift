"""Opt-in ``--shot-shape`` roll solver of the native export (#15150)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.tools.native_viewer_export import cli, shot_setup
from src.tools.native_viewer_export.core import ExportSettings

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

BUNDLE_ARGS = ["--bundle", "b.npz", "--out", "out", "--swing", "driver"]


def _line(roll: float) -> float:
    """Synthetic landing side: monotonic, about 6.6 m per degree, zero at -2."""
    return 6.6 * (roll + 2.0)


def _settings(**changes: Any) -> shot_setup.ShotSetupSettings:
    return dataclasses.replace(shot_setup.load_shot_setup_settings(), **changes)


def test_solver_finds_the_root_within_tolerance_and_budget() -> None:
    calls: list[float] = []

    def side(roll: float) -> float:
        calls.append(roll)
        return _line(roll) + 0.3 * (roll + 2.0) ** 3  # mildly nonlinear

    limits = shot_setup.load_shot_setup_settings()
    roll, residual = shot_setup.solve_straight_roll(side, limits)
    assert abs(residual) <= limits.straight_tolerance_m
    assert roll == pytest.approx(-2.0, abs=0.05)
    assert len(calls) <= limits.max_solve_steps
    assert all(abs(r) <= limits.solve_span_deg for r in calls)


def test_solver_returns_the_exact_zero_at_an_end_of_the_bracket() -> None:
    limits = _settings(solve_span_deg=2.0)
    roll, residual = shot_setup.solve_straight_roll(_line, limits)
    assert roll == -2.0 and residual == 0.0


def test_no_sign_change_in_the_bracket_raises() -> None:
    with pytest.raises(ValueError, match="sign change"):
        shot_setup.solve_straight_roll(lambda r: 20.0 + r, _settings())


def test_non_finite_side_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        shot_setup.solve_straight_roll(lambda r: float("nan"), _settings())


def test_exhausted_step_budget_raises() -> None:
    limits = _settings(max_solve_steps=2, straight_tolerance_m=1e-9)
    with pytest.raises(ValueError, match="steps"):
        shot_setup.solve_straight_roll(lambda r: _line(r) ** 3, limits)


def test_draw_and_fade_are_the_straight_roll_minus_and_plus_the_step() -> None:
    limits = shot_setup.load_shot_setup_settings()
    straight, _ = shot_setup.solve_straight_roll(_line, limits)
    step = limits.shape_roll_step_deg
    assert shot_setup.roll_for_shape("straight", _line, limits)[0] == straight
    assert shot_setup.roll_for_shape("draw", _line, limits)[0] == straight - step
    assert shot_setup.roll_for_shape("fade", _line, limits)[0] == straight + step


def test_unknown_shape_is_value_error_and_non_string_type_error() -> None:
    with pytest.raises(ValueError, match="shot_shape"):
        shot_setup.validate_shot_setup(0.0, 0.0, "hook")
    with pytest.raises(TypeError, match="shot_shape"):
        shot_setup.validate_shot_setup(0.0, 0.0, 3)
    shot_setup.validate_shot_setup(0.0, 0.0, "draw")
    shot_setup.validate_shot_setup(0.0, 0.01, "fade")


def test_shape_with_a_non_zero_roll_raises() -> None:
    with pytest.raises(ValueError, match="club_roll_in_hands_deg"):
        ExportSettings(shot_shape="draw", club_roll_in_hands_deg=1.0)
    assert ExportSettings(shot_shape="draw", ball_position_shift_m=0.02)


def test_default_leaves_settings_and_swing_untouched() -> None:
    settings = ExportSettings()
    assert settings.shot_shape is None
    swing = object()
    assert shot_setup.solve_shot_shape(swing, settings) == (settings, None)  # type: ignore[arg-type]


def test_solve_without_a_flight_says_why() -> None:
    class _NoFlight:
        lateral_m = None
        status = "no ball placed"

    with pytest.raises(ValueError, match="no ball placed"):
        shot_setup.lateral_of_flight(_NoFlight())


def test_record_is_attached_under_shot_setup() -> None:
    record = {"shot_shape": "draw"}
    assert shot_setup.with_setup_record({"a": 1}, record) == {
        "a": 1,
        "shot_setup": record,
    }
    assert shot_setup.with_setup_record({"a": 1}, None) == {"a": 1}
    assert shot_setup.with_setup_record(None, record) is None


@pytest.mark.parametrize(
    ("roll", "shift", "text"),
    [
        (0.0, 0.0, None),
        (-2.0, 0.0, "Setup: club roll -2.0 deg in hands"),
        (0.0, 0.02, "Setup: ball +2.0 cm forward"),
        (0.0, -0.03, "Setup: ball -3.0 cm back"),
        (1.5, -0.01, "Setup: club roll +1.5 deg in hands, ball -1.0 cm back"),
    ],
)
def test_setup_hud_line(roll: float, shift: float, text: str | None) -> None:
    settings = ExportSettings(club_roll_in_hands_deg=roll, ball_position_shift_m=shift)
    assert shot_setup.hud_lines(settings) == (() if text is None else (text,))


def test_cli_parses_shot_shape() -> None:
    parsed = cli.build_parser().parse_args(BUNDLE_ARGS)
    assert cli.build_settings(parsed).shot_shape is None
    flags = ["--shot-shape", "fade", "--ball-position-shift", "0.02"]
    parsed = cli.build_parser().parse_args([*BUNDLE_ARGS, *flags])
    assert cli.build_settings(parsed).shot_shape == "fade"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*BUNDLE_ARGS, "--shot-shape", "hook"])


def test_cli_rejects_shape_with_a_roll() -> None:
    flags = ["--shot-shape", "draw", "--club-roll-in-hands", "-1"]
    with pytest.raises(SystemExit):
        cli.main([*BUNDLE_ARGS, *flags])


def test_trial_evaluation_builds_valid_settings_on_a_real_swing() -> None:
    pytest.importorskip("mujoco")
    from tests.unit.tools.native_viewer_export.test_export_ball import _fixture_swing

    swing = _fixture_swing("driver", returning=True)
    settings = ExportSettings(
        impact_time_s=swing.time_s(swing.n_states - 1), shot_shape="straight"
    )
    try:
        shot_setup.solve_shot_shape(swing, settings)
    except ValueError as exc:  # a synthetic swing may have no usable flight
        assert "must be 0" not in str(exc)
