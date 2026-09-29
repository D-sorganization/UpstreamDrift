"""Unit tests for ``ClubBallTarget`` and the default ball-impact extractor."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.motion_matching.club_ball_target import (
    DEFAULT_ELASTICITY_FACTOR,
    MAX_LAUNCH_SPEED_MPS,
    MAX_SPIN_RPM,
    BallImpactState,
    ClubBallTarget,
    extract_ball_impact_from_clubtarget,
)
from src.shared.python.motion_matching.club_target import (
    MAX_POSITION_NORM_M,
    ClubTarget,
)

from ._fixtures import make_provenance, make_target

# --- BallImpactState happy path ------------------------------------------


def _good_ball_impact_state(**overrides: object) -> BallImpactState:
    """Build a valid ``BallImpactState`` with optional field overrides."""
    kwargs: dict[str, object] = {
        "position_at_impact_m": np.array([0.0, 0.5, 0.05]),
        "launch_direction": np.array([1.0, 0.0, 0.0]),
        "launch_speed_mps": 60.0,
        "spin_rpm": 2500.0,
    }
    kwargs.update(overrides)
    return BallImpactState(**kwargs)  # type: ignore[arg-type]


def test_ball_impact_state_happy_path() -> None:
    state = _good_ball_impact_state()
    assert state.launch_speed_mps == 60.0
    assert np.isclose(np.linalg.norm(state.launch_direction), 1.0)


def test_ball_impact_state_is_frozen() -> None:
    state = _good_ball_impact_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.launch_speed_mps = 70.0  # type: ignore[misc]


def test_ball_impact_state_allows_all_unknown_optional_fields() -> None:
    state = _good_ball_impact_state(
        launch_direction=np.full(3, np.nan),
        launch_speed_mps=float("nan"),
        spin_rpm=float("nan"),
    )
    assert np.all(np.isnan(state.launch_direction))
    assert np.isnan(state.launch_speed_mps)
    assert np.isnan(state.spin_rpm)


# --- BallImpactState validation rules ------------------------------------


def test_position_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="shape \\(3,\\)"):
        _good_ball_impact_state(position_at_impact_m=np.zeros(4))


def test_position_nan_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        _good_ball_impact_state(position_at_impact_m=np.array([np.nan, 0.0, 0.0]))


def test_position_implausible_radius_rejected() -> None:
    big = np.array([MAX_POSITION_NORM_M + 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="\\|r\\| >="):
        _good_ball_impact_state(position_at_impact_m=big)


def test_launch_direction_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="shape \\(3,\\)"):
        _good_ball_impact_state(launch_direction=np.zeros(4))


def test_launch_direction_partial_nan_rejected() -> None:
    bad = np.array([np.nan, 0.0, 1.0])
    with pytest.raises(ValueError, match="fully NaN or fully finite"):
        _good_ball_impact_state(launch_direction=bad)


def test_launch_direction_non_unit_rejected() -> None:
    with pytest.raises(ValueError, match="unit-norm"):
        _good_ball_impact_state(launch_direction=np.array([2.0, 0.0, 0.0]))


def test_launch_direction_inf_rejected() -> None:
    bad = np.array([np.inf, 0.0, 0.0])
    with pytest.raises(ValueError, match="Inf"):
        _good_ball_impact_state(launch_direction=bad)


def test_launch_speed_negative_rejected() -> None:
    with pytest.raises(ValueError, match="launch_speed_mps"):
        _good_ball_impact_state(launch_speed_mps=-1.0)


def test_launch_speed_too_large_rejected() -> None:
    with pytest.raises(ValueError, match="launch_speed_mps"):
        _good_ball_impact_state(launch_speed_mps=MAX_LAUNCH_SPEED_MPS + 1.0)


def test_launch_speed_inf_rejected() -> None:
    with pytest.raises(ValueError, match="launch_speed_mps"):
        _good_ball_impact_state(launch_speed_mps=float("inf"))


def test_spin_negative_rejected() -> None:
    with pytest.raises(ValueError, match="spin_rpm"):
        _good_ball_impact_state(spin_rpm=-10.0)


def test_spin_too_large_rejected() -> None:
    with pytest.raises(ValueError, match="spin_rpm"):
        _good_ball_impact_state(spin_rpm=MAX_SPIN_RPM + 1.0)


# --- ClubBallTarget composite -------------------------------------------


def test_club_ball_target_happy_path() -> None:
    club = make_target()
    state = _good_ball_impact_state()
    composite = ClubBallTarget(club=club, ball_impact=state)
    # Delegating properties.
    assert composite.impact_idx == club.impact_idx
    assert composite.time is club.time


def test_club_ball_target_is_frozen() -> None:
    composite = ClubBallTarget(
        club=make_target(), ball_impact=_good_ball_impact_state()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        composite.club = make_target()  # type: ignore[misc]


def test_club_ball_target_rejects_wrong_club_type() -> None:
    with pytest.raises(TypeError, match="ClubTarget"):
        ClubBallTarget(  # type: ignore[arg-type]
            club="not a club target",
            ball_impact=_good_ball_impact_state(),
        )


def test_club_ball_target_rejects_wrong_ball_type() -> None:
    with pytest.raises(TypeError, match="BallImpactState"):
        ClubBallTarget(  # type: ignore[arg-type]
            club=make_target(), ball_impact={"not": "a state"}
        )


# --- Default extractor ---------------------------------------------------


def _linear_clubhead_target(
    velocity: np.ndarray, *, n: int = 51, impact_idx: int = 25
) -> ClubTarget:
    """Synthetic ClubTarget whose clubhead moves at constant ``velocity``."""
    time = np.linspace(0.0, 0.3, n)
    butt = np.zeros((n, 3))
    clubhead = np.array([0.0, 0.0, 1.0])[None, :] + velocity[None, :] * time[:, None]
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=impact_idx,
        source=make_provenance(),
    )


def test_extractor_position_matches_clubhead_at_impact() -> None:
    target = _linear_clubhead_target(np.array([10.0, 0.0, 0.0]))
    state = extract_ball_impact_from_clubtarget(target)
    expected = target.clubhead[target.impact_idx - 1]
    assert np.allclose(state.position_at_impact_m, expected)


def test_extractor_launch_direction_matches_velocity_unit() -> None:
    velocity = np.array([8.0, 6.0, 0.0])  # |v| = 10
    target = _linear_clubhead_target(velocity)
    state = extract_ball_impact_from_clubtarget(target)
    expected_dir = velocity / np.linalg.norm(velocity)
    assert np.allclose(state.launch_direction, expected_dir, atol=1e-10)


def test_extractor_launch_speed_uses_elasticity_factor() -> None:
    velocity = np.array([10.0, 0.0, 0.0])
    target = _linear_clubhead_target(velocity)
    state = extract_ball_impact_from_clubtarget(target)
    expected = 10.0 * DEFAULT_ELASTICITY_FACTOR
    assert state.launch_speed_mps == pytest.approx(expected, rel=1e-9)


def test_extractor_launch_speed_clamped_to_max() -> None:
    # Build a tiny linear-motion target so |position| stays inside the
    # ClubTarget envelope while clubhead speed is comfortably above
    # MAX_LAUNCH_SPEED_MPS / DEFAULT_ELASTICITY_FACTOR.
    n = 3
    time = np.linspace(0.0, 1.0e-3, n)  # 1 ms window
    velocity = np.array([80.0, 0.0, 0.0])  # 80 m/s -> 120 m/s ball
    clubhead = np.array([0.0, 0.0, 1.0])[None, :] + velocity[None, :] * time[:, None]
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    target = ClubTarget(
        time=time,
        butt=np.zeros((n, 3)),
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=2,
        source=make_provenance(),
    )
    state = extract_ball_impact_from_clubtarget(target)
    assert state.launch_speed_mps == pytest.approx(MAX_LAUNCH_SPEED_MPS)


def test_extractor_spin_is_nan() -> None:
    target = _linear_clubhead_target(np.array([10.0, 0.0, 0.0]))
    state = extract_ball_impact_from_clubtarget(target)
    assert np.isnan(state.spin_rpm)


def test_extractor_zero_velocity_yields_nan_direction() -> None:
    n = 51
    time = np.linspace(0.0, 0.3, n)
    clubhead = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    target = ClubTarget(
        time=time,
        butt=np.zeros((n, 3)),
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=25,
        source=make_provenance(),
    )
    state = extract_ball_impact_from_clubtarget(target)
    assert np.all(np.isnan(state.launch_direction))
    assert state.launch_speed_mps == pytest.approx(0.0)


def test_extractor_works_at_first_frame() -> None:
    target = _linear_clubhead_target(np.array([1.0, 0.0, 0.0]), impact_idx=1)
    state = extract_ball_impact_from_clubtarget(target)
    assert np.allclose(state.launch_direction, [1.0, 0.0, 0.0])


def test_extractor_works_at_last_frame() -> None:
    n = 51
    target = _linear_clubhead_target(np.array([1.0, 0.0, 0.0]), n=n, impact_idx=n)
    state = extract_ball_impact_from_clubtarget(target)
    assert np.allclose(state.launch_direction, [1.0, 0.0, 0.0])


def test_extractor_rejects_wrong_target_type() -> None:
    with pytest.raises(TypeError, match="ClubTarget"):
        extract_ball_impact_from_clubtarget("not a target")  # type: ignore[arg-type]


def test_extractor_rejects_negative_elasticity() -> None:
    target = _linear_clubhead_target(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="elasticity_factor"):
        extract_ball_impact_from_clubtarget(target, elasticity_factor=-0.1)


def test_extractor_rejects_non_finite_elasticity() -> None:
    target = _linear_clubhead_target(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="elasticity_factor"):
        extract_ball_impact_from_clubtarget(target, elasticity_factor=float("nan"))


def test_extractor_zero_elasticity_yields_zero_speed() -> None:
    target = _linear_clubhead_target(np.array([10.0, 0.0, 0.0]))
    state = extract_ball_impact_from_clubtarget(target, elasticity_factor=0.0)
    assert state.launch_speed_mps == pytest.approx(0.0)


# --- Re-exports through target.py and the package root -------------------


def test_re_exported_from_target_module() -> None:
    from src.shared.python.motion_matching import target as target_mod

    assert target_mod.BallImpactState is BallImpactState
    assert target_mod.ClubBallTarget is ClubBallTarget
    assert (
        target_mod.extract_ball_impact_from_clubtarget
        is extract_ball_impact_from_clubtarget
    )


def test_re_exported_from_package_root() -> None:
    import src.shared.python.motion_matching as mm

    assert mm.BallImpactState is BallImpactState
    assert mm.ClubBallTarget is ClubBallTarget
    assert mm.extract_ball_impact_from_clubtarget is extract_ball_impact_from_clubtarget
