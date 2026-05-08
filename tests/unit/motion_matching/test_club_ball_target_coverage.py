"""Coverage tests for ``club_ball_target`` validation paths."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.club_ball_target import (
    DEFAULT_ELASTICITY_FACTOR,
    LAUNCH_DIR_NORM_TOL,
    MAX_LAUNCH_SPEED_MPS,
    BallImpactState,
    ClubBallTarget,
    extract_ball_impact_from_clubtarget,
)

from ._fixtures import make_target


def _good_state(**overrides) -> BallImpactState:
    base = {
        "position_at_impact_m": np.array([0.0, 0.0, 0.0]),
        "launch_direction": np.array([1.0, 0.0, 0.0]),
        "launch_speed_mps": 50.0,
        "spin_rpm": 1000.0,
    }
    base.update(overrides)
    return BallImpactState(**base)


# --- BallImpactState validation --------------------------------------------


def test_position_must_be_ndarray() -> None:
    """Pin: list position rejected with TypeError."""
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        _good_state(position_at_impact_m=[0.0, 0.0, 0.0])


def test_position_shape() -> None:
    """Pin: wrong-shape position rejected."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        _good_state(position_at_impact_m=np.zeros(2))


def test_position_finite() -> None:
    """Pin: NaN position rejected."""
    with pytest.raises(ValueError, match="NaN or Inf"):
        _good_state(position_at_impact_m=np.array([np.nan, 0.0, 0.0]))


def test_position_norm_too_big() -> None:
    """Pin: large-norm position rejected."""
    with pytest.raises(ValueError, match=r"\|r\| >="):
        _good_state(position_at_impact_m=np.array([100.0, 0.0, 0.0]))


def test_launch_direction_type() -> None:
    """Pin: non-ndarray direction rejected."""
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        _good_state(launch_direction=[1.0, 0, 0])


def test_launch_direction_shape() -> None:
    """Pin: wrong-shape direction rejected."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        _good_state(launch_direction=np.zeros(2))


def test_launch_direction_partial_nan_rejected() -> None:
    """Pin: partial-NaN direction rejected."""
    with pytest.raises(ValueError, match="fully NaN or fully finite"):
        _good_state(launch_direction=np.array([np.nan, 0.0, 1.0]))


def test_launch_direction_inf_rejected() -> None:
    """Pin: ``inf`` in direction rejected."""
    with pytest.raises(ValueError, match="contains Inf"):
        _good_state(launch_direction=np.array([np.inf, 0.0, 0.0]))


def test_launch_direction_non_unit_rejected() -> None:
    """Pin: non-unit direction rejected."""
    with pytest.raises(ValueError, match="unit-norm"):
        _good_state(launch_direction=np.array([2.0, 0.0, 0.0]))


def test_launch_direction_all_nan_ok() -> None:
    """Pin: fully-NaN direction is the documented 'unknown' sentinel."""
    s = _good_state(launch_direction=np.full(3, np.nan))
    assert np.all(np.isnan(s.launch_direction))


def test_launch_speed_negative_rejected() -> None:
    """Pin: negative launch speed rejected."""
    with pytest.raises(ValueError, match=r"launch_speed_mps must be in"):
        _good_state(launch_speed_mps=-1.0)


def test_launch_speed_too_big_rejected() -> None:
    """Pin: launch speed above the bound rejected."""
    with pytest.raises(ValueError, match=r"launch_speed_mps must be in"):
        _good_state(launch_speed_mps=MAX_LAUNCH_SPEED_MPS + 1.0)


def test_launch_speed_inf_rejected() -> None:
    """Pin: ``inf`` launch speed rejected as non-finite."""
    with pytest.raises(ValueError, match="finite or NaN"):
        _good_state(launch_speed_mps=float("inf"))


def test_launch_speed_nan_ok() -> None:
    """Pin: NaN launch speed is the 'unknown' sentinel."""
    s = _good_state(launch_speed_mps=float("nan"))
    assert np.isnan(s.launch_speed_mps)


def test_spin_rpm_too_big_rejected() -> None:
    """Pin: spin above the bound rejected."""
    with pytest.raises(ValueError, match="spin_rpm"):
        _good_state(spin_rpm=20_000.0)


# --- ClubBallTarget --------------------------------------------------------


def test_clubball_target_type_checks() -> None:
    """Pin: ClubBallTarget rejects non-ClubTarget club and non-state ball."""
    state = _good_state()
    with pytest.raises(TypeError, match="club must be a ClubTarget"):
        ClubBallTarget(club="not a target", ball_impact=state)  # type: ignore[arg-type]
    t = make_target()
    with pytest.raises(TypeError, match="ball_impact must be a BallImpactState"):
        ClubBallTarget(club=t, ball_impact={})  # type: ignore[arg-type]


def test_clubball_target_delegates_time_and_impact() -> None:
    """Pin: time/impact_idx delegate to the underlying ClubTarget."""
    t = make_target()
    cbt = ClubBallTarget(club=t, ball_impact=_good_state())
    assert np.array_equal(cbt.time, t.time)
    assert cbt.impact_idx == t.impact_idx


# --- extract_ball_impact_from_clubtarget -----------------------------------


def test_extract_requires_clubtarget() -> None:
    """Pin: extractor rejects non-ClubTarget input."""
    with pytest.raises(TypeError, match="must be a ClubTarget"):
        extract_ball_impact_from_clubtarget("nope")  # type: ignore[arg-type]


def test_extract_invalid_elasticity() -> None:
    """Pin: NaN/negative elasticity factor rejected."""
    t = make_target()
    with pytest.raises(ValueError, match="elasticity_factor"):
        extract_ball_impact_from_clubtarget(t, elasticity_factor=float("nan"))
    with pytest.raises(ValueError, match="elasticity_factor"):
        extract_ball_impact_from_clubtarget(t, elasticity_factor=-1.0)


def test_extract_returns_validated_state() -> None:
    """Pin: extracted state is a valid BallImpactState."""
    t = make_target()
    s = extract_ball_impact_from_clubtarget(
        t, elasticity_factor=DEFAULT_ELASTICITY_FACTOR
    )
    assert isinstance(s, BallImpactState)
    # spin is unknown (NaN) per the documented fall-back.
    assert np.isnan(s.spin_rpm)


def test_extract_clamps_speed() -> None:
    """Pin: huge elasticity gets clamped to MAX_LAUNCH_SPEED_MPS."""
    t = make_target()
    s = extract_ball_impact_from_clubtarget(t, elasticity_factor=1e6)
    assert s.launch_speed_mps <= MAX_LAUNCH_SPEED_MPS


def test_extract_marks_direction_unknown_when_velocity_zero() -> None:
    """Pin: a stationary clubhead at impact yields NaN launch direction."""
    import hashlib

    from src.shared.python.motion_matching.club_target import (
        ClubTarget,
        SourceProvenance,
    )

    n = 11
    time = np.linspace(0.0, 0.1, n)
    butt = np.zeros((n, 3))
    clubhead = np.zeros((n, 3))  # stationary -> zero velocity
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    t = ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=n // 2,
        source=SourceProvenance(
            filename="x",
            format="synthetic",
            subject_id="u",
            trial_id="0",
            sha256=hashlib.sha256(b"").hexdigest(),
        ),
    )
    s = extract_ball_impact_from_clubtarget(t, elasticity_factor=1.0)
    assert np.all(np.isnan(s.launch_direction))


def test_extract_velocity_endpoints() -> None:
    """Pin: impact at index 1 and last index uses one-sided differencing."""
    import hashlib

    from src.shared.python.motion_matching.club_target import (
        ClubTarget,
        SourceProvenance,
    )

    n = 5
    time = np.linspace(0.0, 0.04, n)
    butt = np.zeros((n, 3))
    clubhead = np.column_stack([time, np.zeros(n), np.zeros(n)])  # moves +x
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    prov = SourceProvenance(
        filename="x",
        format="synthetic",
        subject_id="u",
        trial_id="0",
        sha256=hashlib.sha256(b"").hexdigest(),
    )
    # impact_idx is 1-based; ``impact_idx=1`` -> k==0 endpoint branch.
    t1 = ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=1,
        source=prov,
    )
    s1 = extract_ball_impact_from_clubtarget(t1, elasticity_factor=1.0)
    # 1.0 m/s velocity in +x; tolerant equality.
    assert s1.launch_direction[0] > 1.0 - LAUNCH_DIR_NORM_TOL

    # impact_idx == n -> k==n-1 endpoint branch.
    tn = ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=n,
        source=prov,
    )
    sn = extract_ball_impact_from_clubtarget(tn, elasticity_factor=1.0)
    assert sn.launch_direction[0] > 1.0 - LAUNCH_DIR_NORM_TOL
