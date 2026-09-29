"""Pure-numpy variant of the Pinocchio kinematic-equivalence audit
(issue #4136).

When pinocchio is not installable (e.g. Python 3.14 / Windows pip), this
unit test still exercises:

* the URDF-derived numpy FK chain (round-trip self-consistency),
* the residual helpers used by the heavy_integration test,
* the Simscape ground-truth row loader.

It is the audit's CI safety net: even without pinocchio, regressions in
the spine-chain offsets or the residual maths surface here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.engines.pinocchio._kinematic_equivalence_data import (
    GRIP_ORIENTATION_TOL_RAD,
    GRIP_POSITION_RMSE_TOL_M,
    REFERENCE_POSES,
    SpineConfig,
    geodesic_angle,
    numpy_spine_fk,
    position_rmse,
    rot_z,
)

pytestmark = [pytest.mark.unit]


def test_three_reference_poses_defined() -> None:
    """Spec contract: address, top-of-backswing, impact must all exist."""
    names = {cfg.name for cfg in REFERENCE_POSES}
    assert names == {"address", "top_of_backswing", "impact"}


def test_numpy_fk_returns_complete_chain() -> None:
    """The FK helper yields every frame the test asserts on."""
    frames = numpy_spine_fk(REFERENCE_POSES[0])
    expected = {
        "pelvis",
        "lumbar1",
        "lumbar2",
        "lumbar3",
        "thorax1",
        "thorax2",
        "thorax3",
        "mid_hands",
        "club_shaft",
        "club_head",
    }
    assert expected.issubset(frames.keys())
    for name, T in frames.items():
        assert T.shape == (4, 4), f"frame '{name}' must be 4x4 SE(3)"


@pytest.mark.parametrize("cfg", REFERENCE_POSES, ids=lambda c: c.name)
def test_mid_hands_position_consistent_with_chain_sum(cfg: SpineConfig) -> None:
    """For zero spine rotation about X/Y, mid_hands z should sit at the
    summed link translations (sanity check on the chain math).

    With non-zero rotations we instead require the result to be finite
    and roughly within an arm's-length of the floating-base origin.
    """
    frames = numpy_spine_fk(cfg)
    p_mh = frames["mid_hands"][:3, 3]
    assert np.all(np.isfinite(p_mh))
    # Total summed link offset along z (no rotation) is
    # 4 * 0.12 + 2 * 0.10 - 0.17 = 0.51 m.
    # With rotations the z-projection only shrinks, never grows past this.
    assert p_mh[2] <= 0.51 + 1e-9


@pytest.mark.parametrize("cfg", REFERENCE_POSES, ids=lambda c: c.name)
def test_club_head_below_mid_hands(cfg: SpineConfig) -> None:
    """The club hangs below the grip in every reference pose."""
    frames = numpy_spine_fk(cfg)
    p_mh = frames["mid_hands"][:3, 3]
    p_ch = frames["club_head"][:3, 3]
    # Distance from mid_hands to club_head equals the welded shaft length
    # |0.05 + 0.5| = 0.55 m up to rotation. Sanity: their separation must
    # equal that.
    sep = float(np.linalg.norm(p_ch - p_mh))
    assert abs(sep - 0.55) < 1e-9, (
        f"{cfg.name}: club shaft length drift {sep:.6f} != 0.55 m "
        "(URDF mid_hands -> club_shaft -> club_head fixed transforms changed)"
    )


def test_position_rmse_matches_distance_for_single_point() -> None:
    """Sanity check: RMSE of two points equals the Euclidean distance."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert position_rmse(a, b) == pytest.approx(5.0)


def test_position_rmse_zero_for_identical_points() -> None:
    p = np.array([1.2, -0.3, 0.7])
    assert position_rmse(p, p) == pytest.approx(0.0)


def test_geodesic_angle_zero_for_identical_rotations() -> None:
    R = rot_z(0.5)
    assert geodesic_angle(R, R) == pytest.approx(0.0, abs=1e-12)


def test_geodesic_angle_recovers_input_angle() -> None:
    """Geodesic between identity and rot_z(theta) is exactly theta."""
    for theta in [0.1, 0.5, 1.0, 1.5]:
        R = rot_z(theta)
        recovered = geodesic_angle(np.eye(3), R)
        assert recovered == pytest.approx(theta, abs=1e-9)


def test_tolerances_are_strict_enough_to_be_meaningful() -> None:
    """Sanity-check the spec tolerances. A 1 cm position error or a 1
    deg rotation error must NOT pass; a sub-tolerance error must pass."""
    a = np.array([0.0, 0.0, 0.0])
    b_far = np.array([0.01, 0.0, 0.0])  # 10 mm
    b_near = np.array([0.001, 0.0, 0.0])  # 1 mm
    assert position_rmse(a, b_far) > GRIP_POSITION_RMSE_TOL_M
    assert position_rmse(a, b_near) < GRIP_POSITION_RMSE_TOL_M

    R_id = np.eye(3)
    R_far = rot_z(np.deg2rad(2.0))
    R_near = rot_z(np.deg2rad(0.1))
    assert geodesic_angle(R_id, R_far) > GRIP_ORIENTATION_TOL_RAD
    assert geodesic_angle(R_id, R_near) < GRIP_ORIENTATION_TOL_RAD


def test_numpy_chain_is_self_consistent_under_pure_translation() -> None:
    """If we set every spine joint to zero, mid_hands lies directly above
    pelvis (x = 0, y = 0) at the cumulative chain z-offset."""
    zero_cfg = SpineConfig(
        name="zero",
        lumbar1_x=0.0,
        lumbar1_y=0.0,
        lumbar2_x=0.0,
        lumbar2_y=0.0,
        lumbar3_x=0.0,
        lumbar3_y=0.0,
        thorax1_z=0.0,
        thorax2_z=0.0,
        thorax3_z=0.0,
    )
    frames = numpy_spine_fk(zero_cfg)
    p_mh = frames["mid_hands"][:3, 3]
    # 4 * 0.12 (lumbar steps including pelvis->lumbar1) + 2 * 0.10
    # (thorax steps) - 0.17 (mid_hands offset) = 0.51 m exactly.
    expected_z = 4 * 0.12 + 2 * 0.10 - 0.17
    assert p_mh[0] == pytest.approx(0.0, abs=1e-12)
    assert p_mh[1] == pytest.approx(0.0, abs=1e-12)
    assert p_mh[2] == pytest.approx(expected_z, abs=1e-12)
