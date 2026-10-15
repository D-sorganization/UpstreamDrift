"""Tests for segmental spine kinematics (SPN-1b, ADR-0055 items 1-3)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from src.shared.python.spine import (
    LevelTransform,
    SegmentalKinematicsSeries,
    SegmentalPose,
    SpinalLevel,
)

pytestmark = pytest.mark.unit

L = SpinalLevel


def _lt(level, deg, neutral=None, translation=None):
    return LevelTransform.from_isb_angles(
        level, np.deg2rad(deg), neutral=neutral, translation=translation
    )


def _neutral(rotvec, trans):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    t[:3, 3] = trans
    return t


def _worked_pose():
    return SegmentalPose(
        (
            _lt(L.L3_L4, (-10, 5, -15)),
            _lt(L.L2_L3, (-10, 8, -20)),
            _lt(L.L1_L2, (-15, 10, -25)),
        )
    )


def test_adr_worked_example_composed_angles_and_sum():
    pose = _worked_pose()
    composed = pose.compose()[:3, :3]
    from src.shared.python.biomechanics.joint_conventions import (
        anatomical_joint_angles,
    )

    got = anatomical_joint_angles(composed, "isb_spine", degrees=True).values
    np.testing.assert_allclose(got, (-25.362, 31.832, -57.573), atol=1e-3)
    np.testing.assert_allclose(
        np.rad2deg(pose.angle_sum_diagnostic()), (-35, 23, -60), atol=1e-9
    )


def test_angle_sum_differs_from_relative_rotation_angles():
    pose = SegmentalPose(
        (
            _lt(L.L3_L4, (20, 0, 0)),
            _lt(L.L2_L3, (0, 15, 0)),
            _lt(L.L1_L2, (0, 0, 30)),
        )
    )
    rel = LevelTransform(
        L.L3_L4,
        np.eye(4),
        pose.relative_rotation(L.L3_L4, L.L1_L2),
        np.zeros(3),
    ).isb_angles()
    assert not np.allclose(rel, pose.angle_sum_diagnostic(), atol=1e-3)


def test_compose_equals_explicit_product_with_neutral_offsets():
    rng = np.random.default_rng(3)
    levels = [L.L5_S1, L.L4_L5, L.L3_L4, L.L2_L3]
    tr = []
    for lv in levels:
        tr.append(
            LevelTransform(
                lv,
                _neutral(rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.03),
                Rotation.from_rotvec(rng.normal(size=3) * 0.4).as_matrix(),
                rng.normal(size=3) * 0.002,
            )
        )
    pose = SegmentalPose(tuple(tr))
    expected = tr[0].matrix() @ tr[1].matrix() @ tr[2].matrix() @ tr[3].matrix()
    np.testing.assert_allclose(pose.compose(), expected, atol=1e-12, rtol=0)
    sub = tr[1].matrix() @ tr[2].matrix()
    np.testing.assert_allclose(
        pose.compose(L.L4_L5, L.L3_L4), sub, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        pose.relative_rotation(L.L4_L5, L.L3_L4), sub[:3, :3], atol=1e-12
    )
    np.testing.assert_allclose(
        pose.compose(caudal=L.L3_L4), (tr[2].matrix() @ tr[3].matrix()), atol=1e-12
    )


_vec = st.lists(
    st.floats(-1.5, 1.5, allow_nan=False), min_size=3, max_size=3
).map(np.array)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(_vec, _vec), min_size=1, max_size=6))
def test_property_compose_matches_numpy_product(draws):
    levels = list(SpinalLevel)[2 : 2 + len(draws)]
    tr = [
        LevelTransform(
            lv,
            _neutral(n * 0.3, n * 0.01),
            Rotation.from_rotvec(r).as_matrix(),
            np.zeros(3),
        )
        for lv, (r, n) in zip(levels, draws)
    ]
    expected = np.eye(4)
    for t in tr:
        expected = expected @ t.matrix()
    np.testing.assert_allclose(
        SegmentalPose(tuple(tr)).compose(), expected, atol=1e-12, rtol=0
    )


def test_round_trip_away_from_gimbal_lock():
    ang = np.deg2rad([12.0, -7.0, 33.0])
    lt = LevelTransform.from_isb_angles(L.L4_L5, ang)
    np.testing.assert_allclose(lt.isb_angles(), ang, atol=1e-12)
    assert not bool(lt.isb_angles_result().singular)


def test_gimbal_lock_flagged_and_reconstructs():
    ang = np.array([0.3, np.pi / 2, 0.2])
    lt = LevelTransform.from_isb_angles(L.L4_L5, ang)
    res = lt.isb_angles_result()
    assert bool(res.singular)
    again = LevelTransform.from_isb_angles(L.L4_L5, res.values)
    np.testing.assert_allclose(again.rotation, lt.rotation, atol=1e-9)


def test_neutral_excluded_from_angles_included_in_compose():
    neutral = _neutral([0.0, 0.0, 0.3], [0.0, 0.02, 0.0])
    lt = LevelTransform.from_isb_angles(L.L4_L5, np.deg2rad([5, 2, 3]), neutral)
    np.testing.assert_allclose(lt.isb_angles(), np.deg2rad([5, 2, 3]), atol=1e-12)
    assert not np.allclose(lt.matrix()[:3, :3], lt.rotation)
    np.testing.assert_allclose(SegmentalPose((lt,)).compose(), lt.matrix())


def test_translation_enters_matrix():
    lt = LevelTransform.from_isb_angles(
        L.L4_L5, np.zeros(3), translation=np.array([0.0, 0.0, 0.001])
    )
    np.testing.assert_allclose(lt.matrix()[:3, 3], (0, 0, 0.001))


def test_arrays_read_only():
    lt = _lt(L.L4_L5, (1, 2, 3))
    for arr in (lt.neutral, lt.rotation, lt.translation):
        assert not arr.flags.writeable


def test_level_transform_preconditions():
    eye = np.eye(3)
    bad = eye.copy()
    bad[0, 0] = -1.0  # improper
    with pytest.raises(ValueError, match="proper"):
        LevelTransform(L.L4_L5, np.eye(4), bad, np.zeros(3))
    nan = eye.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        LevelTransform(L.L4_L5, np.eye(4), nan, np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        LevelTransform(L.L4_L5, np.eye(4), eye, np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError, match="shape"):
        LevelTransform(L.L4_L5, np.eye(3), eye, np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        LevelTransform(L.L4_L5, np.eye(4), eye, np.zeros(2))
    with pytest.raises(TypeError, match="SpinalLevel"):
        LevelTransform("L4_L5", np.eye(4), eye, np.zeros(3))  # type: ignore[arg-type]
    skew = np.eye(4)
    skew[3, :3] = 1.0
    with pytest.raises(ValueError, match="neutral"):
        LevelTransform(L.L4_L5, skew, eye, np.zeros(3))
    with pytest.raises(ValueError, match="proper"):
        LevelTransform(L.L4_L5, np.diag([2.0, 1, 1, 1]), eye, np.zeros(3))
    with pytest.raises(ValueError, match="angles"):
        LevelTransform.from_isb_angles(L.L4_L5, np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        LevelTransform.from_isb_angles(L.L4_L5, np.array([np.nan, 0, 0]))


def test_pose_preconditions():
    with pytest.raises(ValueError, match="empty|at least"):
        SegmentalPose(())
    with pytest.raises(ValueError, match="contiguous"):
        SegmentalPose((_lt(L.L5_S1, (0, 0, 0)), _lt(L.L3_L4, (0, 0, 0))))
    with pytest.raises(ValueError, match="contiguous|unique"):
        SegmentalPose((_lt(L.L4_L5, (0, 0, 0)), _lt(L.L4_L5, (0, 0, 0))))
    with pytest.raises(ValueError, match="contiguous"):
        SegmentalPose((_lt(L.L3_L4, (0, 0, 0)), _lt(L.L4_L5, (0, 0, 0))))
    with pytest.raises(TypeError):
        SegmentalPose(("x",))  # type: ignore[arg-type]
    pose = _worked_pose()
    with pytest.raises(ValueError, match="not in pose"):
        pose.compose(L.T1_T2, None)
    with pytest.raises(ValueError, match="caudal"):
        pose.compose(L.L1_L2, L.L3_L4)
    with pytest.raises(TypeError):
        pose.relative_rotation("L3_L4", L.L1_L2)  # type: ignore[arg-type]


def _series(**kw):
    poses = (_worked_pose(), _worked_pose())
    args = dict(
        time_s=np.array([0.0, 0.1]),
        poses=poses,
        distribution_source="measured:synthetic",
        valid=np.array([True, True]),
    )
    args.update(kw)
    return SegmentalKinematicsSeries(**args)


def test_series_ok_and_relative_rotations():
    s = _series()
    rr = s.relative_rotations(L.L3_L4, L.L1_L2)
    assert rr.shape == (2, 3, 3)
    np.testing.assert_allclose(
        rr[0], s.poses[0].relative_rotation(L.L3_L4, L.L1_L2), atol=1e-15
    )
    assert not s.time_s.flags.writeable and not s.valid.flags.writeable
    assert not rr.flags.writeable


def test_series_preconditions():
    with pytest.raises(ValueError, match="distribution_source"):
        _series(distribution_source="  ")
    with pytest.raises(TypeError, match="distribution_source"):
        _series(distribution_source=None)
    with pytest.raises(ValueError, match="increasing"):
        _series(time_s=np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="finite"):
        _series(time_s=np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="length"):
        _series(valid=np.array([True]))
    with pytest.raises(ValueError, match="length"):
        _series(poses=(_worked_pose(),))
    with pytest.raises(ValueError, match="level set"):
        _series(
            poses=(
                _worked_pose(),
                SegmentalPose((_lt(L.L3_L4, (0, 0, 0)),)),
            )
        )
    with pytest.raises(ValueError, match="bool"):
        _series(valid=np.array([1.0, 0.0]))
