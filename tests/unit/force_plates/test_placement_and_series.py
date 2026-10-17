"""OF-01 (#13061): stance-to-world placement and the ground-reaction series view."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import (
    NET_LABEL,
    GroundReactionSeries,
)
from src.shared.python.force_plates import (
    ObservedGroundReaction,
    PlatePlacement,
    observability,
    to_ground_reaction_series,
)
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


# --------------------------------------------------------------- placement --
def test_identity_placement_lifts_stance_points_onto_the_ground_plane() -> None:
    world = PlatePlacement.identity().to_world(np.array([[0.1, -0.2]]))
    np.testing.assert_allclose(world, [[0.1, -0.2, 0.0]])


def test_yaw_placement_maps_a_known_point() -> None:
    placement = PlatePlacement.from_yaw(math.pi / 2, origin_m=(1.0, 2.0, 0.022))
    # +x of the stance frame becomes +y of the world after a quarter turn.
    np.testing.assert_allclose(
        placement.to_world(np.array([[0.3, 0.0]])), [[1.0, 2.3, 0.022]], atol=1e-12
    )
    assert placement.surface_height_m == pytest.approx(0.022)


def test_placement_keeps_nan_rows_nan() -> None:
    world = PlatePlacement.identity().to_world(np.array([[np.nan, np.nan]]))
    assert np.isnan(world).all()


@pytest.mark.parametrize(
    "rotation",
    [
        # tilt about x: the vertical axis would lean
        [
            [1, 0, 0],
            [0, math.cos(0.1), -math.sin(0.1)],
            [0, math.sin(0.1), math.cos(0.1)],
        ],
        # mirror x: handedness would flip silently
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # not orthonormal
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_placement_rejects_tilt_mirror_and_scale(rotation) -> None:
    with pytest.raises(ValueError, match="rotation"):
        PlatePlacement(rotation=np.array(rotation, float), origin_m=np.zeros(3))


def test_facing_constructor_turns_stance_forward_onto_the_given_direction() -> None:
    # ADR-0041: the golfer faces world -X; stance +x (toward the right foot) then
    # lies along world +Y, so a right-handed golfer's target (-x stance) is -Y.
    placement = PlatePlacement.facing((-1.0, 0.0), origin_m=(0.0, 0.0, 0.0))
    np.testing.assert_allclose(
        placement.to_world(np.array([[0.0, 1.0], [1.0, 0.0]])),
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        atol=1e-12,
    )


@pytest.mark.parametrize("facing", [(0.0, 0.0), (np.nan, 1.0), (1.0, 2.0, 3.0)])
def test_facing_must_be_a_finite_horizontal_direction(facing) -> None:
    with pytest.raises(ValueError, match="facing"):
        PlatePlacement.facing(facing, origin_m=(0.0, 0.0, 0.0))


@pytest.mark.parametrize("yaw", [-2.0, -0.3, 0.0, 0.7, 3.0])
def test_yaw_preserves_distances_and_the_surface_height(yaw) -> None:
    placement = PlatePlacement.from_yaw(yaw, origin_m=(0.4, -0.1, 0.03))
    pts = np.array([[-0.185, 0.02], [0.185, -0.02], [np.nan, np.nan]])
    world = placement.to_world(pts)
    assert np.linalg.norm(world[0] - world[1]) == pytest.approx(
        np.linalg.norm(pts[0] - pts[1])
    )
    np.testing.assert_allclose(world[:2, 2], 0.03)
    assert np.isnan(world[2]).all()


def test_from_yaw_rejects_nan_and_short_origins() -> None:
    with pytest.raises(ValueError, match="yaw_rad"):
        PlatePlacement.from_yaw(np.nan, origin_m=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="origin_m"):
        PlatePlacement.from_yaw(0.0, origin_m=(0.0, 0.0))


def test_placement_rejects_bad_origin() -> None:
    with pytest.raises(ValueError, match="origin_m"):
        PlatePlacement(rotation=np.eye(3), origin_m=np.array([0.0, np.nan, 0.0]))


def test_to_world_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="stance"):
        PlatePlacement.identity().to_world(np.zeros((2, 3)))


# ------------------------------------------------------------ observability --
def test_observability_names_only_vertical_load_and_cop() -> None:
    assert observability() == {
        "fx": False,
        "fy": False,
        "fz": True,
        "cop": True,
        "free_moment": False,
    }


# ------------------------------------------------------------------ series --
def _observed() -> ObservedGroundReaction:
    placement = PlatePlacement.from_yaw(0.0, origin_m=(0.0, 0.0, 0.02))
    return to_ground_reaction_series(tf.make_trial(), placement)


def _series() -> GroundReactionSeries:
    return _observed().series


def test_series_carries_its_observability_record() -> None:
    observed = _observed()
    assert observed.observed == observability()
    assert observed.is_observed("fz") and not observed.is_observed("fx")
    with pytest.raises(ValueError, match="component"):
        observed.is_observed("torque")


def test_series_is_a_ground_reaction_series_keyed_by_feet_and_net() -> None:
    series = _series()
    assert isinstance(series, GroundReactionSeries)
    assert tuple(series.force_n) == ("left", "right", NET_LABEL)
    np.testing.assert_allclose(series.times_s, tf.TIMES_S)


def test_series_force_is_vertical_with_unobserved_shear_as_nan_not_zero() -> None:
    series = _series()
    for key in series.force_n:
        assert np.isnan(series.force_n[key][:, :2]).all()
    np.testing.assert_allclose(series.force_n["left"][:, 2], tf.LEFT_N)
    np.testing.assert_allclose(series.force_n[NET_LABEL][:, 2], [800.0] * 3)


def test_series_free_moment_and_com_terms_are_unobserved() -> None:
    series = _series()
    for table in (
        series.free_moment_nm,
        series.moment_about_com_nm,
        series.force_moment_about_com_nm,
    ):
        for key in table:
            assert np.isnan(table[key]).all()
    assert np.isnan(series.com_m).all()


def test_series_cop_is_placed_on_the_plate_surface() -> None:
    series = _series()
    np.testing.assert_allclose(series.cop_m["right"][0], [0.185, -0.02, 0.02])
    np.testing.assert_allclose(series.cop_m[NET_LABEL][0], [0.0, 0.0, 0.02], atol=1e-12)
    assert np.isnan(series.cop_m["left"][2]).all()
    np.testing.assert_allclose(series.ground_height_m, [0.02] * 3)


def test_series_contact_follows_the_source_threshold() -> None:
    series = _series()
    np.testing.assert_array_equal(series.in_contact["left"], [True, True, False])
    np.testing.assert_array_equal(series.in_contact[NET_LABEL], [True, True, True])


def test_net_contact_means_any_foot_in_contact() -> None:
    # 15 N per foot: 30 N in total but neither foot reaches the 20 N threshold.
    trial = tf.make_trial(
        times_s=np.array([0.0, 0.005]),
        load_n={"left": np.array([15.0, 400.0]), "right": np.array([15.0, 400.0])},
        cop_m={
            "left": np.array([[np.nan, np.nan], [-0.2, 0.0]]),
            "right": np.array([[np.nan, np.nan], [0.2, 0.0]]),
            "net": np.array([[np.nan, np.nan], [0.0, 0.0]]),
        },
        events_s={},
    )
    series = to_ground_reaction_series(trial, PlatePlacement.identity()).series
    np.testing.assert_array_equal(series.in_contact[NET_LABEL], [False, True])


def test_net_series_cop_is_hand_computed_after_placement() -> None:
    np.testing.assert_allclose(_series().cop_m[NET_LABEL][1], [0.05125, 0.00625, 0.02])


def test_observability_record_is_validated() -> None:
    series = _series()
    with pytest.raises(ValueError, match="observed"):
        ObservedGroundReaction(series=series, observed={"bogus": True})
    bad = {**observability(), "fz": 1}
    with pytest.raises(ValueError, match="observed"):
        ObservedGroundReaction(series=series, observed=bad)


def test_series_rejects_wrong_types() -> None:
    with pytest.raises(TypeError, match="trial"):
        to_ground_reaction_series("trial", PlatePlacement.identity())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="placement"):
        to_ground_reaction_series(tf.make_trial(), None)  # type: ignore[arg-type]


def test_unobserved_tables_do_not_share_arrays() -> None:
    # The series is a mutable snapshot: filling one unobserved table must not
    # silently change another.
    series = _series()
    series.moment_about_com_nm["left"][0, 0] = 1.0
    assert np.isnan(series.free_moment_nm["left"][0, 0])
    assert np.isnan(series.force_moment_about_com_nm["left"][0, 0])
