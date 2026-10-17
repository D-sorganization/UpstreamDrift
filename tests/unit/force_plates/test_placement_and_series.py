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
def _series() -> GroundReactionSeries:
    placement = PlatePlacement.from_yaw(0.0, origin_m=(0.0, 0.0, 0.02))
    return to_ground_reaction_series(tf.make_trial(), placement)


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


def test_series_rejects_wrong_types() -> None:
    with pytest.raises(TypeError, match="trial"):
        to_ground_reaction_series("trial", PlatePlacement.identity())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="placement"):
        to_ground_reaction_series(tf.make_trial(), None)  # type: ignore[arg-type]
