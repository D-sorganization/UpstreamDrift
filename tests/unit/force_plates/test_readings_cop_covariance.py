"""Random x-y covariance of the CoP reading (rotated overlay CoP ellipses).

Each stance CoP channel carries its own random 1-sigma,
``hypot(declared CoP sigma, skew * stance rate)``, independently of the other
channel, so the stance covariance is diagonal: an assumption the caller must
declare (``ReadingOptions.cop_skew_independent``) whenever ``skew_s > 0``.  The plate yaw ``R`` carries it
to the world as ``R_xy diag(var) R_xy^T``: its principal axes are the plate's
axes, so the ellipse turns with the plate.  The per-axis ``|R_xy|`` bound of
``cop_sigma_random_xy_m`` is never smaller than the exact world-axis sigma.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.force_plates import (
    PlacedTrial,
    PlatePlacement,
    PlateTimeAlignment,
    ReadingOptions,
    reading_at,
)
from src.shared.python.force_plates.readings import FootSample
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_ALIGN = PlateTimeAlignment(
    offset_s=0.0, sigma_offset_s=0.001, kind="manual", note="test clock"
)
# Right CoP of the factory trial in [0, 0.005]: stance x rate 1 m/s, y rate
# 6 m/s; declared CoP sigma 0.010 m, skew 1 ms (as test_readings_cop_axes).
_BASE, _SKEW = 0.010, 1.0e-3
_OPTS = ReadingOptions(
    coverage=2.0, skew_s=_SKEW, cop_curvature_floor=1.0, cop_skew_independent=True
)
_VAR_X = _BASE**2 + (_SKEW * 1.0) ** 2
_VAR_Y = _BASE**2 + (_SKEW * 6.0) ** 2
_YAWS = [0.0, math.pi / 6, math.pi / 4, math.pi / 2, 2.0]


def _right(yaw: float, options: ReadingOptions = _OPTS) -> FootSample:
    placement = PlatePlacement.from_yaw(yaw, origin_m=[0.0, 0.0, 0.0])
    placed = PlacedTrial(tf.make_trial(), placement)
    return reading_at(placed, 0.0025, _ALIGN, options).feet["right"]


def _matrix(cov: tuple[float, float, float] | None) -> np.ndarray:
    assert cov is not None
    xx, yy, xy = cov
    return np.array([[xx, xy], [xy, yy]])


def test_unrotated_covariance_is_the_stance_variances() -> None:
    assert _right(0.0).cop_cov_random_xy_m2 == pytest.approx((_VAR_X, _VAR_Y, 0.0))


@pytest.mark.parametrize("yaw", _YAWS)
def test_covariance_is_the_stance_variances_rotated_by_the_yaw(yaw: float) -> None:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    expected = rot @ np.diag([_VAR_X, _VAR_Y]) @ rot.T
    got = _matrix(_right(yaw).cop_cov_random_xy_m2)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("yaw", _YAWS)
def test_principal_axes_turn_with_the_plate(yaw: float) -> None:
    values, vectors = np.linalg.eigh(_matrix(_right(yaw).cop_cov_random_xy_m2))
    assert values == pytest.approx([_VAR_X, _VAR_Y], rel=1e-12)
    major = vectors[:, 1]  # stance y, the larger variance
    assert abs(float(major @ [-math.sin(yaw), math.cos(yaw)])) == pytest.approx(1.0)


@pytest.mark.parametrize("yaw", _YAWS)
def test_per_axis_bound_is_never_below_the_exact_sigma(yaw: float) -> None:
    right = _right(yaw)
    xx, yy, _ = right.cop_cov_random_xy_m2
    sx, sy = right.cop_sigma_random_xy_m
    assert math.sqrt(xx) <= sx * (1 + 1e-12)
    assert math.sqrt(yy) <= sy * (1 + 1e-12)


def test_an_unloaded_cop_is_the_declared_variance_and_uncorrelated() -> None:
    placement = PlatePlacement.from_yaw(0.7, origin_m=[0.0, 0.0, 0.0])
    placed = PlacedTrial(tf.make_trial(), placement)
    left = reading_at(placed, 0.0075, _ALIGN, ReadingOptions(coverage=2.0)).feet["left"]
    assert left.cop_m is None
    xx, yy, xy = left.cop_cov_random_xy_m2
    assert (xx, yy) == pytest.approx((_BASE**2, _BASE**2))
    assert xy == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize(
    "bad",
    [
        (1e-4, 1e-4),
        (-1e-4, 1e-4, 0.0),
        (1e-4, math.nan, 0.0),
        (1e-4, 1e-4, math.inf),
        (1e-4, 1e-4, 2e-4),  # |xy| above sqrt(xx * yy): not a covariance
    ],
)
def test_covariance_is_validated(bad: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="cop_cov_random_xy_m2"):
        FootSample(
            load_n=1.0,
            sigma_load_n=0.0,
            cop_m=None,
            sigma_cop_m=0.0,
            sigma_sensing_m=0.0,
            cop_cov_random_xy_m2=bad,  # type: ignore[arg-type]
        )


def test_a_skewed_reading_without_declared_independence_has_no_covariance() -> None:
    # A skew shared by both channels would correlate them along the signed CoP
    # velocity, which the reading does not keep: no covariance is claimed.
    undeclared = ReadingOptions(coverage=2.0, skew_s=_SKEW, cop_curvature_floor=1.0)
    right = _right(0.5, undeclared)
    assert right.cop_cov_random_xy_m2 is None
    assert right.cop_sigma_random_xy_m == _right(0.5).cop_sigma_random_xy_m


def test_without_skew_the_covariance_needs_no_declaration() -> None:
    # No skew term: only the declared per-axis sigma remains, whatever the
    # channels' timing correlation.
    right = _right(0.5, ReadingOptions(coverage=2.0, cop_curvature_floor=1.0))
    xx, yy, xy = right.cop_cov_random_xy_m2
    assert (xx, yy) == pytest.approx((_BASE**2, _BASE**2))
    assert xy == pytest.approx(0.0, abs=1e-18)


def test_skew_independence_must_be_a_bool() -> None:
    with pytest.raises(TypeError, match="cop_skew_independent"):
        ReadingOptions(coverage=2.0, cop_skew_independent=1)  # type: ignore[arg-type]
