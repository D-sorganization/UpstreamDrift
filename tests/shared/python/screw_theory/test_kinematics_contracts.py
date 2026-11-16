"""Boundary contracts and invariants of :func:`compute_screw_axis`.

A non-finite or mis-shaped twist used to yield a ScrewAxis full of NaN with
``is_singular=False`` (or a numpy broadcasting error); it must be refused.
The geometric tests use an independent construction: a twist generated from a
chosen screw (axis, point, pitch) must be recovered exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.screw_theory.kinematics import (
    Twist,
    compute_screw_axis,
    compute_screw_endpoints,
)


def _twist(angular, linear, reference=(0.0, 0.0, 0.0)) -> Twist:
    return Twist(
        angular=np.asarray(angular, dtype=float),
        linear=np.asarray(linear, dtype=float),
        body_name="b",
        reference_point=np.asarray(reference, dtype=float),
    )


@pytest.mark.parametrize(
    ("angular", "linear", "reference"),
    [
        ([np.nan, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, np.inf], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, np.nan, 0.0]),
        ([0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_non_finite_or_misshaped_twist_is_refused(angular, linear, reference) -> None:
    with pytest.raises(ValueError):
        compute_screw_axis(_twist(angular, linear, reference))


@pytest.mark.parametrize("threshold", [0.0, -1e-6, float("nan"), float("inf")])
def test_invalid_singularity_threshold_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError):
        compute_screw_axis(_twist([0, 0, 1], [0, 0, 0]), threshold)


@pytest.mark.parametrize("length", [float("nan"), float("inf"), -0.5])
def test_endpoint_length_must_be_finite_non_negative(length: float) -> None:
    screw = compute_screw_axis(_twist([0, 0, 1], [0, 0, 0]))
    with pytest.raises(ValueError):
        compute_screw_endpoints(screw, length=length)


def test_accepts_plain_sequences_without_mutating_the_twist() -> None:
    twist = Twist([0, 0, 2], [1, 0, 4], "b", [0, 0, 0])  # type: ignore[arg-type]
    screw = compute_screw_axis(twist)
    assert screw.pitch == pytest.approx(2.0)
    assert twist.reference_point == [0, 0, 0]


def test_recovers_a_chosen_screw() -> None:
    """Twist built from (axis s, point q, pitch h, rate w) is inverted exactly.

    v at the origin is  w * (-s x q + h s)  (Murray-Li-Sastry, ch. 2).
    """
    s = np.array([1.0, 2.0, -2.0]) / 3.0
    q = np.array([0.3, -1.2, 0.7])
    h, w = 0.25, 1.7
    v_origin = w * (-np.cross(s, q) + h * s)
    reference = np.array([0.4, 0.1, -0.2])
    # velocity of the body point at ``reference``: v_origin + w x reference
    v_ref = v_origin + np.cross(w * s, reference)
    screw = compute_screw_axis(_twist(w * s, v_ref, reference))
    np.testing.assert_allclose(screw.axis_direction, s, atol=1e-12)
    assert screw.pitch == pytest.approx(h, abs=1e-12)
    # the returned point must lie on the true axis: (p - q) parallel to s
    offset = screw.axis_point - q
    np.testing.assert_allclose(np.cross(offset, s), 0.0, atol=1e-12)
    assert screw.angular_magnitude == pytest.approx(abs(w))
    assert math.isfinite(screw.linear_magnitude)
