"""Lossless source-clock restriction preserves Hermite segment semantics."""

from dataclasses import FrozenInstanceError
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    HermiteBoundsDomain,
    SplineTrajectoryEvaluation,
)
from src.shared.python.motion_matching.historical_fit import ImageSplineStart
from src.shared.python.motion_matching.historical_fit.spline_restriction import (
    SplineIntervalRestriction,
    restrict_image_spline_interval,
)

pytestmark = pytest.mark.unit


def example() -> ImageSplineStart:
    knots = np.array([2.0, 2.4, 3.2, 5.0])
    q = np.array([[0.1, 0.4], [0.3, 0.1], [0.7, 0.3], [0.2, 0.5]])
    v = np.array([[0.2, -0.2], [0.1, 0.15], [-0.1, 0.1], [0.05, 0.0]])
    trajectory = CubicHermiteSplineTrajectory(knots, 2)
    return ImageSplineStart.from_coefficients(
        knots,
        trajectory.pack(q, v),
        ("free_a", "locked", "free_b"),
        ("free_a", "free_b"),
        "native-synthetic",
    )


def evaluate(start: ImageSplineStart, times: np.ndarray) -> SplineTrajectoryEvaluation:
    return CubicHermiteSplineTrajectory(np.array(start.knot_times), 2).evaluate(
        np.array(start.spline_coefficients),
        times,
    )


def test_inside_segment_and_nonuniform_interior_qva_preservation() -> None:
    original = example()
    snapshot = original.to_record()
    receipt = restrict_image_spline_interval(original, 2.15, 4.4)
    new = receipt.restricted_start
    assert receipt.original_start is original
    assert new.knot_times == (2.15, 2.4, 3.2, 4.4)
    assert new.model_sha == original.model_sha
    assert new.coordinate_order == original.coordinate_order
    assert new.free_coordinates == original.free_coordinates
    assert new.coefficient_sha256 != original.coefficient_sha256
    times = np.unique(np.r_[np.linspace(2.15, 4.4, 101), new.knot_times])
    old, cut = evaluate(original, times), evaluate(new, times)
    for field in ("q", "v", "a"):
        np.testing.assert_allclose(
            getattr(old, field), getattr(cut, field), atol=2e-13, rtol=0
        )
    assert original.to_record() == snapshot
    # Locked values are caller-owned; reconstruct the same full pose, never infer a new lock.
    old_full = np.column_stack((old.q[:, 0], np.full(len(times), 0.37), old.q[:, 1]))
    new_full = np.column_stack((cut.q[:, 0], np.full(len(times), 0.37), cut.q[:, 1]))
    np.testing.assert_allclose(old_full, new_full, atol=2e-13, rtol=0)


def test_full_interval_is_exact_and_deterministic() -> None:
    original = example()
    receipt = restrict_image_spline_interval(original, 2.0, 5.0)
    assert receipt.restricted_start is original
    assert receipt == restrict_image_spline_interval(original, 2.0, 5.0)
    assert (
        SplineIntervalRestriction.from_record(
            json.loads(json.dumps(receipt.to_record()))
        )
        == receipt
    )
    record = receipt.to_record()
    record["original_start"]["knot_times"][0] = -3
    assert original.knot_times[0] == 2
    with pytest.raises(FrozenInstanceError):
        receipt.restricted_start = original


def test_upper_existing_knot_uses_retained_left_acceleration() -> None:
    original = example()
    new = restrict_image_spline_interval(original, 2.4, 3.2).restricted_start
    at = np.array([2.4, 2.8, 3.2])
    old, cut = evaluate(original, at), evaluate(new, at)
    np.testing.assert_allclose(old.q, cut.q, atol=1e-13, rtol=0)
    np.testing.assert_allclose(old.v, cut.v, atol=1e-13, rtol=0)
    np.testing.assert_allclose(old.a[:-1], cut.a[:-1], atol=1e-13, rtol=0)
    assert not np.allclose(old.a[-1], cut.a[-1])
    # Independent left-segment endpoint second derivative, not parent's right-side convention.
    q, v = CubicHermiteSplineTrajectory(np.array(original.knot_times), 2).unpack(
        np.array(original.spline_coefficients)
    )
    h = 3.2 - 2.4
    expected = 6 * (q[1] - q[2]) / h**2 + (2 * v[1] + 4 * v[2]) / h
    np.testing.assert_allclose(cut.a[-1], expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (True, 4.0),
        (2.0, False),
        ("2", 4.0),
        (2.0, "4"),
        (float("nan"), 4.0),
        (2.0, float("inf")),
        (1.9, 4.0),
        (2.0, 5.1),
        (3.0, 3.0),
        (4.0, 3.0),
    ],
)
def test_invalid_interval_rejected(lower: Any, upper: Any) -> None:
    with pytest.raises(ValueError):
        restrict_image_spline_interval(example(), lower, upper)


def test_untyped_start_rejected() -> None:
    with pytest.raises(ValueError, match="typed"):
        restrict_image_spline_interval(example().to_record(), 2.0, 4.0)


@pytest.mark.parametrize("change", ["coefficient", "model", "order", "free", "knots"])
def test_forged_restriction_receipt_rejected(change: str) -> None:
    original = example()
    receipt = restrict_image_spline_interval(original, 2.15, 4.4)
    new = receipt.restricted_start
    data = new.to_record()
    if change == "coefficient":
        data["spline_coefficients"][0] += 0.01
    elif change == "model":
        data["model_sha"] = "other-model"
    elif change == "order":
        data["coordinate_order"] = ["free_b", "locked", "free_a"]
    elif change == "free":
        data["free_coordinates"] = ["free_b", "free_a"]
    else:
        data["knot_times"][1] += 0.01
    forged = ImageSplineStart.from_coefficients(
        data["knot_times"],
        data["spline_coefficients"],
        tuple(data["coordinate_order"]),
        tuple(data["free_coordinates"]),
        data["model_sha"],
    )
    with pytest.raises(ValueError):
        SplineIntervalRestriction(original, forged)


def test_strict_record_fields_and_typed_receipt() -> None:
    original = example()
    with pytest.raises(ValueError):
        SplineIntervalRestriction(None, original)
    with pytest.raises(ValueError):
        SplineIntervalRestriction.from_record({"original_start": original})
    record = restrict_image_spline_interval(original, 2.1, 4.0).to_record()
    record["unexpected"] = True
    with pytest.raises(ValueError):
        SplineIntervalRestriction.from_record(record)


def test_restriction_keeps_bernstein_bounds_without_zeroing() -> None:
    original = example()
    new = restrict_image_spline_interval(original, 2.1, 4.7).restricted_start
    for start in (original, new):
        domain = HermiteBoundsDomain(
            tuple(start.knot_times), ((-1.0, 1.0), (-1.0, 1.0))
        )
        encoded = domain.encode(np.array(start.spline_coefficients))
        np.testing.assert_allclose(
            domain.decode(encoded), start.spline_coefficients, atol=2e-15, rtol=0
        )
    _, v = CubicHermiteSplineTrajectory(np.array(new.knot_times), 2).unpack(
        np.array(new.spline_coefficients)
    )
    assert np.count_nonzero(v) > 0


def test_single_segment_cut_matches_independent_cubic_derivatives() -> None:
    knots = np.array([0.0, 0.4, 1.0, 1.8])
    q = np.column_stack((knots**3, 2 * knots**2 - knots))
    v = np.column_stack((3 * knots**2, 4 * knots - 1))
    parent = ImageSplineStart.from_coefficients(
        knots,
        CubicHermiteSplineTrajectory(knots, 2).pack(q, v),
        ("a", "b"),
        ("a", "b"),
        "synthetic-cubic",
    )
    child = restrict_image_spline_interval(parent, 0.55, 0.95).restricted_start
    assert child.knot_times == (0.55, 0.95)
    times = np.linspace(0.55, 0.95, 17)
    actual = evaluate(child, times)
    expected = (
        np.column_stack((times**3, 2 * times**2 - times)),
        np.column_stack((3 * times**2, 4 * times - 1)),
        np.column_stack((6 * times, np.full(len(times), 4.0))),
    )
    for field, values in zip(("q", "v", "a"), expected, strict=True):
        np.testing.assert_allclose(getattr(actual, field), values, atol=2e-13, rtol=0)


def test_curated_facade_is_sdk_free_in_cold_process() -> None:
    root = Path(__file__).resolve().parents[3]
    code = """
import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'mujoco','pinocchio','pydrake','opensim','PyQt6'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from src.shared.python.motion_matching.historical_fit import SplineIntervalRestriction, restrict_image_spline_interval
assert callable(restrict_image_spline_interval)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
