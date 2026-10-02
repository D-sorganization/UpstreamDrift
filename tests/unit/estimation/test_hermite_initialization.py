"""Authored initialization changes are explicit and independent of acceptance."""

from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from src.shared.python.estimation.hermite_bounds import HermiteBoundsDomain
from src.shared.python.estimation.hermite_initialization import (
    initialize_authored_hermite,
)
from src.shared.python.estimation.map_estimator import CubicHermiteSplineTrajectory


def test_projected_seed_is_feasible_and_reports_each_changed_knot():
    domain = HermiteBoundsDomain((10.0, 10.25, 12.0), ((-1.0, 1.0), None, (3.0, 3.0)))
    q = np.array([[-2.0, 9.0, 2.0], [0.2, -4.0, 3.0], [2.0, 7.0, 4.0]])
    v = np.array([[8.0, -1.0, 2.0], [0.0, 0.0, 0.0], [-7.0, 3.0, -1.0]])
    original = np.concatenate((q.ravel(), v.ravel()))
    saved = original.copy()
    result = initialize_authored_hermite(domain, original, ("angle", "root", "fixed"))
    np.testing.assert_array_equal(original, saved)
    initialized = np.asarray(result.coefficients)
    iq, iv = initialized.reshape(2, 3, 3)
    np.testing.assert_array_equal(iq[:, 0], [-1.0, 0.2, 1.0])
    np.testing.assert_array_equal(iq[:, 1], q[:, 1])
    np.testing.assert_array_equal(iq[:, 2], [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(iv, np.zeros_like(v))
    np.testing.assert_allclose(domain.decode(domain.encode(initialized)), initialized)
    assert len(result.changes) == 6
    for change in result.changes:
        column = ("angle", "root", "fixed").index(change.coordinate_name)
        knot = change.knot_index
        assert change.source_time == domain.knot_times[knot]
        assert change.original_position == q[knot, column]
        assert change.initialized_position == iq[knot, column]
        assert change.original_velocity == v[knot, column]
        assert change.initialized_velocity == 0.0
    assert [
        (d.coordinate_name, d.maximum_absolute_position_displacement)
        for d in result.maximum_position_displacements
    ] == [("angle", 1.0), ("root", 0.0), ("fixed", 1.0)]
    trajectory = CubicHermiteSplineTrajectory(np.asarray(domain.knot_times), 3)
    sampled = trajectory.evaluate(initialized, np.linspace(10, 12, 501)).q
    assert sampled[:, 0].min() >= -1.0 and sampled[:, 0].max() <= 1.0
    np.testing.assert_allclose(sampled[:, 2], 3.0, atol=1e-14)


def test_result_is_deeply_immutable_and_hashes_are_canonical():
    domain = HermiteBoundsDomain((0.0, 2.0), ((0.0, 1.0),))
    candidate = np.array([-1.0, 2.0, 1.0, -1.0])
    result = initialize_authored_hermite(domain, candidate, ("joint",))
    assert (
        result.original_coefficient_sha256
        == "sha256:" + hashlib.sha256(candidate.astype("<f8").tobytes()).hexdigest()
    )
    initialized = np.asarray(result.coefficients, dtype="<f8")
    assert (
        result.initialized_coefficient_sha256
        == "sha256:" + hashlib.sha256(initialized.tobytes()).hexdigest()
    )
    assert result.original_coefficient_sha256 != result.initialized_coefficient_sha256
    assert result == initialize_authored_hermite(
        domain, candidate.astype(">f8"), ("joint",)
    )
    candidate[:] = 0.0
    assert result.coefficients == (0.0, 1.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        result.changes[0].initialized_position = 7.0
    with pytest.raises(TypeError):
        result.coefficients[0] = 7.0


def test_feasible_zero_slope_candidate_has_no_changes_and_equal_hashes():
    domain = HermiteBoundsDomain((0.0, 1.0), (None, (4.0, 4.0)))
    candidate = np.array([1.0, 4.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    result = initialize_authored_hermite(domain, candidate, ("root", "fixed"))
    assert result.changes == ()
    assert result.original_coefficient_sha256 == result.initialized_coefficient_sha256
    assert all(
        d.maximum_absolute_position_displacement == 0.0
        for d in result.maximum_position_displacements
    )


def test_all_fixed_domain_has_no_decisions_after_explicit_initialization():
    domain = HermiteBoundsDomain((4.0, 4.3, 8.0), ((2.0, 2.0),))
    result = initialize_authored_hermite(
        domain, np.array([1.0, 2.0, 3.0, 7.0, -4.0, 6.0]), ("fixed",)
    )
    assert result.coefficients == (2.0, 2.0, 2.0, 0.0, 0.0, 0.0)
    assert domain.encode(np.asarray(result.coefficients)).size == 0
    assert domain.decode_jacobian(np.array([])).shape == (6, 0)
    assert len(result.changes) == 3


def test_unrepresentable_displacement_is_rejected_without_mutation():
    domain = HermiteBoundsDomain((0.0, 1.0), ((-1e308, -1e308),))
    original = np.array([1e308, 1e308, 0.0, 0.0])
    with pytest.raises(ValueError, match="displacement"):
        initialize_authored_hermite(domain, original, ("joint",))
    np.testing.assert_array_equal(original, [1e308, 1e308, 0.0, 0.0])


@pytest.mark.parametrize(
    "names",
    [("joint", "joint"), ("joint",), ("joint", ""), ("joint", 3), ["joint", "root"]],
)
def test_invalid_native_names_fail(names):
    domain = HermiteBoundsDomain((0.0, 1.0), (None, None))
    with pytest.raises(ValueError):
        initialize_authored_hermite(domain, np.zeros(8), names)


@pytest.mark.parametrize(
    "candidate",
    [
        np.zeros(3),
        np.zeros((2, 2)),
        [0.0, 0.0, np.nan, 0.0],
        [0.0, 0.0, np.inf, 0.0],
        [False] * 4,
    ],
)
def test_invalid_coefficient_vectors_fail(candidate):
    domain = HermiteBoundsDomain((0.0, 1.0), ((0.0, 1.0),))
    with pytest.raises(ValueError):
        initialize_authored_hermite(domain, np.asarray(candidate), ("joint",))
