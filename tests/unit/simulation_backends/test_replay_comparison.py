"""Per-quantity replay acceptance contract for #15493."""

import numpy as np
import pytest

from src.shared.python.simulation_backends.validation import (
    ReplayTolerance,
    compare_replay_quantities,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe, pytest.mark.contract]


def compare(recorded, replayed, tolerances):
    return compare_replay_quantities(
        recorded,
        replayed,
        tolerances,
        recorded_runtime="pinned-runtime-sha256",
        replay_runtime="pinned-runtime-sha256",
    )


def test_quantity_specific_budgets_and_deterministic_reports():
    reports = compare(
        {"torque_nm": [100.0], "q_rad": [0.0]},
        {"q_rad": [0.002], "torque_nm": [100.5]},
        {
            "torque_nm": ReplayTolerance(atol=0.0, rtol=0.01),
            "q_rad": ReplayTolerance(atol=0.001, rtol=0.0),
        },
    )
    assert list(reports) == ["q_rad", "torque_nm"]
    assert not reports["q_rad"].passed
    assert reports["q_rad"].max_abs_error == pytest.approx(0.002)
    assert reports["q_rad"].atol == 0.001
    assert reports["torque_nm"].passed


def test_relative_budget_is_scaled_by_recorded_value():
    report = compare(
        {"q": [1.0]}, {"q": [2.0]}, {"q": ReplayTolerance(atol=0.0, rtol=0.5)}
    )["q"]
    assert not report.passed  # Scaling by the replay would incorrectly pass.


def test_exact_boundary_scalars_and_zero_tolerance():
    assert compare(
        {"impact_s": 1.0},
        {"impact_s": 1.125},
        {"impact_s": ReplayTolerance(atol=0.125, rtol=0)},
    )["impact_s"].passed
    assert compare({"q": [0.0]}, {"q": [0.0]}, {"q": ReplayTolerance(atol=0, rtol=0)})[
        "q"
    ].passed
    assert not compare(
        {"q": [0.0]}, {"q": [1e-15]}, {"q": ReplayTolerance(atol=0, rtol=0)}
    )["q"].passed


@pytest.mark.parametrize(
    "atol,rtol", [(-1, 0), (0, -1), (np.nan, 0), (0, np.inf), (np.inf, 0)]
)
def test_invalid_tolerances_are_rejected(atol, rtol):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ReplayTolerance(atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "recorded,replayed,budgets",
    [
        ({}, {}, {}),
        ({"q": [1]}, {}, {"q": ReplayTolerance(0, 0)}),
        ({"q": [1]}, {"q": [1], "v": [1]}, {"q": ReplayTolerance(0, 0)}),
        ({"q": [1]}, {"q": [1]}, {}),
        (
            {"q": [1]},
            {"q": [1]},
            {"q": ReplayTolerance(0, 0), "v": ReplayTolerance(0, 0)},
        ),
    ],
)
def test_quantity_sets_must_match_exactly(recorded, replayed, budgets):
    with pytest.raises(ValueError, match="quantities"):
        compare(recorded, replayed, budgets)


@pytest.mark.parametrize("value", [[], [np.nan], [np.inf], [1 + 2j], [True], ["1"]])
def test_invalid_arrays_are_rejected_on_either_side(value):
    for recorded, replayed in [
        ({"q": value}, {"q": [1.0]}),
        ({"q": [1.0]}, {"q": value}),
    ]:
        with pytest.raises(ValueError, match="q.*nonempty finite real"):
            compare(recorded, replayed, {"q": ReplayTolerance(0, 0)})


def test_shape_mismatch_is_not_broadcast():
    with pytest.raises(ValueError, match="shapes disagree"):
        compare({"q": [[1.0], [2.0]]}, {"q": [1.0, 2.0]}, {"q": ReplayTolerance(0, 0)})


@pytest.mark.parametrize("runtime", ["different", "", " ", None])
def test_runtime_must_match_a_declared_pin(runtime):
    with pytest.raises(ValueError, match="runtime"):
        compare_replay_quantities(
            {"q": [1]},
            {"q": [1]},
            {"q": ReplayTolerance(0, 0)},
            recorded_runtime="pinned",
            replay_runtime=runtime,
        )


def test_inputs_are_not_mutated():
    recorded, replayed = np.array([1.0, 2.0]), np.array([1.0, 2.1])
    compare({"q": recorded}, {"q": replayed}, {"q": ReplayTolerance(0.01, 0)})
    np.testing.assert_array_equal(recorded, [1.0, 2.0])
    np.testing.assert_array_equal(replayed, [1.0, 2.1])
