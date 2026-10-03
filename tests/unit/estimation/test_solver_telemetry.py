"""Measured counts and durations cannot be substituted with requested budgets."""

import pytest

from src.shared.python.estimation import SolverBackend, SolverTelemetry


@pytest.mark.parametrize("count", [True, 1.0, "1", -1])
def test_counts_require_nonnegative_native_integers(count):
    with pytest.raises(ValueError, match="nfev"):
        SolverTelemetry(nfev=count)


@pytest.mark.parametrize("duration", [True, "1", -0.1, float("nan"), float("inf")])
def test_duration_is_measured_finite_nonnegative_seconds(duration):
    with pytest.raises(ValueError, match="solver_elapsed_s"):
        SolverTelemetry(solver_elapsed_s=duration)


def test_typed_roundtrip_and_legacy_unavailability():
    record = SolverTelemetry(
        3,
        None,
        0.125,
        2.5,
        "gtol reached",
        SolverBackend("scipy.optimize.least_squares", "trf", "1.15"),
    )
    assert SolverTelemetry.from_record(record.to_record()) == record
    legacy = SolverTelemetry.from_record(None)
    assert legacy.nfev is None and legacy.worker_elapsed_s is None
    assert legacy.unavailable_reason == "legacy_record_has_no_telemetry"
    with pytest.raises(ValueError, match="Unknown"):
        SolverTelemetry.from_record({"requested_budget": 120})


def test_backend_and_nested_constructor_are_strict():
    with pytest.raises(ValueError, match="backend"):
        SolverTelemetry(backend={"name": "scipy"})
    with pytest.raises(ValueError, match="name"):
        SolverBackend("")


@pytest.mark.parametrize(
    "field",
    [
        "nfev",
        "njev",
        "solver_elapsed_s",
        "worker_elapsed_s",
        "termination_reason",
        "backend",
        "unavailable_reason",
    ],
)
def test_new_schema_requires_every_measurement_key_even_when_null(field):
    record = SolverTelemetry().to_record()
    del record[field]
    with pytest.raises(ValueError, match="fields"):
        SolverTelemetry.from_record(record)
