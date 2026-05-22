"""Tests for ProvenanceRecord / ProvenanceValue (epic #5968, Phase 0.4)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.shared.python.ux.provenance import (
    ProvenanceError,
    ProvenanceRecord,
    ProvenanceValue,
)

pytestmark = pytest.mark.unit


def _record(**overrides) -> ProvenanceRecord:
    base = {
        "formula": "fps = 1.0 / timestep",
        "inputs": ("simulation.timestep",),
        "source": "mujoco:run-abc123",
        "computed_at": datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        "engine": "mujoco",
        "run_id": "run-abc123",
    }
    base.update(overrides)
    return ProvenanceRecord(**base)


def test_provenance_record_constructs_and_is_frozen():
    rec = _record()
    assert rec.formula == "fps = 1.0 / timestep"
    with pytest.raises((AttributeError, TypeError)):
        rec.formula = "mutated"  # type: ignore[misc]


def test_provenance_record_requires_non_empty_formula():
    with pytest.raises((ValueError, ProvenanceError)):
        _record(formula="")


def test_provenance_record_inputs_must_be_dotted_ids():
    with pytest.raises((ValueError, ProvenanceError)):
        _record(inputs=("bad id",))


def test_provenance_record_inputs_may_be_empty_for_constants():
    rec = _record(formula="constant 9.81", inputs=())
    assert rec.inputs == ()


def test_provenance_record_computed_at_must_be_timezone_aware():
    with pytest.raises((ValueError, ProvenanceError)):
        _record(computed_at=datetime(2025, 1, 1))  # naive


def test_provenance_record_to_dict_roundtrip():
    rec = _record()
    again = ProvenanceRecord.from_dict(rec.to_dict())
    assert again == rec


def test_provenance_value_carries_value_and_record():
    rec = _record()
    pv = ProvenanceValue(value=500.0, record=rec, display_units="Hz")
    assert pv.value == 500.0
    assert pv.record is rec
    assert pv.display_units == "Hz"


def test_provenance_value_rejects_non_record():
    with pytest.raises((TypeError, ProvenanceError)):
        ProvenanceValue(value=1.0, record="not a record", display_units="Hz")  # type: ignore[arg-type]


def test_provenance_value_describe_includes_formula_and_source():
    rec = _record()
    pv = ProvenanceValue(value=500.0, record=rec, display_units="Hz")
    description = pv.describe()
    assert "fps = 1.0 / timestep" in description
    assert "mujoco:run-abc123" in description
    assert "simulation.timestep" in description


def test_provenance_value_describe_handles_constant_with_no_inputs():
    rec = _record(formula="constant 9.81", inputs=())
    pv = ProvenanceValue(value=9.81, record=rec, display_units="m/s^2")
    description = pv.describe()
    assert "constant 9.81" in description
    assert "(no inputs)" in description


def test_provenance_error_subclasses_value_error():
    assert issubclass(ProvenanceError, ValueError)
