"""NSX-40: the report schema and the six evidence axes (#12572)."""

from __future__ import annotations

import copy

import pytest

jsonschema = pytest.importorskip("jsonschema")
pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.facade import AnalysisOptions, analyze  # noqa: E402
from tests.unit.loop_nullspace.report_support import (  # noqa: E402
    load_schema,
    planar_engine,
    planar_trajectory,
)

pytestmark = pytest.mark.unit

AXES = (
    "allocation_dependence",
    "validation_status",
    "model_provenance",
    "input_provenance",
    "conventions",
    "hypotheses",
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return load_schema()


@pytest.fixture(scope="module")
def report() -> dict:
    return analyze(planar_trajectory(), planar_engine(), "envelope_only")


def _quantities(node):
    """Every dict that looks like a quantity (has ``units`` and ``values``)."""
    if isinstance(node, dict):
        if "units" in node and "values" in node:
            yield node
        for value in node.values():
            yield from _quantities(value)
    elif isinstance(node, list):
        for value in node:
            yield from _quantities(value)


def test_schema_is_a_valid_draft_2020_12_schema(schema) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_fixture_report_validates(schema, report) -> None:
    jsonschema.validate(report, schema, cls=jsonschema.Draft202012Validator)
    assert report["status"] == "complete"


def test_every_quantity_carries_all_six_evidence_axes(report) -> None:
    found = list(_quantities(report))
    assert len(found) > 10
    for q in found:
        for axis in AXES:
            assert axis in q, axis


@pytest.mark.parametrize("axis", AXES)
def test_a_quantity_missing_an_axis_is_rejected(schema, report, axis) -> None:
    bad = copy.deepcopy(report)
    next(_quantities(bad["invariants"])).pop(axis)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema, cls=jsonschema.Draft202012Validator)


def test_unavailable_quantity_must_state_a_reason(schema, report) -> None:
    bad = copy.deepcopy(report)
    q = next(_quantities(bad["invariants"]))
    q.update(allocation_dependence="UNAVAILABLE", values=None, reason=None)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema, cls=jsonschema.Draft202012Validator)


def test_invariant_quantity_cannot_carry_hypotheses(schema, report) -> None:
    bad = copy.deepcopy(report)
    q = next(
        q
        for q in _quantities(bad["invariants"])
        if q["allocation_dependence"] == "INVARIANT"
    )
    q["hypotheses"] = ["H-TORQUE-CAPACITY"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema, cls=jsonschema.Draft202012Validator)


def test_infeasible_frame_report_validates_and_names_its_conflict(schema) -> None:
    rep = analyze(planar_trajectory(), planar_engine(limit_nm=1e-3), "envelope_only")
    jsonschema.validate(rep, schema, cls=jsonschema.Draft202012Validator)
    frames = rep["envelopes"]["frames"]
    assert all(not f["feasible"] for f in frames)
    assert all(f["conflict_hypotheses"] == ["H-TORQUE-CAPACITY"] for f in frames)


def test_allocation_report_validates(schema) -> None:
    pytest.importorskip("clarabel")
    rep = analyze(
        planar_trajectory(),
        planar_engine(),
        "capacity_min_effort",
        AnalysisOptions(include_allocation=True),
    )
    jsonschema.validate(rep, schema, cls=jsonschema.Draft202012Validator)
    assert rep["allocation"]["status"] == "optimal"
    assert rep["allocation"]["torque"]["allocation_dependence"] == "FAMILY_DEPENDENT"


def test_schema_version_is_pinned_to_major_one(schema, report) -> None:
    bad = {**report, "schema_version": "2.0.0"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema, cls=jsonschema.Draft202012Validator)
