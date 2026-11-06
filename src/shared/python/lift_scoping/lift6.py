"""LIFT-6 (#11746) — OpenSim full-body musculoskeletal lift pipeline contracts.

Imports `opensim_full_body_lifts` **inside** functions so headless environments
without the full dependency graph can still load the JSON-only parts of this
package when tests split.

Epic acceptance still needs end-to-end receipts, reserve qualification and
owner video evidence; see :data:`ISSUE_TODOS_GATE_ON_HOST`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from ._json_validate import validate_payload

ISSUE_ID: Final = 11746

FIXTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "issue", "composition", "kinetics"],
    "properties": {
        "schema_version": {"const": "ud.lift_scoping.lift6.v1"},
        "issue": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"const": ISSUE_ID}},
        },
        "composition": {
            "type": "object",
            "required": ["expected_muscle_count"],
            "properties": {
                "expected_muscle_count": {
                    "type": "integer",
                    "minimum": 1,
                },
            },
        },
        "kinetics": {
            "type": "object",
            "required": ["expect_qualified_when_opensim_and_trajectory"],
            "properties": {
                "expect_qualified_when_opensim_and_trajectory": {"type": "boolean"},
            },
        },
    },
}

ISSUE_TODOS_GATE_ON_HOST: Final[tuple[str, ...]] = (
    "Verify muscle-force qualification (reserve share, CMC/SO choice) meets "
    "acceptance thresholds on a host with native OpenSim and opensim-models.",
    "Record joint moments, hand wrench (GCV-7), GRF (GCV-1) and L5/S1 load "
    "receipts per lift with provenance; owner clips stay off-repo per epic rules.",
)


def validate_lift6_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_payload(record, FIXTURE_SCHEMA)
    return dict(record)


def issue_todo_block() -> str:
    lines = [f"LIFT-6 (#{ISSUE_ID}) — OpenSim host and evidence gates:", ""]
    lines.extend(f"- TODO: {t}" for t in ISSUE_TODOS_GATE_ON_HOST)
    return "\n".join(lines)


def assert_opensim_lift_composition_counts(
    *,
    compose: Callable[..., Any] | None = None,
    fixture: Mapping[str, Any],
) -> None:
    """When OpenSim is available, composed squat model carries expected muscle count."""
    validate_payload(fixture, FIXTURE_SCHEMA)
    from src.shared.python.engine_core.engine_availability import is_engine_available
    from src.shared.python.lifting.opensim_full_body_lifts import (
        OpenSimFullBodyLiftSpec,
        compose_opensim_full_body_lift,
    )

    if not is_engine_available("opensim"):
        return
    fn = compose_opensim_full_body_lift if compose is None else compose
    comp = fn(OpenSimFullBodyLiftSpec("squat"))
    if comp.status != "composed":
        return
    mc = comp.muscle_count
    assert mc is not None
    assert int(mc) == int(fixture["composition"]["expected_muscle_count"])


def assert_evaluate_kinetics_contract(
    *,
    evaluate: Callable[..., Any] | None = None,
    fixture: Mapping[str, Any],
) -> None:
    """`evaluate_opensim_lift_kinetics` never returns qualified without a trajectory."""
    validate_payload(fixture, FIXTURE_SCHEMA)
    from src.shared.python.engine_core.engine_availability import is_engine_available
    from src.shared.python.lifting.opensim_full_body_lifts import (
        OpenSimFullBodyLiftSpec,
        evaluate_opensim_lift_kinetics,
    )

    if not is_engine_available("opensim"):
        return
    fn = evaluate_opensim_lift_kinetics if evaluate is None else evaluate
    spec = OpenSimFullBodyLiftSpec("squat")
    res = fn(spec, None)
    assert not res.is_qualified
    if fixture["kinetics"]["expect_qualified_when_opensim_and_trajectory"]:
        assert res.status == "unqualified"
