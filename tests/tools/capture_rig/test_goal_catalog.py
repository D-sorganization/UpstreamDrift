"""The executable capture graph is separate from informational data-flow arrows."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.capability_atlas.model import build, validate_graph
from src.tools.capture_rig.goal_catalog import load_catalog, workflow_evidence
from src.tools.capture_rig.goal_planner import Readiness, evaluate, resolve
from src.tools.capture_rig.session import SessionMedia, ViewMedia

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_default_goals_are_bound_to_existing_capabilities() -> None:
    catalog = load_catalog()
    assert catalog.model_dump(mode="json") == build(ROOT)["capture_goals"]
    for goal in catalog.goals:
        assert resolve(catalog, [goal.id]).steps
    edit = resolve(catalog, ["edit"])
    assert edit.minimum_views == 0
    assert not any(s.needs_calibration for s in edit.steps)
    projected = resolve(catalog, ["project_reference"])
    assert any(s.needs_calibration for s in projected.steps)
    assert "clubs.bag" not in edit.step_ids
    model = resolve(catalog, ["fit_model"])
    assert next(s for s in model.steps if s.id == "clubs.bag").optional


@pytest.mark.parametrize(
    "field,value",
    [
        ("action", "arbitrary.command"),
        ("node_id", "absent.capability"),
        ("workflow_key", "invented"),
    ],
)
def test_graph_rejects_unbound_navigation(field: str, value: str) -> None:
    graph = deepcopy(build(ROOT))
    graph["capture_goals"]["steps"][0][field] = value
    with pytest.raises(ValueError):
        validate_graph(graph, ROOT)


def test_architecture_edges_do_not_change_executable_routes() -> None:
    graph = build(ROOT)
    graph["edges"] = []
    validate_graph(graph, ROOT)
    assert graph["capture_goals"] == load_catalog().model_dump(mode="json")


def test_existing_workflow_rules_and_explicit_staleness_feed_the_route(
    tmp_path,
) -> None:
    media = SessionMedia(
        root=tmp_path,
        plan_name="Imported Swing",
        views=(
            ViewMedia(
                "front",
                "front",
                tmp_path / "swing.mp4",
                None,
                tmp_path / "joints.json",
                30,
            ),
        ),
        swing_summary=None,
        reconstruction=None,
        problems=(),
    )
    route = resolve(load_catalog(), ["analyze_2d"])
    reviewed = {
        "capture.library": Readiness("done"),
        "capture.selection": Readiness("done"),
    }
    evidence = workflow_evidence(route, media, reviewed)
    assert evidence["step.detect"].status == "done"
    assert evidence["step.analyze_2d"].status == "ready"
    stale = workflow_evidence(
        route, media, reviewed, invalidated={"detect": "Swing edits changed"}
    )
    states = evaluate(route, stale, view_count=1, calibration_compatible=False)
    assert states[-1].status == "blocked"
    assert stale["step.detect"].reason == "Swing edits changed"


def test_custom_evidence_cannot_override_workflow_readiness() -> None:
    route = resolve(load_catalog(), ["analyze_2d"])
    states = workflow_evidence(route, None, {"step.detect": Readiness("done")})
    assert states["step.detect"].status == "blocked"
