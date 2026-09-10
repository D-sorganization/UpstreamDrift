"""Outcome routes share prerequisites and refuse stale or unsupported resumes."""

import pytest

from src.tools.capture_rig.goal_planner import (
    CaptureGoalCatalog,
    CaptureProgress,
    Readiness,
    resolve,
    evaluate,
    restore,
)

pytestmark = pytest.mark.unit


def catalog() -> CaptureGoalCatalog:
    return CaptureGoalCatalog.model_validate(
        {
            "steps": [
                {"id": "capture.library", "action": "library"},
                {
                    "id": "capture.selection",
                    "action": "edit",
                    "requires": ["capture.library"],
                },
                {
                    "id": "capture.calibration",
                    "action": "calibration",
                    "requires": ["capture.library"],
                },
                {
                    "id": "step.detect",
                    "action": "ingest",
                    "requires": ["capture.selection"],
                    "workflow_key": "detect",
                },
                {
                    "id": "step.reconstruct",
                    "action": "reconstruct",
                    "requires": ["capture.calibration", "step.detect"],
                    "minimum_views": 2,
                    "needs_calibration": True,
                    "workflow_key": "reconstruct",
                },
            ],
            "goals": [
                {"id": "edit", "title": "Edit a Swing", "steps": ["capture.selection"]},
                {
                    "id": "reconstruct",
                    "title": "Reconstruct in 3-D",
                    "steps": ["step.reconstruct"],
                },
            ],
        }
    )


def test_editing_only_does_not_require_calibration_or_multiple_cameras() -> None:
    route = resolve(catalog(), ["edit"])
    assert route.step_ids == ("capture.library", "capture.selection")
    assert route.minimum_views == 0
    assert not any(step.needs_calibration for step in route.steps)


def test_multiple_goals_share_prerequisites_in_deterministic_order() -> None:
    first = resolve(catalog(), ["reconstruct", "edit", "edit"])
    assert first == resolve(catalog(), ["edit", "reconstruct"])
    assert first.step_ids.count("capture.library") == 1
    assert first.step_ids.index("step.detect") < first.step_ids.index(
        "step.reconstruct"
    )
    assert first.minimum_views == 2


@pytest.mark.parametrize("mutation", ["cycle", "dangling", "duplicate"])
def test_invalid_dependency_graphs_are_rejected(mutation: str) -> None:
    data = catalog().model_dump()
    steps = list(data["steps"])
    if mutation == "duplicate":
        steps.append(steps[0])
    else:
        steps[0]["requires"] = (
            "step.reconstruct" if mutation == "cycle" else "absent",
        )
    data["steps"] = steps
    with pytest.raises(ValueError):
        CaptureGoalCatalog.model_validate(data)


def test_unknown_outcomes_and_incompatible_routes_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        resolve(catalog(), ["arbitrary-command"])
    with pytest.raises(ValueError, match="Select"):
        resolve(catalog(), [])
    data = catalog().model_dump()
    steps = list(data["steps"])
    steps[0]["maximum_views"] = 1
    data["steps"] = steps
    with pytest.raises(ValueError, match="incompatible"):
        resolve(CaptureGoalCatalog.model_validate(data), ["reconstruct"])


def test_camera_profile_and_stale_evidence_block_downstream_completion() -> None:
    route = resolve(catalog(), ["reconstruct"])
    evidence = {key: Readiness("done") for key in route.step_ids}
    states = evaluate(route, evidence, view_count=2, calibration_compatible=False)
    assert states[-1].status == "blocked"
    assert "calibration" in states[-1].reason.lower()
    evidence["step.detect"] = Readiness("blocked", "Swing edits changed; detect again")
    states = evaluate(route, evidence, view_count=2, calibration_compatible=True)
    assert states[-1].status == "blocked"
    assert "step.detect" in states[-1].reason


def test_resume_binds_capture_graph_and_input_revision() -> None:
    graph = catalog()
    route = resolve(graph, ["edit"])
    progress = CaptureProgress(
        capture_id="swing-1",
        goals=route.goals,
        catalog_revision=graph.revision,
        input_revision="a" * 64,
        current_step="capture.selection",
    )
    saved = CaptureProgress.model_validate_json(progress.model_dump_json())
    assert restore(graph, saved, capture_id="swing-1", input_revision="a" * 64) == route
    for capture_id, revision in [("swing-2", "a" * 64), ("swing-1", "b" * 64)]:
        with pytest.raises(ValueError, match="changed|another capture"):
            restore(graph, saved, capture_id=capture_id, input_revision=revision)
    changed = graph.model_dump()
    goals = list(changed["goals"])
    goals[0]["title"] = "Updated Editing Route"
    changed["goals"] = goals
    with pytest.raises(ValueError, match="graph changed"):
        restore(
            CaptureGoalCatalog.model_validate(changed),
            saved,
            capture_id="swing-1",
            input_revision="a" * 64,
        )
