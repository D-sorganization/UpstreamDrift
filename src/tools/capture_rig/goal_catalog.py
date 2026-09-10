"""Load the shared capability authority and adapt existing workflow evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from dataclasses import dataclass

from .goal_planner import CaptureGoalCatalog, CaptureRoute, Readiness, State

if TYPE_CHECKING:
    from .session import SessionMedia

CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config/capability_connections.json"
)
MAX_CATALOG_BYTES = 2 * 1024 * 1024
# Navigation identifiers, never shell commands or arbitrary URLs. Qt adapters
# must implement this complete set before exposing a route to the operator.
NAVIGATION_ACTIONS = frozenset(
    {
        "library",
        "edit",
        "draw",
        "references",
        "compare_reference",
        "my_clubs",
        "calibration",
        "workflow",
    }
)


def validate_bindings(
    catalog: CaptureGoalCatalog,
    node_ids: Iterable[str],
    workflow_keys: Iterable[str],
) -> None:
    """Reject navigation targets missing from the existing capability authorities."""
    nodes, keys = set(node_ids), set(workflow_keys)
    for step in catalog.steps:
        if (step.node_id or step.id) not in nodes:
            raise ValueError(f"Unknown capability for capture step: {step.id}")
        if step.action not in NAVIGATION_ACTIONS:
            raise ValueError(f"Unsupported capture navigation: {step.action}")
        if step.workflow_key is not None and step.workflow_key not in keys:
            raise ValueError(f"Unknown workflow key: {step.workflow_key}")
        if step.action == "workflow" and step.workflow_key is None:
            raise ValueError(f"Workflow navigation needs a step key: {step.id}")


def _connections(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        content = stream.read(MAX_CATALOG_BYTES + 1)
    if len(content) > MAX_CATALOG_BYTES:
        raise ValueError("Capture capability catalog is too large")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Capability connections must be an object")
    return data


def load_catalog(path: Path = CATALOG_PATH) -> CaptureGoalCatalog:
    """Read bounded, validated outcome metadata from the architecture source."""
    from .workflow import STEPS

    connections = _connections(path)
    catalog = CaptureGoalCatalog.model_validate(connections["capture_goals"])
    keys = [step.key for step in STEPS]
    nodes = [node["id"] for node in connections["nodes"]]
    validate_bindings(catalog, nodes + [f"step.{key}" for key in keys], keys)
    return catalog


@dataclass(frozen=True)
class StepDescription:
    title: str
    purpose: str
    instructions: tuple[str, ...] = ()


def step_descriptions() -> dict[str, StepDescription]:
    """Labels and detailed help stay with the existing workflow and capability map."""
    from .workflow import STEPS

    descriptions = {
        node["id"]: StepDescription(
            node["title"], node["description"], tuple(node.get("instructions", ()))
        )
        for node in _connections(CATALOG_PATH)["nodes"]
    }
    descriptions.update(
        {
            f"step.{step.key}": StepDescription(
                step.title, step.purpose, step.requirements + step.instructions
            )
            for step in STEPS
        }
    )
    return descriptions


def workflow_evidence(
    route: CaptureRoute,
    media: SessionMedia | None,
    navigation_evidence: Mapping[str, Readiness],
    *,
    invalidated: Mapping[str, str] | None = None,
) -> dict[str, Readiness]:
    """Delegate readiness to workflow.evaluate; explicit stale evidence only blocks.

    Non-workflow steps require evidence from their existing editor/library
    adapters. Merely visiting a page does not mark its artifact as complete.
    """
    from .workflow import evaluate

    states = {state.step.key: state for state in evaluate(media)}
    invalidated = invalidated or {}
    if any(key not in states for key in invalidated):
        raise ValueError("Cannot invalidate an unknown workflow step")
    evidence = {}
    for step in route.steps:
        key = step.workflow_key
        if key is None:
            evidence[step.id] = navigation_evidence.get(
                step.id, Readiness("blocked", "Open this step to review its inputs")
            )
        elif key in invalidated:
            evidence[step.id] = Readiness("blocked", invalidated[key])
        else:
            state = states[key]
            evidence[step.id] = Readiness(cast(State, state.status.value), state.reason)
    return evidence
