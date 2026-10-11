"""Guard the delivery graph against protocol/scientific-evidence deadlocks."""

from __future__ import annotations

from graphlib import TopologicalSorter
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_feedback_milestones_are_acyclic_and_resolve_every_prerequisite() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/feedback_controls/milestone_dependencies.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document["milestones"]
    graph = {row["id"]: set(row["depends_on"]) for row in rows}
    assert len(graph) == len(rows), "milestone identifiers must be unique"
    assert all(dependency in graph for deps in graph.values() for dependency in deps)
    assert set(TopologicalSorter(graph).static_order()) == set(graph)
    assert "T02_contract" in graph["D02"]
    assert "F07_ready" in graph["F08_ready"]
    assert "F07_accepted" not in graph["F08_ready"]
    assert "D03" in graph["F07_accepted"]
    assert "F10" not in graph["D04"]
    assert "D04" in graph["F10"]
