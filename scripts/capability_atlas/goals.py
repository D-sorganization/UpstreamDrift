"""Outcome routes rendered from the same validated planner used by Capture Rig."""

from __future__ import annotations

import html

from src.tools.capture_rig.goal_planner import CaptureGoalCatalog, resolve

from .model import Graph


def routes(graph: Graph) -> str:
    catalog = CaptureGoalCatalog.model_validate(graph["capture_goals"])
    titles = {node["id"]: node["title"] for node in graph["nodes"]}
    cards = []
    for goal in catalog.goals:
        route = resolve(catalog, [goal.id])
        steps = "".join(
            f"<li>{html.escape(titles[step.node_id or step.id])}"
            f"{' (Optional)' if step.optional else ''}</li>"
            for step in route.steps
        )
        cameras = (
            f"At least {route.minimum_views} camera views."
            if route.minimum_views > 1
            else "Single-view analysis only."
            if route.maximum_views == 1
            else "Use an existing video or record a new capture."
        )
        calibration = (
            " Compatible calibration and optical settings are required."
            if any(step.needs_calibration for step in route.steps)
            else ""
        )
        cards.append(
            f'<article id="goal-{goal.id}"><h3>{html.escape(goal.title)}</h3>'
            f"<p>{cameras}{calibration}</p><ol>{steps}</ol>"
            f'<label><input type="checkbox" name="capture-goal" value="{goal.id}" '
            f'style="width:auto"> Select {html.escape(goal.title)}</label></article>'
        )
    return (
        f'<div id="capture-goals" data-revision="{catalog.revision}">'
        "<p>Select outcomes and save a plan for the desktop capture wizard. "
        "The wizard checks current inputs when the plan is opened. "
        "Some single-view and multi-view outcomes need separate sessions.</p>"
        '<div class="cards">' + "\n".join(cards) + "</div>"
        '<p><button type="button" id="save-capture-plan">Save Selected Capture Plan</button></p>'
        '<p id="capture-plan-status" role="status" aria-live="polite"></p></div>'
    )


def prerequisite_mermaid(graph: Graph) -> str:
    """An executable prerequisite DAG, distinct from architecture/data-flow edges."""
    catalog = CaptureGoalCatalog.model_validate(graph["capture_goals"])
    titles = {node["id"]: node["title"] for node in graph["nodes"]}
    ids = {step.id: f"s{i}" for i, step in enumerate(catalog.steps)}
    lines = [
        "flowchart LR",
        "  accTitle: Capture Goal Prerequisites",
        "  accDescr: Required inputs for guided capture outcomes. Optional steps may be skipped.",
    ]
    for step in catalog.steps:
        title = html.escape(titles[step.node_id or step.id]).replace('"', "#quot;")
        suffix = " (Optional)" if step.optional else ""
        lines.append(f'  {ids[step.id]}["{title}{suffix}"]')
        for dependency in step.requires:
            lines.append(f"  {ids[dependency]} --> {ids[step.id]}")
    for index, goal in enumerate(catalog.goals):
        title = html.escape(goal.title).replace('"', "#quot;")
        lines.append(f'  g{index}(["{title}"])')
        for key in goal.steps:
            lines.append(f"  {ids[key]} --> g{index}")
    return "\n".join(lines) + "\n"
