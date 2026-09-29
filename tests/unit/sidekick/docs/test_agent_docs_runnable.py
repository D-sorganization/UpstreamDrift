"""Doctest-style checks for the sidekick.agent worked examples
(epic #5967 / S9 / #5978).

The worked examples in ``docs/sidekick/agent.md`` describe how to
add an action, register a host port, run a workflow, and consume the
chat-surface envelope. These tests assert that the public symbols those
examples use exist and behave as documented. Drift between the docs and
the code surface is loud here, not silent.

The doc samples are exercise-quality: we don't try to run every line
verbatim (the "real port" example references widget code that hasn't
landed yet). Instead, we run minimal end-to-end constructions that touch
every named symbol so a rename or signature change here would surface
immediately.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_doc_example_imports_resolve() -> None:
    """Every symbol named in docs/sidekick/agent.md is importable."""
    from sidekick.agent import (  # noqa: F401
        ActionChipModel,
        ActionChipState,
        ActionDescriptor,
        ActionResult,
        ChatActionEnvelope,
        HostActionPort,
        HostAdapter,
        HostCapability,
        HostInvocationResult,
        JsonlActionAudit,
        MemoryActionAudit,
        PendingUserDecision,
        PlannedStep,
        PolicyDecision,
        SidekickActionHandler,
        SidekickActionPolicy,
        SidekickActionService,
        SidekickAgentPlanner,
        SidekickWorkflow,
        SubtabActionPort,
        SubtabAdapter,
        ToolCall,
        WorkflowOutcome,
        WorkflowStep,
        WorkflowStepResult,
        WorkflowStepStatus,
        action_step,
        build_chip,
        build_feature_catalog,
        build_sidekick_system_prompt,
        lookup_feature,
        redact_secrets,
        run_sidekick_workflow,
        search_features,
        serialize_envelope,
    )


def test_doc_example_audit_wiring_works(tmp_path) -> None:
    """The "Audit log" docs section shows wiring JsonlActionAudit +
    SidekickActionPolicy into the service. Construct exactly that."""
    from sidekick.agent import (
        JsonlActionAudit,
        SidekickActionPolicy,
        SidekickActionService,
    )

    audit_path = tmp_path / "actions.jsonl"
    service = SidekickActionService(
        audit_sink=JsonlActionAudit(path=audit_path),
        policy=SidekickActionPolicy(
            allow_write=frozenset({"subtab.workspace.set_variable"}),
            allow_destructive=frozenset({"subtab.workspace.clear"}),
        ),
    )
    # Service constructed without raising means the docs' wiring is real.
    assert service is not None


def test_doc_example_workflow_shape_runs() -> None:
    """The "Workflows" docs section shows the action_step + workflow
    surface. Construct a 1-step workflow against a stub adapter and run it."""
    from collections.abc import Mapping, Sequence

    from sidekick.agent import (
        ActionDescriptor,
        ActionResult,
        SidekickActionService,
        SidekickWorkflow,
        WorkflowStepStatus,
        action_step,
        run_sidekick_workflow,
    )

    class _Stub:
        namespace = "doc"

        def describe(self) -> Sequence[ActionDescriptor]:
            return (
                ActionDescriptor(
                    action_id="doc.step",
                    summary="doc-example step",
                    params_schema={"type": "object"},
                    side_effects="read",
                    reversible=False,
                ),
            )

        def invoke(self, action_id: str, params: Mapping[str, object]) -> ActionResult:
            return ActionResult(ok=True, value="done")

    service = SidekickActionService()
    service.register(_Stub())
    workflow = SidekickWorkflow(
        name="docs",
        steps=(action_step("doc.step", {}),),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is True
    assert outcome.step_results[0].status == WorkflowStepStatus.COMPLETED


def test_doc_example_chip_envelope_serialises() -> None:
    """The "Chat surface integration" docs section shows the wire shape
    of one envelope. Construct one and assert it matches the documented
    keys."""
    import json

    from sidekick.agent import (
        ActionDescriptor,
        ActionResult,
        ChatActionEnvelope,
        PlannedStep,
        SidekickActionService,
        build_chip,
        serialize_envelope,
    )

    class _Stub:
        namespace = "doc"

        def describe(self):  # type: ignore[no-untyped-def]
            return (
                ActionDescriptor(
                    action_id="doc.run",
                    summary="Run a thing.",
                    params_schema={"type": "object"},
                    side_effects="destructive",
                    reversible=False,
                ),
            )

        def invoke(self, action_id, params):  # type: ignore[no-untyped-def]
            return ActionResult(ok=True)

    service = SidekickActionService()
    service.register(_Stub())
    chip = build_chip(
        step=PlannedStep(action_id="doc.run", params={}, rationale="docs"),
        service=service,
    )
    payload = serialize_envelope(ChatActionEnvelope(steps=(), chips=(chip,)))
    encoded = json.loads(json.dumps(payload))  # round-trip
    assert set(encoded["chips"][0]) == {
        "action_id",
        "summary",
        "params",
        "side_effects",
        "reversible",
        "rationale",
        "state",
        "error_message",
    }
    # The destructive chip starts locked per the docs.
    assert encoded["chips"][0]["state"] == "locked"
