# Sidekick — Agentic action layer

Reference: ADR-0017 in [`docs/adr/0017-sidekick-agentic-action-layer.md`](../adr/0017-sidekick-agentic-action-layer.md).
Epic: [#5967](https://github.com/D-sorganization/UpstreamDrift/issues/5967).
Code: [`src/shared/python/sidekick/agent/`](../../src/shared/python/sidekick/agent/).

The agent layer makes Sidekick a first-class operator of its own
features. This page is for contributors who want to add a new action,
add a new host integration, write a workflow, or extend the chip UI.

## Module map

| Module               | Purpose                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `feature_catalog.py` | Self-knowledge index: build / lookup / search Sidekick features.                             |
| `action_service.py`  | The dispatch facade, descriptors, audit + policy + undo wiring.                              |
| `subtab_adapter.py`  | Drives `tools_sidebar` actions through a `SubtabActionPort`.                                 |
| `host_adapter.py`    | Bridges embedding-host actions via a `HostActionPort`.                                       |
| `planner.py`         | Validates LLM tool calls into `PlannedStep`s; exports tool-registry entries + system prompt. |
| `action_audit.py`    | `MemoryActionAudit` + `JsonlActionAudit` sinks (with redaction).                             |
| `access_policy.py`   | Default-deny policy with per-side-effects allowlists + confirmation gating.                  |
| `workflow_bridge.py` | Composes actions into workflows with `abort`/`retry`/`skip`/`ask_user` recovery.             |
| `chat_surface.py`    | Wire-shaped `ActionChipModel` consumed by both PyQt and React surfaces.                      |

## Architecture in one diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ AIAssistantPanel (PyQt) / ChatPanel (Tauri)                     │
│   ▲ build_chip(...) + with_confirmation()                       │
│ ChatActionEnvelope ← chat_surface.py                            │
│   ▲                                                             │
│ SidekickAgentPlanner   ← planner.py                             │
│   plan_from_tool_calls(...) / execute(step)                     │
│   ▼                                                             │
│ SidekickActionService  ← action_service.py                      │
│   ├─ SubtabAdapter   → SubtabActionPort  → tools_sidebar        │
│   ├─ HostAdapter     → HostActionPort    → launcher / ...       │
│   ├─ FeatureCatalog  (read-only catalogue of installed actions) │
│   ├─ Policy gating   ← access_policy.py                         │
│   ├─ Undo tokens                                                │
│   └─ Audit sink      ← action_audit.py                          │
└─────────────────────────────────────────────────────────────────┘
```

Every agentic action is one call to `service.invoke(action_id, params)`.
Adapters never reach into each other; UI code never reaches into the
service.

## Worked example: add a new subtab action

We'll add `subtab.workspace.clear` — wipe every workspace variable.
This action is reversible (undo restores the prior snapshot) and
destructive (so it gets the confirmation gate).

### 1. Extend the port

In `src/shared/python/sidekick/agent/subtab_adapter.py`, add the
method to `SubtabActionPort`:

```python
@runtime_checkable
class SubtabActionPort(Protocol):
    ...
    def workspace_clear(self) -> WorkspaceSnapshot:
        """Empty every workspace variable and return the prior snapshot."""
        ...
```

### 2. Register the descriptor

In `_build_descriptors()`:

```python
ActionDescriptor(
    action_id="subtab.workspace.clear",
    summary="Empty the workspace registry.",
    params_schema={"type": "object", "properties": {}},
    side_effects="destructive",   # gates the chip + policy
    reversible=True,              # makes undo possible
),
```

### 3. Wire the dispatch

In `SubtabAdapter.__init__`, add to `self._dispatch`:

```python
"subtab.workspace.clear": self._workspace_clear,
```

And the per-action method:

```python
def _workspace_clear(self, params: Mapping[str, Any]) -> ActionResult:
    prior = self._port.workspace_clear()
    return ActionResult(
        ok=True,
        value=None,
        metadata={
            "_undo": {
                "action_id": "subtab.workspace.restore",
                "params": {"snapshot": dict(prior.values)},
            },
        },
    )
```

The `_undo` metadata key is consumed by the service: it generates an
opaque token, stashes the inverse-action payload, and returns the
token in `result.undo_token`. The user never sees the metadata; the
chat layer never sees the inverse action.

### 4. Add tests first (TDD)

Before writing the dispatch method, add tests in
`tests/unit/sidekick/agent/test_subtab_adapter.py`:

```python
def test_workspace_clear_returns_prior_values() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke("subtab.workspace.clear", {})
    assert result.ok is True
    assert result.undo_token  # reversible


def test_workspace_clear_is_destructive() -> None:
    adapter = SubtabAdapter(port=_FakePort())
    by_id = {d.action_id: d for d in adapter.describe()}
    assert by_id["subtab.workspace.clear"].side_effects == "destructive"
```

Run `python3 -m pytest tests/unit/sidekick/agent/test_subtab_adapter.py`
— the tests fail (red). Implement until green. Run `ruff check` and
`ruff format --check`. Done.

### 5. (Future) Real port implementation

When you write the real `SubtabActionPort` against
`UnifiedToolsSidebar`, you implement `workspace_clear` by snapshotting
`WorkspaceRegistry`, calling `delete_variable` for every name, and
returning the snapshot. The adapter doesn't change.

## Worked example: add a new host integration

Pose Studio wants to expose two actions: load a pose and publish a
canonical pose.

### 1. Implement the port in the host

In Pose Studio's code (NOT in `sidekick/agent/`):

```python
# src/tools/pose_studio/sidekick_host_port.py

from sidekick.agent import HostActionPort, HostCapability, HostInvocationResult

class PoseStudioHostPort:
    host_id = "pose_studio"

    def list_capabilities(self) -> Sequence[HostCapability]:
        return (
            HostCapability(
                capability_id="host.pose_studio.load_pose",
                summary="Load a named pose into the studio.",
                params_schema={
                    "type": "object",
                    "properties": {"pose_id": {"type": "string"}},
                    "required": ["pose_id"],
                },
                requires_confirmation=False,
            ),
            HostCapability(
                capability_id="host.pose_studio.publish",
                summary="Publish the current pose over the realtime channel.",
                params_schema={"type": "object", "properties": {}},
                requires_confirmation=True,  # destructive: side-effects on subscribers
            ),
        )

    def invoke(self, capability_id, params) -> HostInvocationResult:
        if capability_id == "host.pose_studio.load_pose":
            # ... real implementation ...
            return HostInvocationResult(ok=True, value=None)
        ...
```

### 2. Register the port with the adapter

Wherever the host starts up:

```python
from sidekick.agent import HostAdapter, SidekickActionService

service = SidekickActionService(...)
host_adapter = HostAdapter(port=PoseStudioHostPort())
service.register(host_adapter)
```

Sidekick now lists `host.pose_studio.load_pose` and
`host.pose_studio.publish` as available actions; the chat chip for
`publish` starts `LOCKED` because the host marked it as requiring
confirmation.

### Dependency direction

The host imports from `sidekick.agent`. **Sidekick never imports the
host.** A static hygiene test asserts this for `host_adapter.py`.

## Workflows

Compose actions into a sequence with explicit recovery per step:

```python
from sidekick.agent import (
    SidekickWorkflow,
    action_step,
    run_sidekick_workflow,
)

workflow = SidekickWorkflow(
    name="size_wgs_reactor",
    steps=(
        action_step("subtab.show", {"tab_id": "calculator"}),
        action_step(
            "subtab.calculator.run",
            {"calculator_id": "wgs_reactor", "inputs": {...}},
            on_failure="ask_user",  # pause and let the user decide
        ),
        action_step(
            "subtab.workspace.set_variable",
            {"name": "wgs_result", "value": "<placeholder>"},
            on_failure="skip",       # nice to have, not critical
        ),
    ),
)

outcome = run_sidekick_workflow(workflow, service=service)
if outcome.completed:
    print(outcome.outputs)
```

The four recovery strategies (`abort`, `retry`, `skip`, `ask_user`)
mirror the vocabulary in `shared.python.ai.workflow_engine` so the
future migration to the unified engine is mechanical.

## Audit log

Wire the JSONL sink at service construction time:

```python
from pathlib import Path
import platformdirs
from sidekick.agent import (
    JsonlActionAudit,
    SidekickActionPolicy,
    SidekickActionService,
)

audit_path = Path(platformdirs.user_log_dir("sidekick")) / "actions.jsonl"
service = SidekickActionService(
    audit_sink=JsonlActionAudit(path=audit_path),
    policy=SidekickActionPolicy(
        allow_write={"subtab.workspace.set_variable", "subtab.focus"},
        allow_destructive={"subtab.workspace.clear"},
    ),
)
```

Every call lands as one JSON object on its own line, with sensitive
keys (`password`, `api_key`, ...) redacted. File-write failures
degrade to in-memory storage (`audit.tail`) so a full disk never
aborts dispatch.

## Chat surface integration

The chat code (PyQt panel or React/Tauri page) consumes
`serialize_envelope(envelope)` over the WebSocket. The payload shape is:

```jsonc
{
  "chips": [
    {
      "action_id": "subtab.calculator.run",
      "summary": "Run a named calculator with the given inputs.",
      "params": { "calculator_id": "wgs_reactor", "inputs": { ... } },
      "side_effects": "write",
      "reversible": false,
      "rationale": "user asked for the WGS H2/CO ratio",
      "state": "ready",
      "error_message": ""
    },
    {
      "action_id": "subtab.workspace.clear",
      "summary": "Empty the workspace registry.",
      "params": {},
      "side_effects": "destructive",
      "reversible": true,
      "rationale": "",
      "state": "locked",
      "error_message": ""
    }
  ]
}
```

`state` drives the Run button: `locked` disables it until the user
clicks Confirm (which stamps `params._confirmed=True` and flips the
state to `ready`).

## Conventions

- **Action ids** are dotted strings: `<namespace>.<verb>` or
  `<namespace>.<noun>.<verb>`. Namespaces are stable; verbs are
  imperative.
- **Frozen dataclasses** for every wire type. Validation in
  `__post_init__`.
- **No PyQt6** in `src/shared/python/sidekick/agent/` (asserted by the
  host adapter hygiene test; convention everywhere else).
- **Errors return, not raise.** Handlers translate user-visible failures
  into `ActionResult(ok=False, error=...)`. Bugs (programmer errors)
  raise — examples: `PlannerError`, `PendingUserDecision`.
- **Tests live with the implementation PR.** TDD: red → green →
  refactor. Aim for ≥ 90% branch coverage of new modules.
