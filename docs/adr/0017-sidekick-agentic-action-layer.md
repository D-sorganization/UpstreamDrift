# ADR-0017: Sidekick agentic action layer

- Status: Accepted
- Date: 2026-05-22
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: EPIC [#5967](https://github.com/D-sorganization/UpstreamDrift/issues/5967), sub-issues
  [#5970](https://github.com/D-sorganization/UpstreamDrift/issues/5970) (catalog),
  [#5971](https://github.com/D-sorganization/UpstreamDrift/issues/5971) (action service),
  [#5972](https://github.com/D-sorganization/UpstreamDrift/issues/5972) (subtab adapter),
  [#5973](https://github.com/D-sorganization/UpstreamDrift/issues/5973) (host adapter),
  [#5974](https://github.com/D-sorganization/UpstreamDrift/issues/5974) (planner),
  [#5975](https://github.com/D-sorganization/UpstreamDrift/issues/5975) (audit + undo + policy),
  [#5976](https://github.com/D-sorganization/UpstreamDrift/issues/5976) (workflows),
  [#5977](https://github.com/D-sorganization/UpstreamDrift/issues/5977) (chat surface),
  [#5978](https://github.com/D-sorganization/UpstreamDrift/issues/5978) (this ADR).

## Context

Before this epic, the Sidekick AI assistant was a conversational shell:
users typed, the LLM replied, and a handful of one-off tools in
`src/shared/python/ai/tools/` existed but had no uniform way to drive
Sidekick's own surfaces (the tools_sidebar subtabs, the calculators,
the workspace registry, the state-profile system) or the host
applications that embed Sidekick (the UpstreamDrift launcher per
ADR-0013, Pose Studio, model_explorer, etc.).

Concretely, every new "let the AI do X" request was becoming a one-off
bridge from `ai.tools` into a private subtab API:

- Each bridge invented its own param validation.
- Audit logging was ad-hoc and inconsistent.
- Destructive actions had no shared confirmation pattern.
- Tool-registry entries and system-prompt text drifted from each other.
- Sidekick had to import launcher internals to drive host actions —
  the wrong direction.

Cumulative effect: the 2026-05-21 adversarial review (#5907) flagged
"agentic-side coupling" as a near-term governance risk. Without a
single choke-point, every `ai.tools.*` module was free to bypass
validation and audit, and every new host integration meant another
private dependency edge from sidekick to launcher.

## Decision

Adopt a five-piece agentic layer under
`src/shared/python/sidekick/agent/`, each piece independently
unit-testable and replaceable:

### 1. Feature catalog (S1 / #5970)

`feature_catalog.py` exposes `build_feature_catalog()`,
`lookup_feature(id)`, and `search_features(query)` — Sidekick's
machine-readable self-knowledge index. Built by introspecting the
calculator and process-calculator packages, the theme module, and the
subtab help map. Subtab discovery uses AST parsing of
`help_content.py` so the catalog is robust against unrelated breakage
in the rest of `tools_sidebar`.

### 2. Action service + handler protocol (S2 / #5971)

`action_service.py` defines `SidekickActionHandler` (a runtime-checkable
`Protocol`) and `SidekickActionService` (the dispatch facade). Every
agentic action flows through `service.invoke(action_id, params)`. The
service owns:

- JSON-Schema validation of params (a small built-in validator covering
  the subset our adapters use).
- Audit recording on every call, including denied and errored ones.
- Optional policy check between validation and dispatch.
- Optional `dry_run` short-circuit for chat-side preview.
- Service-issued undo tokens for reversible actions.

Handlers never raise on user input — every error is translated to
`ActionResult(ok=False, error=...)` per ADR-0016.

### 3. Adapters (S3, S4 / #5972, #5973)

Two `SidekickActionHandler` implementations:

- `subtab_adapter.SubtabAdapter` — exposes the tools_sidebar surface
  via a thin `SubtabActionPort` Protocol. The adapter has zero direct
  PyQt6 imports; tests use a fake port; the real port (a follow-up)
  wraps `UnifiedToolsSidebar` + `WorkspaceRegistry` +
  `CommandHistoryController`.
- `host_adapter.HostAdapter` — exposes embedding-host capabilities
  (launcher tiles, theme switches, pose publishing, ...) via a
  `HostActionPort` Protocol that the host implements. Dependency
  direction is fixed at host → sidekick.agent.

### 4. Planner + tool-registry bridge (S5 / #5974)

`planner.SidekickAgentPlanner` validates LLM-emitted `ToolCall`s into
`PlannedStep`s and dispatches them. It also generates the AI
tool-registry entries (`sidekick.action.*` namespace) and the system
prompt — both derived from the same source (`service.list_actions()`)
so the LLM never sees an action in the prompt that isn't in the
registry.

### 5. Safety substrate (S6 / #5975)

Three modules:

- `action_audit.py` — `JsonlActionAudit` (file) and
  `MemoryActionAudit` (in-process), both with case-insensitive
  redaction of `{password, api_key, secret, token, auth, credential}`.
  The JSONL sink degrades to memory on filesystem failure; auditing
  must never abort dispatch.
- `access_policy.SidekickActionPolicy` — default-deny for write and
  destructive actions; destructive always requires
  `params["_confirmed"] is True` even when allowlisted.
- `SidekickActionService.undo(token)` — reverses a reversible action by
  dispatching the inverse the handler supplied via
  `result.metadata["_undo"]`. Tokens are service-owned and consumable
  exactly once.

### 6. Workflow runner (S7 / #5976)

`workflow_bridge.py` composes actions into ordered workflows with the
four standard recovery strategies (`abort`, `retry`, `skip`,
`ask_user`). `ask_user` raises `PendingUserDecision` so the chat layer
can interrupt and resume. The runner is a focused 200-line module
rather than a wrapper around `ai.workflow_engine.WorkflowEngine`,
which is currently in flux; the public surface mirrors the AI engine's
vocabulary so a future migration is mechanical.

### 7. Chat-side surface (S8 / #5977)

`chat_surface.py` owns the `ActionChipModel` wire format both UI
surfaces (PyQt assistant panel, React/Tauri ChatPanel) consume.
Destructive chips start `LOCKED`; `with_confirmation()` stamps
`_confirmed=True` and transitions to `READY`. Per-surface widget code
sits on top of this contract and is intentionally separated so this
module is fully headless-testable.

## Alternatives Considered

1. **Extend `ai.tool_registry` directly.** Rejected — registry is
   provider-agnostic and meant for any AI tool; we needed a Sidekick-
   specific namespace with audit and policy gates around it. The
   tool-registry bridge in S5 gives us the integration without the
   coupling.
2. **Per-adapter dispatchers.** Rejected — would each grow their own
   validation, error mapping, audit, and dry-run logic. The whole
   reason for the single choke-point is to keep those concerns DRY.
3. **Wire LLM directly into adapters.** Rejected — leaves the LLM
   unaudited and unpolicied, and forces every adapter to know about
   tool calling. The planner+service split keeps adapters dumb.
4. **Wrap `WorkflowEngine`.** Considered for S7; deferred to a
   follow-up. The engine is mid-refactor; a small parallel runner
   today is cheaper than tracking the moving target.

## Consequences

- **Positive:**
  - One audited choke-point for every agentic action.
  - Default-deny safety substrate (writes require allowlist;
    destructive requires confirmation).
  - Tool-registry entries and system prompt generated from a single
    source, eliminating drift.
  - Host integrations are dependency-inverted: launcher → sidekick,
    never the reverse.
  - Headless-testable end-to-end: 157 unit tests covering catalog,
    service, adapters, planner, audit, policy, undo, workflows, chips.
- **Negative:**
  - Adds a new public surface for adapters to learn. Mitigated by
    extensive module docstrings and a single worked example in
    `docs/sidekick/agent.md`.
  - Parallel workflow runner is a temporary doubling with
    `ai.workflow_engine`. Tracked for unification once the AI engine
    stabilises.
- **Follow-ups:**
  - Real `SubtabActionPort` implementation wrapping
    `UnifiedToolsSidebar` (depends on the upstream `tools_sidebar`
    registry refactor stabilising).
  - Launcher's `HostActionPort` implementation under
    `src/launchers/sidekick_host_port.py`.
  - PyQt6 chip widget + React `ActionChip` component (each on top of
    the `chat_surface` model).
  - Two starter workflows wired into `ai.workflow_definitions` per
    S7's epic body.

## Validation

- `pytest tests/unit/sidekick/agent/` — 157 tests covering every
  module. All green on this branch.
- `ruff check` and `ruff format --check` — clean on every module under
  `src/shared/python/sidekick/agent/` and `tests/unit/sidekick/agent/`.
- File-size budget — every module is under 700 lines; no exceptions
  required.
- Error-handling ratchet — no new `BLE001` / `F841` / `F401`; all
  exception handling uses `narrow_catch` or explicit narrow `except`
  clauses per ADR-0016.
- No PyQt6 imports anywhere in `src/shared/python/sidekick/agent/`;
  asserted by the import-boundary test for the host adapter.
