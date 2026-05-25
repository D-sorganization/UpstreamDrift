# ADR-0022: Chat / Sidekick boundary

- Status: Accepted
- Date: 2026-05-24
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: [#6098](https://github.com/D-sorganization/UpstreamDrift/issues/6098), [#5922](https://github.com/D-sorganization/UpstreamDrift/issues/5922), [#5967](https://github.com/D-sorganization/UpstreamDrift/issues/5967), [#5969](https://github.com/D-sorganization/UpstreamDrift/issues/5969)

## Context

UpstreamDrift now has two PyQt-side AI/chat surfaces plus the React/Tauri chat
surface:

1. `src/shared/python/chat/_chat_dock_widget_qt.py` exports the legacy
   `ChatDockWidget`, a portable `QDockWidget` still used by dashboards,
   engine launchers, quick-bar flows, and Sidekick runtime tabs.
2. `src/shared/python/ai/gui/assistant_panel.py` exports
   `AIAssistantPanel`, the richer PyQt Sidekick panel used by the
   standalone Sidekick window and by the current launcher-side Sidekick tool.
3. `ui/src/components/ui/ChatPanel.tsx` is the React/Tauri Sidekick chat
   surface bound to the shared `sidekick.color.*` token contract.

The repo has already moved the launcher away from a dedicated embedded AI panel:
`src/launchers/launcher_ui_setup.py` states that the canonical launcher chat
surface is the Sidekick dock's Chat tab and explicitly says not to reintroduce
another `AIAssistantPanel` there. At the same time, the legacy chat package
still exposes an embeddable `chat_assistant` path
(`src/shared/python/chat/sidekick_tool.py`) while `src/tools/sidekick/` exposes
the current `sidekick` embeddable tool backed by `AIAssistantPanel`.

Without an explicit boundary, three risks stay live:

- contributors treat `ChatDockWidget` and Sidekick as co-equal long-term entry
  points and keep adding features to both;
- duplicated launcher/embed entry points drift (`chat_assistant` vs `sidekick`);
- the file-size exception for `src/shared/python/chat/_chat_dock_widget_qt.py`
  reaches its 2026-07-31 expiry without a documented migration path.

## Decision

### 1. Canonical product surfaces

Sidekick is the canonical end-user AI/chat product going forward.

- **PyQt/desktop canonical surface:** `AIAssistantPanel`
  (`src/shared/python/ai/gui/assistant_panel.py`)
- **Standalone canonical shell:** `sidekick.standalone.window.StandaloneSidekickWindow`
  composed from `AIAssistantPanel` plus `UnifiedToolsSidebar`
- **React/Tauri canonical surface:** `ui/src/components/ui/ChatPanel.tsx`
- **Canonical utilities surface:** `src/shared/python/sidekick/ui/tools_sidebar/`

`ChatDockWidget` remains supported only as a **legacy compatibility surface**
for lightweight dock-style integrations that have not yet migrated to the
Sidekick tool stack. New launcher or standalone work must target the Sidekick
surfaces above, not `_chat_dock_widget_qt.py`.

### 2. Shared modules both surfaces must continue to reuse

The boundary is "different shells, shared infrastructure":

- Theme/token contract:
  `src/shared/python/theme/sidekick_tokens.py` and
  `ui/src/api/themeClient.ts`
- Agentic action layer:
  `src/shared/python/sidekick/agent/` per ADR-0017
- Tools/sidebar utilities:
  `src/shared/python/sidekick/ui/tools_sidebar/`
- Shared AI backend and prompts:
  `src/shared/python/ai/`
- Shared chat/session context:
  `src/shared/python/ai/chat_context.py`,
  `src/api/routes/chat_ws.py`, and the shared session-id handshake called out
  in `src/launchers/launcher_ui_setup.py`

The shells may differ (`QDockWidget`, `QWidget`, React component), but they
should not fork the design-token, action, or AI-service layers.

### 3. Duplication that should not persist

The repo should treat the following as migration targets, not permanent parallel
designs:

1. **Two embeddable entry points for chat UX**
   `src/shared/python/chat/sidekick_tool.py` (`chat_assistant`) and
   `src/tools/sidekick/_embed_adapter.py` (`sidekick`) compete for the same
   launcher real estate.
2. **Legacy-vs-Sidekick PyQt shells**
   `ChatDockWidget` and `AIAssistantPanel` both represent a PyQt chat surface,
   but only one should continue to accrete launcher/standalone features.
3. **Theme adaptation overlap**
   `_theme_protocol.py` exists to keep `ChatDockWidget` visually aligned, but
   Sidekick tokens are already the authoritative styling contract elsewhere.

This ADR does **not** require an immediate rewrite. It freezes the direction:
new product work goes through Sidekick, while `ChatDockWidget` is reduced to
compatibility scope.

### 4. Migration plan for `_chat_dock_widget_qt.py`

The file-size exception from issue #5922 stays in place temporarily, but its
destiny is now explicit:

1. **Freeze expansion**: do not add new launcher/standalone features to
   `ChatDockWidget`; only bug fixes or compatibility work are allowed there.
2. **Consolidate entry points**: retire the legacy `chat_assistant` launcher
   path in favor of the `sidekick` tool once remaining compatibility hosts have
   a documented migration path.
3. **Extract shared pieces**: move any reusable session/theme/composer/message
   helpers out of `_chat_dock_widget_qt.py` into smaller shared modules that
   both legacy and Sidekick shells can consume.
4. **Shrink or wrap**: by or before 2026-07-31, `_chat_dock_widget_qt.py`
   should either become a thin compatibility wrapper around shared components or
   be decomposed into focused modules small enough to drop the exception.

The file-size-budget entry for `_chat_dock_widget_qt.py` must reference this
ADR so the expiry is tied to a concrete plan rather than a generic "split it
later" note.

## Alternatives Considered

1. **Make `ChatDockWidget` the canonical Sidekick surface.**
   Rejected: current launcher and standalone direction already centers
   `AIAssistantPanel`, `UnifiedToolsSidebar`, and ADR-0018.
2. **Keep `ChatDockWidget` and Sidekick as permanent co-equal product shells.**
   Rejected: it would preserve duplicate launcher/embed paths and guarantee
   continued drift.
3. **Rewrite everything immediately onto one new shell.**
   Rejected: issue #6098 is a boundary-review task, not a rewrite authorization,
   and there are still compatibility consumers of `ChatDockWidget`.

## Consequences

- Positive:
  - New PyQt and standalone work has one canonical path: Sidekick.
  - The `_chat_dock_widget_qt.py` exception now has a documented end state.
  - Reviewers can reject new feature work that lands on the legacy chat dock by
    pointing to an accepted ADR.
- Negative:
  - The repo will carry both shells for a transition period.
  - Compatibility consumers still need follow-up work before the legacy entry
    point can disappear.
- Follow-ups:
  - Follow-up issue [#6119](https://github.com/D-sorganization/UpstreamDrift/issues/6119)
    tracks the entry-point consolidation and `_chat_dock_widget_qt.py`
    decomposition plan.
  - Cross-link this ADR/issue from the Sidekick epics #5967 and #5969 so the
    owners see the boundary decision.

## Validation

- `tests/unit/repo_hygiene/test_sidekick_docs.py` asserts that ADR-0022 exists,
  stays accepted, names the canonical and legacy surfaces, cross-links
  #5922/#5967/#5969/#6098, updates the ADR index, and ties the chat-dock
  file-size-budget rationale to ADR-0022.
- `python3 -m ruff check tests/unit/repo_hygiene/test_sidekick_docs.py`
- `python3 -m ruff format --check tests/unit/repo_hygiene/test_sidekick_docs.py`
- `python3 -m pytest tests/unit/repo_hygiene/test_sidekick_docs.py -q`
