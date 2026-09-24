---
title: Sidekick
tile_id: sidekick
status: complete
---

# Sidekick

## Purpose

Sidekick is the desktop AI assistant panel. It docks alongside your work
in the launcher and holds a running conversation: you pick a provider and
model, choose an access mode, ask questions, and read streamed replies.
It is the desktop counterpart to the browser AI Assistant page.

## Inputs

- Typed messages in the panel's input area.
- Provider and model selection from the header combos
  (`_provider_combo`, `_model_combo`).
- An access mode (`_mode_combo`), coerced by
  `src.shared.python.ai.access_policy.coerce_access_mode`; the chosen
  `ChatAccessMode` decides which tool declarations
  (`tool_declarations_for_access_mode`) the assistant may use.
- Provider credentials, resolved by `AdapterLifecycleManager`. API keys
  are handled by `_api_keys.py` and configured through the panel's
  settings dialog.
- Optionally a codebase index, built by `IndexingController` for RAG
  retrieval.
- Optionally a `sidekick_action_service`: if the launcher host exposes an
  attribute of that name, the embed adapter passes it to the panel via
  `set_action_service`, letting the assistant drive host actions.

## Outputs

- Streamed assistant replies in the scrollable message log, managed by
  `MessageDisplayController`.
- Persisted chat history. `ChatSessionManager` auto-persists sessions, so
  the adapter reports `is_dirty()` as `False` - there is no unsaved
  buffer and the launcher will not prompt on close.
- Exported conversations via `chat_export.py`.
- A RAG index over the codebase, when indexing is run.
- Any host-side effect performed through the action service, gated by the
  active access mode.

## Method

The tile is an adapter, not an implementation.
[`src/tools/sidekick/_embed_adapter.py`](../../src/tools/sidekick/_embed_adapter.py)
is one of only two files in `src/tools/sidekick/`; it implements the
`EmbeddableTool` protocol and, in `create_main_widget`, lazily imports
and instantiates
[`AIAssistantPanel`](../../src/shared/python/ai/gui/assistant_panel.py).
It declares `prefers_dock=True` with a 360x480 px minimum, matching the
panel's side-dock shape. The import is lazy so the adapter registers
cleanly in headless contexts such as CI and docs builds where PyQt6 may
be unavailable.

`AIAssistantPanel` is itself a thin coordinator over four controllers:
`PanelHeaderController` (header strip), `MessageDisplayController`
(message log), `AdapterLifecycleManager` (provider and key resolution,
adapter creation), and `IndexingController` (RAG indexing lifecycle).
Backgrounding is beneficial and enabled by default: the adapter's own
comment notes that keeping the panel alive while hidden preserves
conversation state cheaply and holds no scarce GPU resource.

## Limitations

- **Despite the tile name, this is not the vendored Tools Sidekick.**
  The adapter wraps `AIAssistantPanel` from
  `src/shared/python/ai/gui/`. The separate Sidekick UI under
  `src/shared/python/sidekick/` - with its tools sidebar, workspace
  variables, and calculator tabs - is a different component and is not
  what this tile opens.
- `vendor/ud-tools/` is a pinned, vendored copy of the Tools repository.
  Never edit it from this repo; fixes belong upstream in Tools.
- Cleanup is best-effort. `AIAssistantPanel` exposes no `cleanup` hook,
  so the adapter falls back to `deleteLater()`.
- It needs a configured provider and, for hosted models, network access
  and valid credentials. Without them the panel opens but cannot reply.
- Maturity is **beta**.
- Host actions are only available when the launcher supplies a
  `sidekick_action_service`; embedded in a host that does not, the
  assistant is conversation-only.

## See Also
- [AI Assistant](chat_assistant.md) - the web equivalent of this tile
- [Vendored Tools repository notes](../../vendor/README.md)
