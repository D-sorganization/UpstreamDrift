# Sidekick — AI Chat / Agentic Assistant

Sidekick is UpstreamDrift's in-app AI assistant. It runs in two host shells
that share a single design-token contract and a single AI tool registry, so
adding a capability once makes it available in both places.

**Sidekick can also run standalone** — see [standalone.md](standalone.md) for
the `pip install` path and the one-file binary download.

---

## Surfaces

| Shell                   | Entry point                                                                                                                                                     | Notes                                                                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PyQt launcher panel** | [`src/shared/python/ai/gui/assistant_panel.py`](../../src/shared/python/ai/gui/assistant_panel.py) (`AIAssistantPanel`, window title "Sidekick")                | Embedded as a splitter pane in `src/launchers/launcher_ui_setup.py` and as a registered `EmbeddableTool` tile so users can open it via right-click → "Launch in Dock" |
| **React/Tauri shell**   | [`ui/src/pages/Chat.tsx`](../../ui/src/pages/Chat.tsx) (route `/chat`) renders [`ui/src/components/ui/ChatPanel.tsx`](../../ui/src/components/ui/ChatPanel.tsx) | Styled with `var(--sidekick-color-*)` CSS variables; communicates over the FastAPI WebSocket at `src/api/routes/chat_ws.py`                                           |
| **Standalone app**      | `sidekick gui` (console script) or downloaded binary                                                                                                            | No UpstreamDrift launcher required — see [standalone.md](standalone.md)                                                                                               |

Both embedded surfaces display the brand name "Sidekick" in their window /
header text and route messages through the same FastAPI chat service.

---

## Layout (ASCII)

```
┌──────────────────────────────────────────────────────────┐
│  PyQt launcher                   React / Tauri (web)     │
│  ┌──────────────────────────┐   ┌──────────────────────┐ │
│  │ AIAssistantPanel         │   │ ChatPanel.tsx         │ │
│  │  ┌─── sidebar ──────┐    │   │  (route /chat)        │ │
│  │  │ Sessions         │    │   └──────────┬───────────┘ │
│  │  │ History          │    │              │              │
│  │  │ Memory           │    │     var(--sidekick-*)       │
│  │  └──────────────────┘    │                            │
│  │  ┌─── transcript ───┐    │                            │
│  │  │ MessageWidget    │    │     sidekick_tokens.py     │
│  │  └──────────────────┘    │     (Python tokens)        │
│  │  ┌─── composer ─────┐    │                            │
│  │  │ ChatInput        │    │                            │
│  │  │ Send button      │    │                            │
│  │  └──────────────────┘    │                            │
│  └──────────────────────────┘                            │
│                    │  WebSocket (both shells)             │
│          ┌─────────▼────────────────────────────────┐    │
│          │  src/api/routes/chat_ws.py  (FastAPI)     │    │
│          │  └── _maybe_inject_chat_context()         │    │
│          └─────────┬────────────────────────────────┘    │
│                    │  tool calls                          │
│          ┌─────────▼────────────────────────────────┐    │
│          │  AI Tool Registry                         │    │
│          │  src/shared/python/ai/tools/              │    │
│          │  e.g. sidekick_analytics.py               │    │
│          └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Design tokens

`src/shared/python/theme/sidekick_tokens.py` (Python) and
`ui/src/api/themeClient.ts` (TypeScript) export matching maps:

- **`COLOR_TOKEN_MAP`** — 21 canonical `sidekick.color.*` keys
  mapped onto active launcher theme color keys.
- **`DEFAULT_SIDEKICK_TOKENS`** — hex fallbacks for environments
  without a running launcher theme.
- **Spacing / radius / font** tokens follow the same pattern.

The two maps are pinned together by
[`tests/unit/theme/test_sidekick_parity.py`](../../tests/unit/theme/test_sidekick_parity.py).
If you add a token in one language, add it in the other in the same PR —
the parity test will fail otherwise.

When styling a new chat-adjacent surface, always prefer:

```css
var(--sidekick-color-surface)
var(--sidekick-color-text)
var(--sidekick-color-accent)
```

over raw hex or Tailwind grays.

---

## Embedding Sidekick as a launcher tile

Sidekick is registered as an `EmbeddableTool` via
[`src/tools/sidekick/_embed_adapter.py`](../../src/tools/sidekick/_embed_adapter.py).
The tile entry in `src/config/models.yaml`:

```yaml
- id: "sidekick"
  name: "Sidekick"
  description: "AI chat / agentic assistant"
  launcher:
    category: "tool"
    default_launch: "dock"
    status: "beta"
```

Key points about the adapter:

- Sets `prefers_dock=True` — Sidekick is a sidebar tool, not a primary
  workspace, so it docks rather than opening in a new tab.
- `create_main_widget` is **idempotent**: opening Sidekick twice reuses
  the same widget instance.
- `cleanup()` is **idempotent**: safe to call on tab close, window close,
  or test fixture teardown.
- PyQt6 imports are **lazy** (inside `create_main_widget`), so the adapter
  module can be introspected on headless CI without a display server.

For a full walkthrough of the `EmbeddableTool` protocol, see
[`docs/development/embedding_a_tool.md`](../development/embedding_a_tool.md).

---

## Chat context bridge

`src/shared/python/ai/chat_context.py` exposes a thread-safe ring buffer
the rest of the app can push events to:

```python
from src.shared.python.ai.chat_context import record_event

record_event("diagnostic", {"name": "engine_check", "status": "ok"})
record_event("simulation", {"engine": "mujoco", "duration_s": 4.2})
```

`chat_ws.py` calls `_maybe_inject_chat_context(session)` at the start of
each `send` action; when the buffer is non-empty the formatted payload is
injected as a `system` message before the user prompt reaches the model.

**Safety nets applied to every dump:**

| Protection | Behavior                                                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Redaction  | Keys matching `password`, `token`, `secret`, `api_key` and values matching `/home/` or `C:\` are replaced with `"<redacted>"` |
| Size cap   | Payload is truncated to ≤ 4 KB; oldest events dropped first                                                                   |

Disable injection for deterministic test runs:

```bash
export UPSTREAMDRIFT_SIDEKICK_CONTEXT=0
```

---

## Adding an agentic tool

New analytical surfaces register through the AI tool registry. The
canonical cross-engine example:

```python
# src/shared/python/ai/tools/sidekick_analytics.py
from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

registry = get_global_registry()

@registry.register(
    name="summarize_simulation_run",
    description="Summarize a completed simulation run by run_id.",
    category=ToolCategory.ANALYSIS,
)
def summarize_simulation_run(run_id: str) -> dict:
    """Return a summary dict for the given simulation run."""
    ...
```

After defining the function, register it in
`src/shared/python/ai/sample_tools.register_golf_suite_tools` so the
default `ChatService` picks it up on startup.

The system prompt in `src/shared/python/ai/system_prompts.py` advertises
registered tools to the assistant automatically — no prompt edits needed.

---

## See also

- [AGENTS.md §B "Sidekick"](../../AGENTS.md) — contributor quick-reference
  with all file pointers.
- [`docs/development/embedding_a_tool.md`](../development/embedding_a_tool.md)
  — `EmbeddableTool` protocol and tile registration guide.
- [`docs/development/sidekick.md`](../development/sidekick.md) — deeper
  architecture and design-token notes.
- [`docs/development/realtime_ipc.md`](../development/realtime_ipc.md)
  — pub-sub IPC for tools that publish live state.
- [`src/tools/sidekick/_embed_adapter.py`](../../src/tools/sidekick/_embed_adapter.py)
  — minimal `EmbeddableTool` adapter for the PyQt panel.
- [`tests/unit/theme/test_sidekick_parity.py`](../../tests/unit/theme/test_sidekick_parity.py)
  — design-token parity test (Python ↔ TypeScript).
