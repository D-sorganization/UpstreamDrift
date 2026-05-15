# Sidekick — AI chat / agentic assistant

Sidekick is UpstreamDrift's in-app AI chat surface. It runs in two shells
that share a single design-token contract and a single tool registry, so
adding capabilities once makes them available in both places.

## Surfaces

| Shell           | File                                                                  | Notes                                                                                                                |
| --------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| React/Tauri     | [`ui/src/pages/Chat.tsx`](../../ui/src/pages/Chat.tsx) (route `/chat`)| Renders `ChatPanel` over the FastAPI WebSocket in `src/api/routes/chat_ws.py`. Styled with `var(--sidekick-*)` vars. |
| PyQt launcher   | [`src/shared/python/ai/gui/assistant_panel.py`](../../src/shared/python/ai/gui/assistant_panel.py) | `AIAssistantPanel`, embedded as splitter pane and as an `EmbeddableTool` tile.                                       |

Both shells display the same brand ("Sidekick") in their window/header
text and route messages through the same FastAPI chat service.

## Architecture

```
┌────────────────────────┐         ┌────────────────────────┐
│  React/Tauri (web)     │         │  PyQt launcher         │
│  ChatPanel.tsx         │         │  AIAssistantPanel      │
└────────┬───────────────┘         └────────┬───────────────┘
         │   var(--sidekick-*)              │   sidekick_tokens
         │  (themeClient.ts)                │  (sidekick_tokens.py)
         ▼                                  ▼
┌──────────────────────────────────────────────────────────┐
│  Design tokens — pinned in lock-step by                  │
│  tests/unit/theme/test_sidekick_parity.py                │
└──────────────────────────────────────────────────────────┘
         │ WebSocket                        │ WebSocket
         ▼                                  ▼
┌──────────────────────────────────────────────────────────┐
│  src/api/routes/chat_ws.py  (FastAPI)                    │
│  └── _maybe_inject_chat_context() ◀── chat_context.py    │
│                                            (ring buffer) │
└──────────────────────────────────────────────────────────┘
         │ tool calls
         ▼
┌──────────────────────────────────────────────────────────┐
│  src/shared/python/ai/tools/*  (e.g. sidekick_analytics) │
│  registered via sample_tools.register_golf_suite_tools   │
└──────────────────────────────────────────────────────────┘
```

## Design tokens

`src/shared/python/theme/sidekick_tokens.py` (Python) and
`ui/src/api/themeClient.ts` (TypeScript) each export:

- `COLOR_TOKEN_MAP` / `SIDEKICK_COLOR_TOKEN_MAP` — 21 canonical
  `sidekick.color.*` keys mapped onto active-theme color keys.
- `DEFAULT_SIDEKICK_TOKENS` / `SIDEKICK_FALLBACK_COLOR_TOKENS` —
  hex defaults plus spacing/radius/font tokens for environments
  without an active launcher theme.

The two maps are pinned together by
`tests/unit/theme/test_sidekick_parity.py`. If you add a token in one
language, add it in the other in the same PR — the parity test will
fail otherwise.

When styling a new chat-adjacent surface, prefer
`var(--sidekick-color-surface)` / `var(--sidekick-color-text)` /
`var(--sidekick-color-accent)` over raw hex or Tailwind grays. The
CSS variables live at `:root` in `ui/src/index.css` and are
auto-applied by `applyThemeToCSSVariables()` in `themeClient.ts`.

## Launcher tile

`src/tools/sidekick/_embed_adapter.py` registers Sidekick as an
`EmbeddableTool` via `register_embeddable_tool()`. The tile metadata
lives in `src/config/models.yaml`:

```yaml
- id: "sidekick"
  name: "Sidekick"
  description: "AI chat / agentic assistant"
  launcher:
    category: "tool"
    default_launch: "dock"
    status: "beta"
```

Right-click the tile → "Launch in Tab" / "Launch in Dock" opens the
panel in the embedded host the same way as `model_explorer` or
`starting_pose_matcher`. See
[`embedding_a_tool.md`](embedding_a_tool.md) for the protocol details.

## Chat context bridge

`src/shared/python/ai/chat_context.py` exposes a thread-safe ring buffer
the rest of the app can push to:

```python
from src.shared.python.ai.chat_context import record_event

record_event("diagnostic", {"name": "engine_check", "status": "ok"})
record_event("simulation", {"engine": "mujoco", "duration_s": 4.2})
```

`chat_ws.py` calls `_maybe_inject_chat_context(session)` at the start of
each `send` action; when the buffer is non-empty the formatted payload
is injected as a `system` message before the user prompt reaches the
model.

The dump applies two safety nets:

- **Redaction.** Leaf values whose key matches `(?i)password|token|secret|api_key|file_path` or whose string value matches `/home/` or `C:\` are replaced with `"<redacted>"`. Nested mappings and lists are traversed.
- **Size cap.** The serialized payload is truncated to ≤ 4 KB; oldest events are dropped until it fits.

Set `UPSTREAMDRIFT_SIDEKICK_CONTEXT=0` to disable injection (e.g. for
deterministic test runs).

## Agentic tools

The chat assistant invokes registered tools through the AI tool
registry. The current cross-engine example:

```python
from src.shared.python.ai.tools.sidekick_analytics import (
    summarize_simulation_run,
)

summary = summarize_simulation_run("run_2026_05_15_42")
# {"run_id": "...", "engine": "mujoco", "duration_s": 4.2,
#  "n_frames": 420, "key_metrics": {...},
#  "summary": "Mujoco run completed in 4.2 s over 420 frames..."}
```

Manifests are read from `~/.golf_modeling_suite/runs/<run_id>/manifest.json`
by default; override via `UPSTREAMDRIFT_SIM_RUNS_DIR`. The tool
rejects `run_id` values containing `/`, `\`, or `..` to prevent path
traversal.

Register new analytical surfaces alongside `sidekick_analytics.py` and
add the registration to
`src/shared/python/ai/sample_tools.register_golf_suite_tools` so the
default `ChatService` picks them up.

## Diagnostics

`LauncherDiagnostics.check_tools_sidebar()` (in
`src/launchers/launcher_diagnostics.py`) reports whether the optional
sibling [`D-sorganization/Tools`](https://github.com/D-sorganization/Tools)
sidebar — which receives the Sidekick design tokens via
`tools_sidebar_integration.install_tools_sidebar()` — is reachable. The
report uses the public probe
`gui_launcher.is_tools_sidebar_available()`.

## See also

- [`embedding_a_tool.md`](embedding_a_tool.md) — `EmbeddableTool`
  protocol and tile registration.
- [`../adr/0013-launcher-composability.md`](../adr/0013-launcher-composability.md)
  — launcher embedding design rationale.
- [AGENTS.md §B "Sidekick"](../../AGENTS.md) — fleet-level pointer for
  contributors and other agents.
