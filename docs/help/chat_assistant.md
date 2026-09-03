---
title: AI Assistant
tile_id: chat_assistant
status: complete
---

# AI Assistant

## Purpose

AI Assistant is the browser chat page. You type a question, optionally
attach files, and a language model answers with streamed text. The
server can prepend a compact summary of recent app state so the
assistant knows which engine and simulation you are working with
without you pasting it in.

## Inputs

- A chat message as plain text, sent as
  `{"action": "send", "message": "..."}` over the WebSocket.
- Optional `engine_context` string, for example `"mujoco"`.
- Optional attachments, each `{"name", "mime", "data"}` with `data`
  base64-encoded. The composer accepts a file picker, drag-and-drop, and
  pasted images.
- Control actions `{"action": "history"}` and
  `{"action": "new_session"}`.
- Automatically injected context, unless
  `UPSTREAMDRIFT_SIDEKICK_CONTEXT` is set to `"0"`: a "recent app state"
  system message built by `src.shared.python.ai.chat_context`, capped at
  roughly 4 KB of serialised size (last N events).
- An AI provider configured server-side. `ChatService` delegates
  inference to the configured adapter - Ollama, OpenAI, Anthropic, or
  Gemini.

## Outputs

Server-to-client frames on the same socket:

| Frame | Payload |
| --- | --- |
| `session_info` | `session_id` |
| `chunk` | `content`, one streamed fragment of the reply |
| `complete` | `session_id`, reply finished |
| `history` | `messages` array |
| `error` | `detail` |

Assistant turns are rendered as Markdown in the page. Sessions are
persisted as files under `~/.golf_modeling_suite/chat_sessions/` on each
message, so they can be shared across processes.

## Method

The page is `ui/src/pages/Chat.tsx`, a 20-line wrapper that centres
`ui/src/components/ui/ChatPanel.tsx`. `ChatPanel` opens a WebSocket to
`{VITE_API_URL || ws://localhost:8000}/ws/chat/new` and reconnects with
exponential backoff capped at 30 s.

The server side is
[`src/api/routes/chat_ws.py`](../../src/api/routes/chat_ws.py), which
runs the shared protocol loop from
`src.shared.python.chat.websocket_protocol.run_chat_websocket_protocol`
and authenticates the socket via `src/api/auth/ws_auth.resolve_ws_user`.
Session state, TTL eviction, and adapter dispatch live in
[`src/api/services/chat_service.py`](../../src/api/services/chat_service.py):
`ChatService` holds at most `MAX_SESSIONS = 50` sessions with a
`SESSION_TTL_SECONDS = 7200` (2 hour) TTL. It is constructed in
`src/api/server.py` with an app-state provider wired to the server's
`EngineManager` and `SimulationService`, so injected context reflects
live engine and simulation state rather than event history alone.

Injected context is deduplicated: a truncated SHA-256 digest is stored
in the session metadata so unchanged app state is not re-sent on every
message. The context builder strips keys and values matching secrets and
PII patterns, and the module docstring states that no file paths or
credentials reach the assistant.

## Limitations

- **The registry gives this tile an empty `path`, deliberately.** Its
  `surface_reason` reads "web chat page (/chat); the desktop equivalent
  is the sidekick dock", and `surfaces` is `["web"]` only. There is no
  PyQt6 AI Assistant window - on the desktop you use the Sidekick dock
  instead.
- **The registry's `capabilities` overstate this tile.** It lists
  `calculator`, `workspace`, and `data_explorer`. Those integrations
  belong to the Sidekick tools sidebar
  (`src/shared/python/sidekick/ui/tools_sidebar/`), not to `ChatPanel`.
  The web chat page's own features are attachments, Markdown rendering,
  retry, quick-action presets, and reconnect. Do not expect a calculator
  or a workspace variable store on this page.
- Sessions are capped and expire. At most 50 live sessions, each evicted
  after 2 hours of inactivity.
- It needs a configured provider. With no adapter credentials or no
  reachable local model, the page connects but cannot answer.
- The assistant sees a summary, not your data. Injected context is
  capped near 4 KB and redacted; it cannot read your dataset files or
  your filesystem.
- Persistence is per-machine plain files in the home directory, with no
  access control beyond filesystem permissions.

## See Also
- [Sidekick](sidekick.md) - the desktop equivalent of this tile
- [Data Explorer](data_explorer.md)
