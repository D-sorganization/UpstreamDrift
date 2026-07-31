# Web chat lacks live app/engine context — desktop Sidekick can query the running simulation, web cannot

## Gap (PyQt6 = model)

The desktop Sidekick sidebar (`src/launchers/launcher_sidekick_sidebar.py` + `src/shared/python/ai/gui/`) shares the launcher process: the assistant can introspect the loaded engine, query live simulation state, and act on the app. The web `Chat` page (`ui/src/pages/Chat.tsx` → `WS /api/ws/chat`) reaches `ChatService` (`src/api/services/chat_service.py`), which was wired to an app-state provider for the desktop path (#5470), but the web session does not carry the API server's engine/simulation context — chat answers are blind to what the user is simulating in the web app.

## Proposed fix

1. Wire `ChatService`'s app-state provider to the API server's `EngineManager`/simulation session state, so a web chat session sees: loaded engines, active model, last/current simulation config, latest analysis summaries, and recent diagnostics — the same context contract the desktop provider fills.
2. Define that context contract once (shared dataclass/Pydantic model) so desktop and API providers can't drift — add a parity test asserting both providers populate the same schema.
3. Web UX: show a small "context: mujoco · golf_swing.urdf · last run 3.0s" chip in the chat header so users know what the assistant can see (mirrors desktop behavior).
4. Explicitly OUT of scope (record as `exempt` in the feature-parity registry, see exemptions issue): OS terminal, Python REPL, Workspace, Jupyter, and Skills tabs — desktop-native by design; remote shells from a browser are a security non-starter under the current local-server model.

## Acceptance criteria

- [ ] Web chat answers questions about the currently loaded engine/model/run ("what engine am I using?", "summarize my last run")
- [ ] One shared context schema, two providers, parity test
- [ ] Registry entries: chat.live-context = parity; sidekick.{terminal,repl,jupyter,skills} = exempt with reason

## References

- #5470 (closed — wired app-state into chat for desktop)
- `src/api/routes/chat_ws.py`, `src/api/services/chat_service.py`
- Related UI bugfixes already tracked in #7426 (attachments/streaming indicator) — don't duplicate
