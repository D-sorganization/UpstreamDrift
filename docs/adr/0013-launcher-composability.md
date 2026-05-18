# ADR 0013: Launcher Composability — Embeddable-tool contract and IPC layer

- Status: Accepted
- Date: 2026-05-09
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: EPIC
  [#4993](https://github.com/D-sorganization/UpstreamDrift/issues/4993),
  this ADR closes the design portion of
  [#5000](https://github.com/D-sorganization/UpstreamDrift/issues/5000).
  Builds on the per-subtask PRs:
  [#5003](https://github.com/D-sorganization/UpstreamDrift/pull/5003)
  (embeddable-tool contract foundation, closes
  [#4994](https://github.com/D-sorganization/UpstreamDrift/issues/4994)),
  [#5013](https://github.com/D-sorganization/UpstreamDrift/pull/5013)
  (tab/dock host widget, closes
  [#4995](https://github.com/D-sorganization/UpstreamDrift/issues/4995)),
  [#5044](https://github.com/D-sorganization/UpstreamDrift/pull/5044)
  (launch-mode routing, closes
  [#4996](https://github.com/D-sorganization/UpstreamDrift/issues/4996)),
  [#5006](https://github.com/D-sorganization/UpstreamDrift/pull/5006)
  (file + WebSocket pub-sub IPC layer, closes
  [#4997](https://github.com/D-sorganization/UpstreamDrift/issues/4997)),
  [#5012](https://github.com/D-sorganization/UpstreamDrift/pull/5012)
  (Pose Studio `create_main_widget()`, part of
  [#4998](https://github.com/D-sorganization/UpstreamDrift/issues/4998))
  and the cross-tool live-pose demo
  ([#4999](https://github.com/D-sorganization/UpstreamDrift/issues/4999)).

## Context

Until this EPIC, every tile in the UpstreamDriftLauncher launched its tool via
`subprocess.Popen` with the system Python. The launcher acted as a
shortcut grid; once a tool was running it was a separate top-level
window with no relationship to the launcher process. Two problems
followed:

1. **No in-app composition.** Users who wanted to compare two tools
   side-by-side had to alt-tab between top-level windows. There was no
   way to dock one tool as a sidebar to another, no shared
   keyboard-shortcut surface, and no shared layout persistence.
2. **No cross-tool data flow.** Pose Studio publishing a canonical
   pose, a downstream subscriber re-rendering it as an FK skeleton,
   and a third tool reacting to that — none of this was possible
   without each tool re-implementing its own IPC. The MATLAB-side and
   FastAPI-side tools each had their own ad-hoc channels (file polling
   for the former, WebSocket for the latter); the desktop tools had
   nothing.

The launch model also encoded a hidden assumption: every tool ran
its own `QApplication` and owned its own GL context. That was true
for one or two tools (legacy Drake dashboards) but wrong for the
modern `src/tools/` cohort — they all use the shared theme stack and
matplotlib QtAgg backend and would happily live as child widgets.

The EPIC body
([#4993](https://github.com/D-sorganization/UpstreamDrift/issues/4993))
calls out the user-visible target: open Pose Studio in a tab, open a
canonical-pose subscriber tool in a dock, drag a slider, watch both
update at 30 Hz. Hitting that target needs three things simultaneously:

- A **contract** that lets tools opt in to embedding without forcing
  every tool to refactor at once.
- A **host widget** that can mount embedded tools as tabs and as
  dock widgets, with sensible close, focus, and dirty-state behaviour.
- A **pub-sub facade** with a transport that handles 30 Hz canonical
  poses without 100+ ms of disk latency.

## Decision

We adopt a three-piece architecture, each piece independently
unit-testable and replaceable:

### 1. `EmbeddableTool` Protocol + capability declaration

Lives in `src/shared/python/launcher_embed/`:

- `EmbedCapabilities` — frozen dataclass that declares
  `supports_embedded`, `prefers_dock`, `min_size`, and
  `requires_separate_qapplication`. Validated in `__post_init__`.
- `EmbeddableTool` — `runtime_checkable` Protocol with `tool_id`,
  `embed_capabilities()`, `create_main_widget(parent)`, `cleanup()`,
  and `is_dirty()`. Importing this module does **not** import PyQt6;
  widget types are spelled `typing.Any` so registries on headless
  CI can still introspect the contract.
- A small registry (`registry.py`) lets tools `register_embeddable_tool(tool)`
  and lets the launcher query `is_embeddable(tool_id)` and
  `get_embeddable_tool(tool_id)`.

### 2. `EmbeddedHostWidget` — tabs + docks under one roof

Lives in `src/launchers/embedded_host.py`. A `QWidget` whose central
area is a `QTabWidget` and whose internal `QMainWindow` provides a
dock surface so callers can add `QDockWidget` instances without the
host itself having to be a top-level window. Public API:
`open_tab(tool_id)`, `close_tab(target)`, `open_dock(tool_id, area)`,
`close_dock(tool_id)`, `set_focus_mode(enabled)`, `state_snapshot()`,
`restore_state(state)`. Dirty tools prompt the user via `QMessageBox`
on close. Double-clicking the tab bar toggles focus mode (the tab bar
hides so the active tab fills the host).

### 3. `LaunchMode` + `resolve_launch_mode()` routing

Lives in `src/launchers/launch_routing.py`. A pure-Python (PyQt-free)
helper that resolves a requested `LaunchMode` (`AUTO`, `NEW_WINDOW`,
`TAB`, `DOCK`, `EXTERNAL`) plus a model record into the concrete
mode that `LauncherSimulationMixin.launch_model()` should dispatch.
Resolution prefers `model.launcher.default_launch` if declared, then
`prefers_dock` if the tool's `EmbedCapabilities` requests it,
otherwise `TAB`. Tools that aren't embeddable always fall through
to `NEW_WINDOW`.

### 4. `realtime` pub-sub facade

Lives in `src/shared/python/realtime/`. One `subscribe(channel, callback)`
/ `publish(channel, payload)` API on top of two transports:

- **File transport** for low-frequency channels — JSONL append under
  `~/.upstream_drift/realtime/<channel>.jsonl`. Polling subscribers,
  durable across process boundaries, easy to inspect with `tail -F`.
- **WebSocket transport** for high-frequency channels — multiplexed
  through the existing FastAPI server (`src/api/`). Sub-millisecond
  fan-out within the same host.

Channel-to-transport routing lives in a single registry (`channels.py`).
Tool authors call `subscribe`/`publish` and the registry picks the
transport; they don't hard-code either side. Default for unregistered
channels is the file transport (durable, low setup cost, easy to
debug).

The cross-tool live-pose demo
([#4999](https://github.com/D-sorganization/UpstreamDrift/issues/4999))
exercises all three pieces end-to-end: Pose Studio refactored to
expose `create_main_widget()`, a tiny `pose_subscriber_demo`
embeddable tool, and the WebSocket transport carrying
`pose/canonical` at 30 Hz between them inside one launcher window.

## Alternatives Considered

1. **`QMdiArea` instead of `QTabWidget` for the host.** Qt's MDI area
   gives free-floating sub-windows-within-the-window. **Rejected**:
   the desktop-app audience for this project (biomechanics
   researchers, not Eclipse refugees) found MDI confusing in
   prototypes — too many overlapping subwindows, a second window
   manager bolted on top of the OS one. Tabs + docks is the modern
   IDE pattern (VS Code, JetBrains, Qt Creator itself) and matches
   user mental models.
2. **REST-only IPC.** Reuse the existing FastAPI surface for
   request/response only; have subscribers poll. **Rejected**: HTTP
   round-trip latency on Windows loopback is 5-15 ms per request even
   with keep-alive; at 30 Hz that's 30-50 % of the budget gone before
   the payload is serialised. Subscribers would also have to manage
   their own polling cadence per channel.
3. **Pure file-based IPC for everything.** A single transport, no
   server dependency, durable. **Rejected**: file watchers on Windows
   add 100-200 ms of latency in the worst case (NTFS notification
   coalescing); JSONL append + read coordination across processes
   needs locking we'd otherwise avoid. Acceptable for low-frequency
   channels (`status/*`, configuration changes), unacceptable for
   `pose/canonical` and the planned `engine/*/state` channels.
4. **Qt LocalSocket / DBus.** Qt's local-socket IPC is fast and
   in-process. **Rejected**: ties the IPC layer to Qt, which means
   the MATLAB-side and any future headless tool (CLI fitter, batch
   runner) cannot subscribe. DBus is Linux-only in practice — the
   Windows build is unmaintained — and we need Windows as a
   first-class platform per [CLAUDE.md](../../CLAUDE.md).
5. **Per-tool ad-hoc IPC, no facade.** What we had before. **Rejected**
   for the same reason ADR 0012 rejected per-engine ad-hoc converters:
   one shared abstraction with one transport-routing point is far
   cheaper to maintain than N tools each rolling their own.
6. **A single `QApplication`-shared-everything model with no
   embeddable contract.** Rip out `subprocess.Popen`, run every tool
   as a child widget unconditionally. **Rejected**: a handful of
   legacy tools genuinely need their own `QApplication` (custom GL
   contexts, pygame). The Protocol-with-capability-flag approach lets
   them keep working as standalone subprocesses while the modern
   tools migrate to embedding.

## Consequences

- **Positive:**
  - Tools opt in to embedding incrementally — Pose Studio first,
    then the rest of `src/tools/` over the next few subtasks. Nothing
    breaks for tools that haven't migrated.
  - The dispatcher (`LauncherSimulationMixin.launch_model()`) is a
    one-line conditional on the resolved `LaunchMode`; legacy
    `subprocess.Popen` is the `NEW_WINDOW` / `EXTERNAL` branch.
  - Pub-sub channel definitions become a single source of truth
    (`channels.py`). Tools publish without caring whether the
    subscriber is local, remote, or another instance of themselves.
  - Layout persistence is free: `EmbeddedHostWidget.state_snapshot()`
    serialises to a small JSON dict that the launcher can save and
    restore between sessions.
- **Negative:**
  - `EmbeddedHostWidget` keeps a hidden internal `QMainWindow` to
    host docks, which means dock state lives one level deeper than a
    naïve subclass would expect. Documented in
    [`docs/development/embedding_a_tool.md`](../development/embedding_a_tool.md).
  - The pub-sub layer adds one network dependency (FastAPI must be
    reachable at `localhost:<port>` for WebSocket channels). The
    fallback when the server is down is degradation to file
    transport, but that is silent unless the tool explicitly checks
    transport health.
  - Tool authors now have two embedding-related Protocols to learn
    (`EmbeddableTool`, plus `EmbedCapabilities`). The cheat-sheet in
    [`docs/development/embedding_a_tool.md`](../development/embedding_a_tool.md)
    is the mitigation.
- **Follow-ups:**
  - Per-tool refactors track under
    [#4998](https://github.com/D-sorganization/UpstreamDrift/issues/4998).
    Pose Studio is done; Cross-Engine Dashboard, Drake Dashboard,
    MuJoCo Dashboard, Pinocchio Dashboard, and Motion Capture
    launcher are the next batch.
  - A future ADR may revisit the WebSocket transport choice if we
    grow a cross-host fleet need (currently all tools live on one
    workstation).
  - Layout persistence (save/restore the launcher's tab/dock
    arrangement across sessions) is wired in `state_snapshot()` /
    `restore_state()` but the launcher does not yet persist it to
    disk — tracked separately.

## Validation

The cross-tool live-pose demo
([#4999](https://github.com/D-sorganization/UpstreamDrift/issues/4999))
is the integration gate for this ADR:

1. Open the launcher.
2. Right-click the Pose Studio tile → **Launch in Tab**.
3. Right-click the `pose_subscriber_demo` tile → **Launch in Dock**.
4. Drag any joint slider in Pose Studio.
5. The subscriber's FK skeleton updates within 50 ms (WebSocket
   transport, asserted by
   `tests/integration/realtime/test_pose_studio_demo.py`).

Per-piece tests:

- `tests/unit/launcher_embed/test_contract.py` — `EmbedCapabilities`
  DbC validation; `EmbeddableTool` runtime-check.
- `tests/unit/launcher_embed/test_registry.py` — register / lookup /
  unregister; `is_embeddable` semantics.
- `tests/unit/launchers/test_embedded_host.py` — open/close/focus
  semantics for tabs and docks; dirty-close prompt; idempotency on
  duplicate opens.
- `tests/unit/launchers/test_launch_routing.py` —
  `resolve_launch_mode` decision table.
- `tests/unit/realtime/` — file and WebSocket transports under their
  own pytest markers; a transport-agnostic round-trip test confirms
  the facade picks the registered transport per channel.
- `tests/docs/test_launcher_embed_doc_links.py` — every internal
  Markdown link in the new user-guide and developer-guide pages
  resolves to a real file.
