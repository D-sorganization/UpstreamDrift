# Real-Time IPC — `subscribe` / `publish` for embedded tools

For tool authors. This page documents
`src/shared/python/realtime/`, the pub-sub facade that lets two
tools talk to each other live — Pose Studio publishing a canonical
pose at 30 Hz, a downstream subscriber re-rendering it as an FK
skeleton, and so on.

> **Background.** Design rationale lives in
> [ADR-0013](../adr/0013-launcher-composability.md). For the
> embedding contract that the publisher and subscriber both
> implement, see
> [`embedding_a_tool.md`](embedding_a_tool.md).

---

## A. The API

Two functions. That's the whole surface area:

```python
from src.shared.python.realtime import publish, subscribe

# Publish a payload on a channel.
publish("pose/canonical", {"version": "canonical-v1", "joints": {...}})

# Subscribe to a channel; the callback runs each time someone publishes.
def _on_pose(payload: dict) -> None:
    ...

unsubscribe = subscribe("pose/canonical", _on_pose)

# Later, when the tool is being torn down:
unsubscribe()
```

`subscribe()` returns a callable; invoking it removes the
subscription. Subscribers should call that from their `cleanup()`
hook (see
[`embedding_a_tool.md`](embedding_a_tool.md#d-cleanup-contract--what-to-release))
so a closed dock doesn't keep firing callbacks on a destroyed
widget.

`publish()` is fire-and-forget — it does not block on subscribers.
The transport layer handles fan-out asynchronously.

Payloads should be **JSON-serialisable** dicts. The file transport
serialises with `json.dumps`; the WebSocket transport serialises
with the same FastAPI JSON encoder used for REST endpoints.

---

## B. Channel naming convention

Channels use a `<scope>/<topic>` shape:

| Scope     | Examples                                          | Notes                                                                                   |
| --------- | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `pose/`   | `pose/canonical`                                  | Canonical pose interchange (see [ADR-0012](../adr/0012-canonical-pose-interchange.md)). |
| `engine/` | `engine/mujoco/state`, `engine/drake/qpos`        | Per-engine live state.                                                                  |
| `tool/`   | `tool/pose_studio/dirty`, `tool/<tool_id>/status` | Tool-internal lifecycle events.                                                         |
| `status/` | `status/launcher`                                 | Cross-process health and progress.                                                      |
| `mocap/`  | `mocap/preview`                                   | Live mocap loaders.                                                                     |

Rules:

- **Lowercase, slash-separated.** No spaces, no dots, no leading
  slash.
- **Stable.** A published channel is part of the tool's public API
  surface. Renaming one is a breaking change for subscribers.
- **Unique within a scope.** Two tools should not publish to the
  same channel unless they're publishing the same shape of payload
  (the payload schema is the contract; the channel name is just the
  routing key).
- **Document new channels** in the registry (`channels.py`) so the
  routing decision is reviewable in code review, not buried in a
  call site.

---

## C. File vs. WebSocket transport — the registry picks

Channel-to-transport routing lives in
`src/shared/python/realtime/channels.py`. You don't pass a
transport name to `publish()` or `subscribe()`; the registry picks
based on the channel.

The registry's defaults:

| Channel pattern        | Transport     | Why                                                                         |
| ---------------------- | ------------- | --------------------------------------------------------------------------- |
| `pose/*`, `engine/*/*` | **WebSocket** | High-frequency (30 Hz +), latency budget < 50 ms.                           |
| `tool/*/*`, `status/*` | **File**      | Low-frequency, durability-friendly, cross-process visible without a server. |
| Unregistered           | **File**      | Conservative default — works without a FastAPI server running.              |

**File transport** writes JSONL append under
`~/.upstream_drift/realtime/<channel-with-slashes-replaced>.jsonl`.
Subscribers tail the file via OS-level file watchers (or polling on
platforms where watchers aren't reliable). Latency: 50-200 ms on
NTFS; 10-50 ms on ext4 / APFS.

**WebSocket transport** multiplexes through the existing FastAPI
server (`src/api/`). Each channel is one WS message type; fan-out
is sub-millisecond within the same host. Requires the FastAPI
server to be running (the launcher starts it automatically when
embedded view is in use).

To force a transport for a one-off case, pass `transport="file"` or
`transport="websocket"` to `publish()` / `subscribe()` — but prefer
to add the channel to the registry instead. Hard-coding transport
at the call site means a future operations change (move WS server
off-host, e.g.) requires touching every call site.

---

## D. Latency budgets

| Transport | p50 latency | p99 latency | Notes                                |
| --------- | ----------- | ----------- | ------------------------------------ |
| WebSocket | 2-5 ms      | 20-30 ms    | Localhost loopback; same host.       |
| File      | 50-100 ms   | 200-500 ms  | NTFS worst case; ext4 / APFS faster. |

These are end-to-end measurements (publisher's `publish()` call
returning to subscriber's callback being invoked) under the
integration test `tests/integration/realtime/`. Beyond p99 you're
into FastAPI server pressure (WS) or filesystem-watcher backoff
(file) — usually a sign the channel is being misused.

If your tool needs anything below 2 ms p50 you're outside the
intended use case for `realtime`. Direct Qt signal/slot inside one
process is the right answer; the IPC layer is for cross-tool, not
inside-tool.

---

## E. Debugging

### File-transport channels

Inspect `~/.upstream_drift/realtime/`:

```bash
# List active channel files
ls ~/.upstream_drift/realtime/

# Tail one in real time
tail -F ~/.upstream_drift/realtime/tool_pose_studio_dirty.jsonl
```

Each line is one published payload as JSON. Channel names map to
filenames by replacing `/` with `_`. The directory is created on
first publish; it's safe to delete files between sessions to clear
backlog (subscribers re-subscribe on next publish).

### WebSocket-transport channels

The FastAPI server logs every subscribe / publish at `INFO` level
when started with `--log-level info`. Look for entries like:

```
INFO     realtime: SUB pose/canonical client=127.0.0.1:54221
INFO     realtime: PUB pose/canonical 480 bytes -> 2 subscribers
```

For deeper inspection, the server exposes
`GET /realtime/_debug/channels` (only when started with the
`--debug-channels` flag) which returns the live registry plus
per-channel subscriber counts.

### Tracing in the publisher

If you suspect a payload isn't getting through, wrap the publish
in a debug log:

```python
import logging

from src.shared.python.realtime import publish

log = logging.getLogger(__name__)

payload = pose.to_dict()
log.debug("publishing pose/canonical: %d bytes", len(json.dumps(payload)))
publish("pose/canonical", payload)
```

`get_logger()` from
`src/shared/python/logging_pkg/logging_config.py` is the canonical
factory — it routes through the JSON-config + rotation pipeline
that the launcher uses elsewhere.

---

## F. Worked example — Pose Studio publishing canonical pose

Pose Studio's `EngineController` publishes a canonical pose every
time a joint slider moves. The hook lives behind a feature flag
(`POSE_STUDIO_PUBLISH_REALTIME=1`) so unit tests don't have to
subscribe.

```python
# src/tools/pose_studio/controllers/engine_controller.py
"""Pose Studio engine controller — drives engines from canonical pose."""

from __future__ import annotations

import os
import time
from typing import Any

from src.shared.python.realtime import publish

_PUBLISH_FLAG = "POSE_STUDIO_PUBLISH_REALTIME"
_DEBOUNCE_S = 1.0 / 30.0  # 30 Hz cap


class EngineController:
    def __init__(self) -> None:
        self._last_publish_t = 0.0

    def set_canonical_pose(self, pose: Any) -> None:
        # ... drive engines ...
        self._maybe_publish(pose)

    def _maybe_publish(self, pose: Any) -> None:
        if os.environ.get(_PUBLISH_FLAG) != "1":
            return
        now = time.monotonic()
        if now - self._last_publish_t < _DEBOUNCE_S:
            return
        self._last_publish_t = now
        publish("pose/canonical", pose.to_dict())
```

The 30 Hz debounce matters — without it, a slider drag emits one
publish per Qt mouseMove (often 100+ Hz on a high-refresh trackpad)
and the WebSocket layer buffers them faster than subscribers can
drain.

The matching subscriber (the cross-tool demo
`pose_subscriber_demo`) is straightforward:

```python
# src/tools/pose_subscriber_demo/gui.py (excerpt)
from src.shared.python.realtime import subscribe

class PoseSubscriberWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._unsubscribe = subscribe("pose/canonical", self._on_pose)
        # ... build UI ...

    def _on_pose(self, payload: dict) -> None:
        # NB: callback runs on the transport's thread. Marshal back
        # to the GUI thread before touching widgets:
        QMetaObject.invokeMethod(
            self,
            lambda: self._render_pose(payload),
            Qt.ConnectionType.QueuedConnection,
        )

    def cleanup(self) -> None:
        self._unsubscribe()
```

The `QMetaObject.invokeMethod` marshalling is required for both
transports — file and WebSocket callbacks both fire on a worker
thread, not the GUI thread, so direct widget mutation from inside
the callback will crash on Linux and corrupt state on Windows.

---

## See also

- [ADR-0013](../adr/0013-launcher-composability.md) — design
  rationale for the IPC layer and why we picked file + WebSocket
  rather than DBus / LocalSocket / REST-only.
- [`embedding_a_tool.md`](embedding_a_tool.md) — the embedding
  contract that the publisher and subscriber both live behind.
- [`docs/user_guide/launcher/embedded_view.md`](../user_guide/launcher/embedded_view.md)
  — the user-facing experience this IPC enables.
- [ADR-0003](../adr/0003-websocket-realtime-simulation.md) — the
  original WebSocket-for-real-time decision that the WS transport
  builds on.
