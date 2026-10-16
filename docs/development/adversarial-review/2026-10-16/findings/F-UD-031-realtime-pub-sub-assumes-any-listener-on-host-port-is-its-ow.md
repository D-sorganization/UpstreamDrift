# F-UD-031

**Realtime pub-sub assumes any listener on `host:port` is its own server**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| ⬜ Low   | correctness | high       | S      | proposed |

**Locations:** `src/shared/python/realtime/ws_pubsub.py:103`, `src/shared/python/realtime/ws_pubsub.py:197`, `src/shared/python/realtime/ws_pubsub.py:202`

## Finding

- Severity: Low - Category: correctness
- Confidence: high
- Location: src/shared/python/realtime/ws_pubsub.py:103-110; src/shared/python/realtime/ws_pubsub.py:197-198;
  src/shared/python/realtime/ws_pubsub.py:202-225
- Evidence: `_port_in_use` does a TCP connect; `start()` skips spawning when it succeeds. Rust-server failures fall through
  `except Exception:  # noqa: BLE001` and `except Exception:` (the latter unflagged because of F-UD-005).
- Failure scenario: a stale server from a previous repo version (different protocol) or an unrelated local service on 8765
  makes every publish a silent 404; two launcher instances with different `GOLF_REALTIME_PORT` defaults never notice.
- Why a newer engineer misses this: "port busy ⇒ already running" is the common idiom; it is only wrong when it matters.
- Proposed change: probe `GET /realtime/health` and compare a protocol version string before adopting a foreign server.
- Verification: start a dummy HTTP server on the port; `WSPubSub.start()` must raise or spawn on another port.
- Effort: S

## Draft Pull Request

**Title:** `fix(realtime): verify the listener on the pub-sub port is our server`

### Why This Matters (The Pitch)

WSPubSub.start() skips spawning its server when a TCP connect to host:port succeeds. A stale server from an older protocol or an unrelated local service on 8765 then makes every publish a silent 404, and two launcher instances with different default ports never notice. Probing GET /realtime/health and comparing a protocol version before adopting a foreign server is a size-S fix.

### Proposed PR Body

## Problem

`src/shared/python/realtime/ws_pubsub.py:103-110` (`_port_in_use` TCP connect), `:197-198` and `:202-225` (Rust-server failures fall through `except Exception:  # noqa: BLE001` and a second unflagged `except Exception:`).

## Change

- [ ] Probe `GET /realtime/health` and compare a protocol version string before adopting an existing server

## Verification

Start a dummy HTTP server on the port; `WSPubSub.start()` must raise or spawn on another port.

## Risk and Rollback

Low: only changes behaviour when the port is already occupied. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-031

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
