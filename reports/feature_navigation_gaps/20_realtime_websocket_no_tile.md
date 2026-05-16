---
title: Realtime WebSocket API has no UI presence
labels: documentation, priority/P2
---

## Problem

The realtime API (`src/api/routes/realtime.py`) provides:

- `POST /realtime/publish` -- publish real-time events
- `WebSocket /realtime/subscribe` -- subscribe to real-time event stream

This enables live simulation updates and is used by the simulation WebSocket (`simulation_ws.py`), but it has no manifest entry and no documentation in the launcher UI.

## Classification

**Internal infrastructure**: This is a transport layer, not a user-facing feature. It should be documented as a capability, not exposed as a tile.

## Acceptance Criteria

- [ ] Add `realtime` to relevant tile `capabilities` arrays
- [ ] Document the realtime API in developer docs
- [ ] No separate tile needed
