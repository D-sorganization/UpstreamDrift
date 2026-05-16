---
title: AIP (AI Protocol) has no tile or documentation in UI
labels: feature, priority/P2
---

## Problem

The AIP module (`src/api/routes/aip.py`) provides structured AI method dispatch:

- `GET /aip/capabilities` -- list available AI methods
- `POST /aip/rpc` -- invoke AI methods via RPC
- `GET /aip/methods` -- enumerate methods

This powers the AI Sidekick but has no manifest entry. Users and developers who want to understand what AI capabilities are available have no way to discover them from the UI.

## Classification

**Borderline**: AIP is infrastructure that powers the Chat/Sidekick feature. It doesn't need its own tile, but its capabilities should be surfaced through the Chat tile's description and capabilities.

## Acceptance Criteria

- [ ] Add `ai_methods` to the chat tile `capabilities` array
- [ ] Update the chat tile description to mention AI method access
- [ ] No separate tile needed
