---
title: Chat/AI Sidekick panel has no manifest entry
labels: feature, ui, priority/P2
---

## Problem

The PyQt6 launcher has an AI Sidekick panel (`src/shared/python/upstream_drift_tools/ui/tools_sidebar/`) with a full sidebar, calculator assist, command history, workspace persistence, and data explorer integration. The Tauri UI has a `/chat` route.

Neither appears in the launcher manifest. The AI panel is activated by a sidebar button in the PyQt6 launcher but has no corresponding tile. The Tauri manifest does not know it exists.

## Classification

**Tile-worthy**: The AI Sidekick is a major feature that users would look for. It should appear in the manifest so both launchers can render it consistently.

## Acceptance Criteria

- [ ] Add a `chat` tile to `launcher_manifest.json` with `web_route: /chat`
- [ ] Assign to the `tool` category
- [ ] Ensure the PyQt6 launcher shows it as a tile in addition to the sidebar button
- [ ] Verify Tauri dashboard renders the chat tile
