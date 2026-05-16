---
title: Unreal Integration (streaming/VR) has no launcher presence
labels: feature, priority/P3
---

## Problem

The Unreal integration module (`src/unreal_integration/`) provides:

- Streaming server for real-time simulation streaming
- Unreal Engine bridge
- Meshcat and PyVista viewer backends
- VR interaction support
- Simulation streamer

This is an advanced visualization capability, but it has no manifest entry and no way for users to discover it.

## Classification

**Borderline**: Unreal integration is an advanced/power-user feature. Most users don't need it, but those who do would actively look for it. Could be a sub-feature of simulation tiles (`Launch in Unreal`) rather than a standalone tile.

## Acceptance Criteria

- [ ] Add `unreal_streaming` to MuJoCo and Drake tile `capabilities` arrays
- [ ] Consider adding an `Advanced Visualization` section in settings or engine dashboards
- [ ] Document the streaming setup in user-facing docs
- [ ] Standalone tile may not be needed; sub-feature access is sufficient
