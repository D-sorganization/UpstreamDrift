---
title: Motion Matching sidebar category has zero tiles
labels: bug, ui, priority/P1
---

## Problem

The sidebar has a `Motion Matching` button, but no tile in the manifest carries the `motion_matching` category. The Motion-Match Preview tile (id: `motion_target_preview`) is categorized as `tool`, not `motion_matching`.

Meanwhile, the motion matching module (src/shared/python/motion_matching/) is extensive:

- CVAE and regressor models for surrogate training
- Multi-source target synthesis (club, body, C3D, .mat)
- Clubhead trace analysis, forward kinematics
- Performance baselines and leaderboard
- Per-step optimization with training scripts

## Classification

**Tile-worthy**: Motion matching is a major feature area. The sidebar button exists specifically for it, and the module has standalone GUI entry points.

## Acceptance Criteria

- [ ] Recategorize `motion_target_preview` to `motion_matching` category, or add a separate `motion_matching` tile
- [ ] Consider adding a `motion_matching_training` tile for the surrogate training workflow
- [ ] Verify the Motion Matching sidebar button shows at least one tile
