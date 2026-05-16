---
title: Motion Pipeline has no dedicated tile (only partially covered by Motion Capture)
labels: feature, ui, priority/P2
---

## Problem

The Motion Pipeline (`src/shared/python/motion_pipeline/`) is a comprehensive system:

- **Orchestrator** for end-to-end motion processing
- **ID backends**: CMC, RRA, torque (MuJoCo), trajectory optimization (Drake)
- **Signal processing**: filter, resample, gap fill, normalize
- **Retargeting adapters**: C3D, BVH, TRC, OpenPose, MediaPipe, HRNet, AlphaPose, STO/MOT, CSV
- **Scaling**: anthropometric scaling, marker maps, OpenSim scale

The `motion_capture` tile only covers C3D viewing and OpenPose/MediaPipe. The full pipeline (retargeting, ID solving, signal processing, scaling) is unreachable from the UI.

## Classification

**Borderline**: The Motion Capture tile covers the capture part. The pipeline's processing capabilities (retargeting, ID, scaling) are distinct workflows that researchers need independently. Could be a separate tile or a mode within the Motion Capture page.

## Acceptance Criteria

- [ ] Decide: separate `motion_pipeline` tile or expand the Motion Capture page
- [ ] Expose retargeting, ID solving, and signal processing workflows
- [ ] Add API route coverage for pipeline operations beyond capture
