---
title: Dashboard Recorder / Advanced Analysis is internal-only
labels: documentation, priority/P3
---

## Problem

The dashboard module (`src/shared/python/dashboard/`) provides:

- Recorder: playback, recording, analysis buffers
- Advanced analysis window
- Runner for launching analyses
- Dashboard-specific widgets

This is only accessible within the engine-specific dashboards (MuJoCo, Drake, Pinocchio), which themselves are not exposed from the launcher (see issue #07).

## Classification

**Internal library**: The recorder is a sub-feature of the engine dashboards. It should become accessible once the dashboards are exposed.

## Acceptance Criteria

- [ ] Expose engine dashboards (see issue #07)
- [ ] Document the recorder as a dashboard capability
- [ ] No separate tile needed
