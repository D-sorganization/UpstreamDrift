---
issue: 11793
summary: "Sample native marker positions on the exact observation clock"
dl_state: "in_review"
next_step: "Integrate sampled native outputs with F09 comparison levels without changing frozen acceptance criteria."
branch: "feat/11793-f09a-output-sampling"
paths: "src/shared/python/motion_matching/replay_metrics.py,src/shared/python/motion_matching/__init__.py,tests/unit/motion_matching/test_replay_observation_sampling.py,docs/development/feedback_controls/F09A-OBSERVATION-SAMPLING.md"
---

Adds a positions-only interpolation boundary that maps native output samples
to the exact observation clock, records independent time-grid and payload
hashes, rejects extrapolation and identity mismatches, and delegates metrics to
the existing replay scorer. Twelve synthetic tests pass. This is measurement
alignment only: no native execution, capture acceptance, new tolerance, or
qualification is claimed; F09 remains open.
