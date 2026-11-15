---
issue: 15015
summary: "Pelvis yaw error pct uses a fixed 90° reference; G1 5% gate is 4.5°."
branch: "cursor/pelvis-yaw-invariant-pct-02e4"
---

Tolerance-Change-Evidence: #15015 — G1 `pelvis_yaw_error_pct` still thresholds at 5%, but
the percentage now means `|Δyaw°| / 90° × 100` instead of `|Δyaw°| / max(|yaw_target°|, 1°) × 100`.
At run-102 uninterrupted `pelvis_yaw_diff_deg` 0.325°, the reported pct drops from the archived
0.596% to 0.361%; the 5% pass boundary is 4.5° angular error (was ~2.73° at that heading under
the old denominator). Archived native evidence JSON is unchanged; harness tests recompute pct.

`compute_replay_five_metrics` reports NaN (not 0%) when waist markers are missing or invalid on
the terminal frame; `compute_pelvis_yaw_metrics` still returns `valid=False` with a 100% sentinel.