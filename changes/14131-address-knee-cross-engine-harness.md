---
issue: 14131
summary: "Live cross-engine address knee harness: same spec and q through each installed engine's FK vs MuJoCo at 0.1 deg."
---

`test_address_knee_flexion.py` gains `_cross_engine_knee` (test-only) and
`test_address_knee_flexion_matches_across_engines`: the anthro-driver seeds spec
and frame 0 of its committed tracking trajectory go through each engine's own FK
adapter, `address_knee.model_knee_flexion_deg` measures both knees, and every
available engine must match MuJoCo within `address_knee_engine_check.TOLERANCE_DEG`.
Engines without an SDK or a spec-driven FK adapter (Pinocchio has no
`body_poses`, OpenSim, MyoSuite) are reported unavailable with a reason, never as
agreement. Unit fixtures cover agree, disagree, unavailable, missing reference and
fail-closed errors. Refs #12684, #12692; no receipts, thresholds or production code change.
