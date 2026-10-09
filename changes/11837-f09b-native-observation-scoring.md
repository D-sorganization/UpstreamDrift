---
issue: 11837
summary: "Score native output against frozen observations with full replay identity"
dl_state: "in_review"
next_step: "Integrate per-engine native outputs and qualification evidence without relaxing gates."
branch: "feat/11837-f09b-native-observation-qualification"
paths: "src/engines/feedback_observation_qualification.py,tests/unit/engines/test_feedback_observation_qualification.py,tests/motion_capture/rig/test_tools_session_export.py,docs/development/feedback_controls/IMPLEMENTATION_PLAN.md,docs/development/feedback_controls/F09B-OBSERVATION-QUALIFICATION.md,docs/development/feedback_controls/TURNOVER.md,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd,vendor/ud-tools"
---

# F09b Native Observation Scoring

The workflow consumes F01 comparison rows, the actual Tools T01 replay bundle,
native marker positions, and exact observed positions. It binds the complete
initial-state payload hash and reuses the existing F09a sampling, metric, and
acceptance contracts. It preserves the six-engine denominator and never
promotes an unqualified registry row. Tests use synthetic fixtures only; no
native, private-data, physiological, or cross-engine acceptance claim follows.
Importing the scorer leaves capture-rig schema availability unchanged; the
Tools namespace seam is extended only when a bundle is validated.
