---
issue: 11792
summary: "Persist and validate the native passive-readiness observation receipt"
dl_state: "in_progress"
next_step: "Complete source-bound readiness review and full-clock replay; muscle-policy evidence and physical qualification remain pending."
owner: "codex"
branch: "feat/native-passive-observation-receipt-12342"
paths: "src/engines/physics_engines/opensim/python/tour_matching/native_passive_receipt.py,src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py,src/engines/physics_engines/opensim/python/tour_matching/native_moco_runner.py,tests/opensim/test_native_passive_receipt.py,manuals/upstreamdrift/chapters/33-native-passive-readiness.qmd,manuals/upstreamdrift/calculation-registry.json,docs/development/feedback_controls/F07_PASSIVE_READINESS_TURNOVER.md"
---

This change records implementation child #12342 in PR #12355. The passive-readiness receipt stores an observation; it is not a complete restart artifact. No solve, replay, native qualification, or physical qualification is established, and all 17 registered model/variant rows remain unqualified.
