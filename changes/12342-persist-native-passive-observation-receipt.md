---
issue: 12342
summary: "Persist and validate the native passive-readiness observation receipt"
dl_state: "in_review"
next_step: "Review receipt lineage and consumer validation; broader physical qualification remains open."
branch: "feat/native-passive-observation-receipt-12342"
paths: "src/engines/physics_engines/opensim/python/tour_matching/native_passive_receipt.py,src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py,src/engines/physics_engines/opensim/python/tour_matching/native_moco_runner.py,tests/opensim/test_native_passive_receipt.py,manuals/upstreamdrift/chapters/33-native-passive-readiness.qmd,manuals/upstreamdrift/calculation-registry.json,docs/development/feedback_controls/F07_PASSIVE_READINESS_TURNOVER.md"
---

Native preparation persists its already-returned passive-readiness audit atomically, with request, source, loaded-model, runtime, prepared-state, observer and optional policy identities. The solve consumer verifies the receipt and complete muscle coverage before use. Missing or stale evidence blocks the software-solve path. This receipt is not a complete restart-state artifact or a physical qualification; passive policy and all broader scientific gates remain unchanged.
