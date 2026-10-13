---
issue: 11792
summary: "Extract the exact-state feedback request calculation as an immutable stateless policy while preserving the torque controller and its allocator boundary."
dl_state: "in_progress"
next_step: "Review and integrate the stateless policy extraction, then implement an explicitly mapped muscle-allocation bridge and validate controller-off replay; native model qualification remains separate."
branch: "feat/stateless-feedback-policy-12390"
paths: "src/shared/python/motion_matching/distributed_feedback.py,tests/unit/motion_matching/test_distributed_feedback_policy.py,manuals/upstreamdrift/chapters/16-distributed-feedback.qmd,manuals/upstreamdrift/calculation-registry.json,docs/development/feedback_controls/DESIGN.md,docs/development/feedback_controls/F09R-STATELESS-FEEDBACK-POLICY-TURNOVER.md"
---
