---
issue: 11786
summary: "Add phase-gated tangent feedback with constrained actuator allocation and auditable applied-torque evidence"
dl_state: "in_review"
next_step: "Integrate native tangent/task adapters and independently replay exact applied torque under F09 gates"
branch: "feat/f02-distributed-feedback-11786"
paths: "src/shared/python/motion_matching/distributed_feedback.py,tests/unit/motion_matching/test_distributed_feedback.py,manuals/upstreamdrift/chapters/13-distributed-feedback.qmd"
---

F02 local analytic and one-joint native fixtures pass; full-body contact, six-engine parity, owner capture and muscle drive remain unqualified. See docs/development/feedback_controls/F02_TURNOVER.md.
