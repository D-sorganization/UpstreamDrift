---
issue: 11828
summary: "Reject truncated replay admission and require a separate observation sampling clock digest"
dl_state: "in_review"
next_step: "Integrate F09 native observation receipts with a frozen output clock and keep input time-grid identity distinct"
branch: "fix/f01b-replay-admission-11828"
paths: "src/engines/feedback_comparison.py,tests/unit/engines/test_feedback_comparison.py,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd"
---

This follow-up to merged F01 #11785 closes an admission hole: truncated within-engine and same-input receipts now fail, and observation accuracy binds its output sampling grid independently of the executed input grid. It adds no numerical or scientific qualification claim.
