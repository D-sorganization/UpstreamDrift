---
issue: 15539
summary: "Wait for async results load without spinning the Qt event loop (#15539)"
dl_state: "in_review"
next_step: "Review the PR; operator handles merge authorization."
branch: "codex/15539-results-wait"
---

- Poll async results loads with a bounded nested `QEventLoop` plus `QTimer.singleShot` instead of tight `processEvents` loops, keeping the GUI responsive without burning CPU while idle (#15539).
- Reject nonpositive `timeout_s` values and preserve existing finished-session and timeout behavior (#15539).
