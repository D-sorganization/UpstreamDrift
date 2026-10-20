---
issue: 13490
summary: "Hang watchdog dumps stacks from a GIL-held Python handler so SIGUSR1 no longer segfaults the xdist controller (exit 139)"
dl_state: "in_review"
next_step: "Merge the PR; watch unit-test-gate for any further exit 139."
branch: "fix/qt-teardown-segv"
---
