---
issue: 14178
summary: "CI apt installer bounds each apt-get attempt (APT_ATTEMPT_TIMEOUT, default 420s) and sets 30s acquire timeouts, so a stalled mirror download is retried instead of holding a merge-queue job to its 150-minute timeout"
---
