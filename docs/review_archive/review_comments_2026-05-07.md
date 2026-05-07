# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T09:33:16.464445

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4329: src/shared/python/upstream_drift/logging_config.py:48

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Close old handlers before clearing root logger**

Calling `root.handlers.clear()` removes handler references but does not close them, so repeated calls to `setup_logging()` can leak file/socket descriptors from previously attached handlers (especially in long-running processes, tests, or apps that reconfigure logging). This can eventually exhaust descriptors or keep stale file handles open; use explicit `hand...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4329#discussion_r3203074397)

---

### PR #4329: src/shared/python/upstream_drift/logging_config.py:15

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Derive JSON timestamp from LogRecord creation time**

Using `datetime.now(...)` in `format()` timestamps the moment the record is formatted, not when the log event occurred. In queued or deferred logging setups, this can shift timestamps by seconds/minutes and reorder events incorrectly in downstream analysis. The formatter should use `record.created` (e.g., `datetime.fromtimestamp(record.created, timezone.ut...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4329#discussion_r3203074402)

---

