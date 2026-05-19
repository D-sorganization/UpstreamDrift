# Review Comments Archive - 2026-04-18

Generated: 2026-04-18T15:59:18.753572

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #2772: src/launchers/upstream_drift_launcher.py:625

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Replace QRunnable signal with QObject-backed notifier**

`_schedule_cleanup` connects to `worker.finished`, but `ProcessCleanupWorker` inherits `QRunnable` rather than `QObject`, so its `pyqtSignal` is not a usable bound Qt signal. In PyQt this causes runtime failure when connecting/emitting (typically on the first cleanup timer tick), which disables cleanup and leaves `running_processes` stale while the UI k...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2772#discussion_r3105964989)

---

### PR #2772: src/api/task_manager.py:229

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Initialize `_closed` before creating semaphore**

`__del__` now assumes `_closed` always exists, but `_closed` is assigned only after `asyncio.Semaphore(...)`. If semaphore construction fails (for example `max_concurrent < 0`), object finalization hits `if not self._closed` and raises an `AttributeError` during garbage collection, obscuring the original constructor failure with an "Exception ignored in __del_...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2772#discussion_r3105964990)

---
