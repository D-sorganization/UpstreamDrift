---
issue: 15150
summary: "Model Explorer teardown flushes only queued signals, so deleting an explorer no longer leaks the widgets and timers deleted after it (fixes the unit-gate shard 2 failure on main)"
---
