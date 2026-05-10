# Review Comments Archive - 2026-05-10

Generated: 2026-05-10T01:50:29.228220

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #5065: src/engines/physics_engines/drake/python/src/_embed_adapter.py:69

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Scope cleanup to the widget being closed**

`EmbeddedHostWidget` can mount the same tool in both a tab and a dock (it tracks them in separate maps), but this adapter stores all created widgets in one list and `cleanup()` clears and cleans every entry at once. When either the tab or dock is closed, host code calls `tool.cleanup()` for that single record, which will also tear down the other still-open Drake wid...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5065#discussion_r3214592589)

---

