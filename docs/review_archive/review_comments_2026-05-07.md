# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T14:04:11.137030

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4354: src/shared/python/qt_utils/wheel_event_filter.py:None

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep wheel filters strongly referenced**

Storing `WheelEventFilter` instances only in a `WeakValueDictionary` means there may be no strong Python reference after `suppress_wheel_on_widget(s)` returns, so the filter can be garbage-collected and wheel events will start changing control values again. This regresses the intended behavior in normal UI usage (especially after GC cycles) and should be fixed by tyin...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4354#discussion_r3204251299)

---

### PR #4354: tests/integration/test_wave2_cross_repo.py:None

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Avoid asserting MANIFEST.md that isn't in this commit**

This test now fails deterministically because it unconditionally requires a root `MANIFEST.md`, but this commit does not add that file and the repository does not contain it, so CI is blocked immediately. Either include the manifest artifact in the same change or gate/skip this assertion when the manifest workflow is not enabled.

Useful? React with 👍 /...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4354#discussion_r3204251306)

---

