# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T12:03:21.196411

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4343: src/shared/python/qt_utils/wheel_event_filter.py:42

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Match wheel events with the correct Qt enum**

Use `QEvent.Type.Wheel` here instead of `QEvent.Type.WheelEvent`; the latter does not match the wheel-event type used elsewhere in this repo and can prevent the filter from ever triggering (or raise at runtime depending on Qt bindings). In that case, mouse-wheel input will continue changing combo/slider values, so the fix in this commit does not actually enforce ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4343#discussion_r3204011386)

---

### PR #4343: tests/integration/test_wave2_cross_repo.py:33

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Stop importing non-existent contracts helpers**

These tests import `precondition_check`/`postcondition_check` from `src.shared.python.contracts`, but that module does not define those names, so this test fails immediately with `ImportError` instead of validating integration behavior. As written, `pytest tests/integration/test_wave2_cross_repo.py` is broken on a clean checkout.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4343#discussion_r3204011392)

---

### PR #4343: src/shared/python/qt_utils/wheel_event_filter.py:68

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Keep installed wheel-event filters alive**

The shared filter is created as a local variable and never retained on a long-lived object, so it can be garbage-collected after `suppress_wheel_on_widgets` returns. When that happens, widgets lose the event filter and wheel-based value changes resume, making the behavior flaky/non-deterministic across sessions and GC timing.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4343#discussion_r3204011397)

---

