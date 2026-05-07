# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T10:57:32.944085

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4338: src/shared/python/qt_utils/wheel_event_filter.py:42

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Match wheel events using the valid Qt enum**

The filter checks `event.type()` against `QEvent.Type.WheelEvent`, but PyQt6 wheel events are identified as `QEvent.Type.Wheel` (as already used in `src/shared/python/pendulum_simulator/__main__.py`). With the current constant, the condition will never match in normal use, so wheel events pass through and controls can still change value via the mouse wheel.

Usefu...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4338#discussion_r3203579388)

---

### PR #4338: src/shared/python/qt_utils/wheel_event_filter.py:57

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve event filter lifetime in helper installation**

`suppress_wheel_on_widget` installs a freshly created `WheelEventFilter()` and immediately drops the only Python reference. In PyQt, unreferenced filter QObjects can be garbage-collected, which disables filtering after installation; callers using this convenience API may see intermittent or no suppression. Keep a persistent reference (e.g., parent it or...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4338#discussion_r3203579393)

---

