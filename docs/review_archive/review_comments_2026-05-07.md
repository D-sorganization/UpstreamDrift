# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T10:59:11.893538

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4339: src/shared/python/qt_utils/wheel_event_filter.py:42

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Use correct Qt wheel event enum**

The filter checks `QEvent.Type.WheelEvent`, but in PyQt6 wheel events are reported as `QEvent.Type.Wheel`; this condition will never match (or can raise at runtime depending on enum access), so wheel events are not reliably blocked and controls can still change on scroll. This breaks the core behavior the utility is meant to enforce.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4339#discussion_r3203646475)

---

### PR #4339: src/shared/python/qt_utils/wheel_event_filter.py:57

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep installed wheel filter alive**

This installs a newly created `WheelEventFilter()` without storing any persistent reference. In PyQt, event filters must stay alive on the Python side; once this function returns, the filter object can be garbage-collected and filtering silently stops. The same lifetime issue also affects `suppress_wheel_on_widgets`.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4339#discussion_r3203646479)

---

### PR #4339: src/shared/python/motion_matching/stability_matrix.py:169

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Allow zero relative error in tolerance checks**

`compute_relative_error` can validly return `0.0` for exact matches, but `is_within_tolerance` calls `require_positive`, which rejects zero and raises a precondition error. That turns the best-case input into an exception instead of a pass, so exact-stability cases can fail unexpectedly.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4339#discussion_r3203646482)

---

