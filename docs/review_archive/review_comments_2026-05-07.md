# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T12:11:46.211466

## Reviewer (chatgpt-codex-connector[bot]) (4 comments)

### PR #4349: src/shared/python/motion_matching/stability_matrix.py:169

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Permit zero relative error in tolerance checks**

`is_within_tolerance` currently rejects `rel_error=0.0` by calling `require_positive`, which enforces a strictly positive value. A perfect engine match (zero error) is a valid and common outcome, so this raises a precondition failure in the success case and can break stability validation flows whenever exact equality occurs.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4349#discussion_r3204070511)

---

### PR #4349: scripts/wave2_manifest_validator.py:101

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Parse generated manifest section headers consistently**

The parser only recognizes module sections that start with `## ...`, but `generate_manifest_sections()` emits module headings as `### ...`. After `--update` writes that format, the next validation run cannot discover any module sections and reports widespread manifest mismatches, effectively making the validator self-inconsistent.

Useful? React with 👍 ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4349#discussion_r3204070517)

---

### PR #4349: src/shared/python/motion_matching/engine_init_profiler.py:455

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Honor `use_cache=False` in thermodynamic query profiling**

The thermodynamic query phase always calls `_query_thermodynamic_db`, which is globally wrapped by `lru_cache`, even when `use_cache=False`. On repeated profiles with the same engine, the query still hits cache, so the “uncached” path is not actually uncached and the reported baseline/optimization deltas become misleading.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4349#discussion_r3204070522)

---

### PR #4349: src/shared/python/qt_utils/wheel_event_filter.py:52

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Prevent unbounded growth of wheel filter cache**

The new module-level `_filter_cache` keeps strong references keyed by `id(widget)` but never removes entries when widgets are destroyed. In long-running UIs that create transient controls/dialogs, this retains one `WheelEventFilter` per historical widget and causes avoidable memory growth over time.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4349#discussion_r3204070528)

---

