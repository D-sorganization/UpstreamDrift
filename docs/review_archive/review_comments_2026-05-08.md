# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T10:29:35.487192

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4504: src/shared/python/motion_matching/loaders/c3d_body.py:393

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Bypass wrist impact detection when impact_source is given**

Even in the `impact_source is not None` branch, the loader still calls `_detect_impact_via_wrist`, so any explicit `marker_set` that omits `RWristTop`/`LWristTop` fails with `ValueError` before loading. This contradicts the documented behavior that `impact_source` provides the shared `time`/`impact_idx`, and it blocks valid custom marker subsets tha...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4504#discussion_r3210255790)

---

### PR #4496: tests/unit/motion_matching/test_body_skeleton.py:7

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Guard optional matplotlib import in unit test**

Importing `matplotlib` at module scope makes this test file fail during collection when the optional GUI dependency is not installed (e.g., minimal CI/dev environments), so the suite errors out instead of skipping cleanly. The repo’s root `AGENTS.md` explicitly requires matplotlib/PyQt-dependent tests to wrap imports and call `pytest.skip(...)` on `ImportError`...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4496#discussion_r3210197994)

---

### PR #4490: src/shared/python/motion_matching/loaders/matlab_dataset.py:150

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Replace params.Impact fallback for non-zero-centered time**

When `time` does not span `0`, `_stamped_impact_index` falls back to `params.Impact` as if it were a full-trajectory row index, but this commit’s own documented dataset behavior says `params.Impact` is swing-segment-relative (not a global row). That means any valid `.mat` export with a shifted timestamp baseline (all-positive/all-negative `time`) wi...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4490#discussion_r3210149886)

---

