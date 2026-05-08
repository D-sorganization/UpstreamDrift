# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T10:29:35.487192

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #4504: src/shared/python/motion_matching/loaders/c3d_body.py:393

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Bypass wrist impact detection when impact_source is given**

Even in the `impact_source is not None` branch, the loader still calls `_detect_impact_via_wrist`, so any explicit `marker_set` that omits `RWristTop`/`LWristTop` fails with `ValueError` before loading. This contradicts the documented behavior that `impact_source` provides the shared `time`/`impact_idx`, and it blocks valid custom marker subsets tha...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4504#discussion_r3210255790)

---

