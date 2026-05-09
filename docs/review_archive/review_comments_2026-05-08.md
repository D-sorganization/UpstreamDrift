# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T17:54:16.933590

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #4751: src/shared/python/motion_matching/loaders/c3d.py:133

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Make marker_set_override actually select cluster handling**

When `marker_set_override` is provided, it only bypasses the new `MarkerSetMismatchError` guard and is never used to choose the cluster-processing branch, so callers following the error guidance (`marker_set_override=MarkerSet.GOLF_CLUSTER`) can still immediately fail with `ValueError` in the non-cluster path if butt/head labels are missing. This me...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4751#discussion_r3212069067)

---

