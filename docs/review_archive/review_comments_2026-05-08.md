# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T15:49:50.717750

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4620: tests/unit/motion_pipeline/_fixtures.py:450

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve passed rig when building matching-result trajectory**

`make_motion_matching_result` accepts a `rig` but constructs `matched_trajectory` via `make_motion_trajectory(...)`, which creates a fresh default rig instead of reusing the caller’s rig. If a caller passes a non-default rig (different joint IDs/prefixes or rig ID), `metadata['residual_report']`/`torque_trajectory` are keyed to the input rig whil...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4620#discussion_r3211158550)

---

### PR #4620: tests/unit/motion_pipeline/test_lod.py:76

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Record imported submodule names in LoD import scan**

For `ImportFrom`, `_collect_imports` only records `node.module`, so statements like `from src import engines` are recorded as `src` and never match `FORBIDDEN_EXACT`/`FORBIDDEN_PREFIXES`. That creates a bypass where forbidden roots can be imported without detection, weakening the architectural guard this test is meant to enforce.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4620#discussion_r3211158553)

---

