# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T16:54:38.258616

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4727: tests/integration/motion_pipeline/adversarial/test_concurrency_safety.py:83

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Fail concurrent load test when worker raises**

This test currently returns `True` in both the success and exception paths, so it passes even if every thread hits an exception during `load_any`. That masks the exact class of concurrency regressions this test is meant to catch (e.g., race-triggered parser errors), because thread-level failures are silently treated as success.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4727#discussion_r3211902234)

---

### PR #4727: tests/integration/motion_pipeline/adversarial/test_real_world_schema_drift.py:42

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Remove always-true exception assertion**

The exception-path assertion is tautological because of the trailing `or True`, so this test will pass for any exception message and cannot verify the promised “fail cleanly” behavior. If the adapter starts failing with unrelated/internal errors, this test will still report green and hide the regression.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4727#discussion_r3211902238)

---

