# Review Comments Archive - 2026-04-19

Generated: 2026-04-19T11:37:23.982728

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #2773: tests/unit/conftest.py:35

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Stop faking optional deps during collection**

Injecting `pinocchio`/`casadi` into `sys.modules` in `pytest_configure` makes optional-dependency gating inaccurate across the unit suite: `engine_availability._probe_engine()` treats a `MagicMock` as valid (its `hasattr(..., "buildModelFromUrdf")` check passes), so tests marked with `@skip_if_unavailable("pinocchio")` run instead of skipping when Pinocchio is no...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2773#discussion_r3106016221)

---

### PR #2773: tests/unit/conftest.py:55

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Avoid clobbering real modules in autouse reset fixture**

The autouse reset fixture overwrites any already-loaded `pydrake`, `casadi`, and `pinocchio` entries with fresh `MagicMock` instances before every test, which can invalidate tests that are supposed to exercise real integrations when those dependencies are installed. This turns imports into fake modules suite-wide and can both hide real regressions and ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2773#discussion_r3106016224)

---

