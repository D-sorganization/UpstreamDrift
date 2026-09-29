---
title: "CRITICAL: MyPy type-safety theater — 1,681 type: ignore suppressions across codebase"
labels: ["critical", "technical-debt", "type-safety", "mypy"]
---

## Severity: Critical

## Summary

The codebase contains **1,681 `# type: ignore` suppressions** and a `pyproject.toml` configuration that systematically silences mypy. This creates a false sense of type safety in CI and makes refactoring extremely high-risk.

## Evidence

### Configuration Erosion

```toml
[tool.mypy]
disallow_untyped_defs = false   # Allows untyped defs globally
ignore_missing_imports = true  # Silences missing stub warnings

[[tool.mypy.overrides]]
ignore_errors = true             # Entire subsystems silenced
```

Five major subsystems (`plotting.*`, `ui.*`, `spatial_algebra.*`, `mujoco_humanoid_golf.*`, `pinocchio_golf.*`) have **all errors ignored**.

### 65+ Files/Directories Excluded

The `exclude` list contains 65+ paths, including critical API routes and physics engines.

### Automated Suppression Addiction

`scripts/mypy_autofix_agent.py` adds `# type: ignore[code]` rather than fixing root causes.

## Root Cause

- `disallow_untyped_defs = false` allows new code to ship without types
- `ignore_errors = true` on whole modules hides regressions
- No CI gate prevents adding new `# type: ignore` lines

## Remediation Plan

### Phase 1: Immediate (Week 1)

- [ ] Set `disallow_untyped_defs = true` for `src/api/` (already has override)
- [ ] Remove `ignore_errors = true` from one subsystem at a time, starting with `spatial_algebra`
- [ ] Add CI check: fail build if `# type: ignore` count increases in a PR

### Phase 2: Short-term (Month 1)

- [ ] Remove `ignore_errors = true` from all override modules
- [ ] Fix or add stubs for `ignore_missing_imports` (target: `false` for internal modules)
- [ ] Remove blanket `# type: ignore` — replace with targeted `[code]` suppressions

### Phase 3: Long-term (Quarter)

- [ ] Reduce total `# type: ignore` count by 50% (target: < 800)
- [ ] Eliminate `typing.Any` usage in `src/` (currently 46)
- [ ] Replace `cast()` calls with proper type narrowing (currently 196)

## Acceptance Criteria

- [ ] `mypy --strict` passes on `src/api/` and `src/shared/python/contracts.py`
- [ ] No new blanket `# type: ignore` added without written justification
- [ ] Coverage of typed defs in `src/` > 90%
