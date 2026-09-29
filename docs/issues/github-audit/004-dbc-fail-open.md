---
title: "MODERATE: DbC precondition evaluator fails open — returns True on eval failure"
labels: ["moderate", "dbc", "contracts", "reliability"]
---

## Severity: Moderate

## Summary

The `contracts.py` precondition evaluator silently returns `True` when it cannot evaluate a condition, defeating the purpose of Design by Contract. A broken lambda (wrong parameter names, type mismatch) is treated as passing.

## Evidence

### Silent Bypass on Evaluation Failure

```python
# src/shared/python/contracts.py:252-261
def _evaluate_precondition(...):
    try:
        # ... name-based binding ...
    except (TypeError, ValueError):
        pass  # ← Silently swallows binding failures
    try:
        return bool(condition(*args, **kwargs))
    except TypeError:
        pass
    return True  # ← Returns True when precondition cannot be evaluated!
```

### Environment-Controllable Enforcement

```python
DBC_LEVEL: ContractLevel = _resolve_contract_level()  # Reads from env var
```

Production can silently disable all contracts via `DBC_LEVEL=off`.

## Root Cause

- Overly defensive evaluator prioritizes "don't break" over "enforce contracts"
- No logging when evaluation fails
- No distinction between "condition is true" and "condition could not be checked"

## Remediation Plan

### Phase 1: Immediate (Week 1)

- [ ] Change `_evaluate_precondition` to raise `ContractEvaluationError` on failure (fail closed)
- [ ] Add `logger.error()` before raising to aid debugging
- [ ] Add test: broken lambda raises instead of returning True

### Phase 2: Short-term (Month 1)

- [ ] Audit all `@precondition`/`@postcondition` decorators for parameter name mismatches
- [ ] Add `strict=True` mode that ignores `DBC_LEVEL=off`
- [ ] Document contract debugging in `CONTRIBUTING.md`

## Acceptance Criteria

- [ ] `_evaluate_precondition` raises on `TypeError`/`ValueError`
- [ ] All existing preconditions still pass after parameter name audit
- [ ] Test coverage for `contracts.py` > 95%
