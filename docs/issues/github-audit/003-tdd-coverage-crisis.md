---
title: "CRITICAL: TDD coverage crisis — 45% threshold, 512 skipped tests, 2,115 mocks"
labels: ["critical", "testing", "coverage", "tdd"]
---

## Severity: Critical

## Summary

The test suite is in a state of collapse: **45% coverage threshold**, **512 skipped/xfailed tests**, **2,115 mock objects**, and **12 vacuous `assert True` tests**. The coverage configuration actively hides untested code.

## Evidence

### Coverage Gate Erosion

```toml
[tool.coverage.report]
fail_under = 45  # CI-enforced minimum
exclude_lines = [
    "pragma: no cover",
    "pass",
    "except ImportError",
    "if TYPE_CHECKING:",
]
```

Excluding `pass` and `except ImportError` inflates apparent coverage artificially.

### Vacuous Tests

```python
# tests/unit/engines/drake/test_example.py
def test_main_execution():
    assert True  # No actual validation
```

### Over-Mocking

| Location | Mock Count | Assessment                                     |
| -------- | ---------- | ---------------------------------------------- |
| `tests/` | 2,115      | Extreme — tests mock rather than exercise code |
| `src/`   | 59         | Mocks in production code                       |

### Skipped Tests

```python
# 512 instances of pytest.skip / @pytest.mark.skip / @pytest.mark.xfail
```

More than half the test suite is bypassed.

## Root Cause

- 45% threshold signals "testing is optional"
- Heavy mocking hides integration failures
- Tests embedded in `src/` (`src/**/tests/`) blur source/test boundaries

## Remediation Plan

### Phase 1: Immediate (Week 1)

- [ ] Ban `assert True` — replace with meaningful assertions or delete
- [ ] Remove `pass` and `except ImportError` from `exclude_lines`
- [ ] Audit 512 skipped tests: delete permanently broken ones, fix flakies

### Phase 2: Short-term (Month 1)

- [ ] Raise `fail_under` from 45% → 55%
- [ ] Move all `src/**/tests/` to top-level `tests/`
- [ ] Cap mock count per test file (e.g., max 20 MagicMock instances)

### Phase 3: Long-term (Quarter)

- [ ] Raise `fail_under` to 75%
- [ ] Zero `assert True` tests
- [ ] Integration test ratio: at least 20% of total tests

## Acceptance Criteria

- [ ] `fail_under >= 55%` in CI
- [ ] Zero `assert True` in `tests/`
- [ ] Skipped test count < 100
