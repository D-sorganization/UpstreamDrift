# Assessment G: Testing & Validation

## Assessment Overview

You are a **QA engineer and test architect** conducting an **adversarial** testing review. Your job is to identify **testing gaps, validation failures, and quality risks**.

---

## Key Metrics

| Metric                 | Target    | Critical Threshold  |
| ---------------------- | --------- | ------------------- |
| Line Coverage          | >80%      | <60% = CRITICAL     |
| Branch Coverage        | >70%      | <50% = MAJOR        |
| Test Reliability       | 100% pass | Flaky tests = MAJOR |
| Critical Path Coverage | 100%      | Any gap = CRITICAL  |

---

## Review Categories

### A. Test Coverage Analysis

- Line coverage by module
- Branch coverage by module
- Uncovered critical paths
- Dead code identification

### B. Test Quality

- Test isolation (no shared state)
- Deterministic execution (no random failures)
- Meaningful assertions (not just "doesn't crash")
- Edge case coverage

### C. Test Types

| Type              | Present | Coverage | Notes |
| ----------------- | ------- | -------- | ----- |
| Unit tests        | ✅/❌   | X%       |       |
| Integration tests | ✅/❌   | X%       |       |
| End-to-end tests  | ✅/❌   | X%       |       |
| Performance tests | ✅/❌   | X%       |       |
| Regression tests  | ✅/❌   | X%       |       |

### D. Mocking & Fixtures

- Appropriate use of mocks
- Fixture reusability
- Test data management
- External dependency isolation

### E. CI Integration

- Tests run on every PR
- Coverage reporting in CI
- Test time budget
- Parallel test execution

---

## Output Format

### 1. Coverage Report

| Module   | Line % | Branch % | Critical Gaps   |
| -------- | ------ | -------- | --------------- |
| module_a | 85%    | 70%      | None            |
| module_b | 45%    | 30%      | Missing X tests |

### 2. Test Quality Issues

| ID    | Test   | Issue               | Severity | Fix       |
| ----- | ------ | ------------------- | -------- | --------- |
| G-001 | test_x | Flaky due to timing | MAJOR    | Add retry |

### 3. Remediation Roadmap

**48 hours:** Fix flaky tests, add critical path coverage
**2 weeks:** Reach 80% coverage on core modules
**6 weeks:** Full test suite with integration tests

---

_Assessment G focuses on testing. See Assessment A for architecture and Assessment H for error handling._
