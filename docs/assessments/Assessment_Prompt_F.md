# Assessment F: Testing Coverage & Quality

**Assessment Type**: Test Suite Audit
**Rotation Day**: Day 6 (Saturday)
**Focus**: Test coverage, test quality, regression protection

---

## Objective

Conduct a comprehensive testing audit identifying:

1. Coverage gaps (uncovered critical paths)
2. Test quality issues (brittle, slow, flaky tests)
3. Missing test categories (unit, integration, e2e)
4. Test infrastructure problems
5. Regression protection gaps

---

## Mandatory Deliverables

### 1. Testing Summary

- Total tests: X
- Coverage: X%
- Pass rate: X%
- Critical gaps identified: X

### 2. Testing Scorecard

| Category               | Score (0-10) | Weight | Evidence Required      |
| ---------------------- | ------------ | ------ | ---------------------- |
| Line Coverage          |              | 2x     | Coverage %             |
| Branch Coverage        |              | 1.5x   | Branch %               |
| Critical Path Coverage |              | 2x     | Key functions tested   |
| Test Quality           |              | 1.5x   | No flaky/brittle tests |
| Test Speed             |              | 1x     | Suite runtime          |
| Test Organization      |              | 1x     | Structure and naming   |

### 3. Coverage Gap Analysis

| Module | Lines | Covered | %   | Priority |
| ------ | ----- | ------- | --- | -------- |
|        |       |         |     |          |

### 4. Test Quality Findings

| ID    | Severity | Category | Location | Issue | Impact | Fix | Effort |
| ----- | -------- | -------- | -------- | ----- | ------ | --- | ------ |
| F-001 |          |          |          |       |        |     |        |

---

## Categories to Evaluate

### 1. Coverage Metrics

- [ ] Line coverage measured
- [ ] Branch coverage measured
- [ ] Coverage threshold enforced in CI
- [ ] Critical modules have 80%+ coverage
- [ ] Coverage trending tracked

### 2. Test Categories Present

- [ ] Unit tests (isolated, fast)
- [ ] Integration tests (component interaction)
- [ ] End-to-end tests (full workflows)
- [ ] Regression tests (bug fixes)
- [ ] Performance tests (benchmarks)
- [ ] Property-based tests (if applicable)

### 3. Test Quality

- [ ] Tests are deterministic (no flakiness)
- [ ] Tests are independent (no order dependency)
- [ ] Tests are fast (unit < 100ms each)
- [ ] Tests have clear assertions
- [ ] Tests follow AAA pattern (Arrange-Act-Assert)
- [ ] Mocks/fixtures properly scoped

### 4. Test Organization

- [ ] Tests mirror source structure
- [ ] Consistent naming convention
- [ ] pytest markers used appropriately
- [ ] Conftest.py for shared fixtures
- [ ] Tests documented where complex

### 5. CI/CD Integration

- [ ] Tests run on every PR
- [ ] Coverage reported in CI
- [ ] Test results visible in PR
- [ ] Slow tests marked and can be skipped
- [ ] Parallelization for speed

### 6. Edge Cases & Error Handling

- [ ] Boundary conditions tested
- [ ] Error paths tested
- [ ] Empty/null inputs tested
- [ ] Invalid inputs tested
- [ ] Exception handling tested

---

## Test Anti-Patterns to Flag

### Critical

- Tests passing when they should fail (false positives)
- Tests that depend on external services without mocking
- Tests that modify shared state
- Tests with no assertions

### Major

- Tests that are order-dependent
- Tests that take >5 seconds (without slow marker)
- Tests with hardcoded paths/dates
- Commented-out tests

### Minor

- Tests without docstrings
- Duplicate test logic
- Overly complex test fixtures

---

## Testing Commands

```bash
# Run tests with coverage
pytest --cov=. --cov-report=html --cov-report=term-missing

# Coverage only for specific modules
pytest --cov=module_name tests/

# Show uncovered lines
pytest --cov=. --cov-report=term-missing | grep -E "^(TOTAL|.*\.py)"

# Run only fast tests
pytest -m "not slow"

# Find slow tests
pytest --durations=20

# Check for flaky tests (run 3 times)
pytest --count=3

# Parallel execution
pytest -n auto

# Generate XML for CI
pytest --cov=. --cov-report=xml --junitxml=results.xml
```

---

## Coverage Targets

| Repository Type      | Minimum | Target | Excellent |
| -------------------- | ------- | ------ | --------- |
| Scientific Computing | 60%     | 75%    | 85%+      |
| Web Applications     | 70%     | 85%    | 90%+      |
| Libraries/Tools      | 80%     | 90%    | 95%+      |
| Games                | 40%     | 60%    | 75%+      |

---

## Critical Paths Checklist

These MUST have tests:

- [ ] Main entry points
- [ ] Public API functions
- [ ] Data validation logic
- [ ] Authentication/authorization
- [ ] Financial/scientific calculations
- [ ] Error handling paths
- [ ] Configuration loading
- [ ] Database operations

---

## Output Format

### Testing Grade

- **A (9-10)**: 90%+ coverage, fast suite, no gaps
- **B (7-8)**: 75%+ coverage, minor gaps
- **C (5-6)**: 60%+ coverage, some critical gaps
- **D (3-4)**: 40%+ coverage, major gaps
- **F (0-2)**: <40% coverage, critical paths untested

---

## Repository-Specific Focus

### For Tools Repository

- Launcher tests
- Individual tool tests
- File processing edge cases

### For Scientific Repositories

- Numerical accuracy tests
- Solver convergence tests
- Edge case handling (singularities, etc.)

### For Games Repository

- Game logic tests
- Input handling tests
- State machine tests

---

_Assessment F focuses on testing. See Assessment A-E for other quality dimensions._
