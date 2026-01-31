---
description: Run all tests and iterate to fix failures
---

# Tests Workflow

Run the complete test suite and fix any failing tests.

## Steps

1. **Discover test configuration**:
   - Check for `pytest.ini`, `pyproject.toml`, or `setup.cfg` for test configuration
   - Identify test directories: `tests/`, `test/`, `*_test.py`, `test_*.py`

// turbo
2. **Run Python tests**:

   ```bash
   pytest -v --tb=short
   ```

1. **For each failing test**:
   - Read the test file and understand what it's testing
   - Read the implementation being tested
   - Identify the root cause of the failure
   - Fix the implementation (preferred) or update the test if incorrect
   - Re-run the specific test to verify:

     ```bash
     pytest path/to/test_file.py::test_name -v
     ```

2. **Iterate until all tests pass**:
   - Continue fixing failures one by one
   - After each fix, run the full test suite to check for regressions

// turbo
5. **Final verification**:

   ```bash
   pytest -v
   ```

## Output

Report:

- Total tests run
- Tests passed/failed/skipped
- Fixes applied
- Any remaining issues that need manual attention
