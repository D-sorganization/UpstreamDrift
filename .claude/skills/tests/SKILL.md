---
name: tests
description: Run all tests and iterate to fix failures
---

# Tests Skill

Run the complete test suite and fix any failing tests.

## Instructions

1. **Discover test configuration**:
   - Check for `pytest.ini`, `pyproject.toml`, or `setup.cfg` for test configuration
   - Check for `package.json` for JavaScript/TypeScript tests
   - Identify test directories: `tests/`, `test/`, `*_test.py`, `test_*.py`

2. **Run Python tests**:
   ```bash
   pytest -v --tb=short
   ```
   
   If pytest is not available, try:
   ```bash
   python -m pytest -v --tb=short
   ```

3. **Run JavaScript/TypeScript tests** (if applicable):
   ```bash
   npm test
   ```
   or
   ```bash
   yarn test
   ```

4. **For each failing test**:
   - Read the test file and understand what it's testing
   - Read the implementation being tested
   - Identify the root cause of the failure
   - Fix the implementation (preferred) or update the test if the test is incorrect
   - Re-run the specific test to verify the fix:
     ```bash
     pytest path/to/test_file.py::test_name -v
     ```

5. **Iterate until all tests pass**:
   - Continue fixing failures one by one
   - After each fix, run the full test suite to check for regressions
   - Document any tests that cannot be fixed and why

6. **Final verification**:
   ```bash
   pytest -v
   ```

## Output

Report:
- Total tests run
- Tests passed/failed/skipped
- Fixes applied
- Any remaining issues that need manual attention
