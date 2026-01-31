---
name: lint
description: Run linting tools (ruff, black, mypy) and fix placeholder/TODO statements
---

# Lint Skill

Run comprehensive linting and code quality checks on the codebase.

## Instructions

1. **Run ruff with auto-fix**:
   ```bash
   ruff check . --fix
   ```

2. **Run black formatter**:
   ```bash
   black .
   ```

3. **Run mypy type checking**:
   ```bash
   mypy . --ignore-missing-imports
   ```

4. **Find and fix placeholder statements**:
   - Search for `pass` statements that should have implementations
   - Search for `TODO`, `FIXME`, `XXX`, `HACK` comments
   - Search for `NotImplementedError` that should be implemented
   - Search for `...` (ellipsis) placeholders in function bodies
   
   ```bash
   grep -rn "TODO\|FIXME\|XXX\|HACK\|NotImplementedError\|pass$" --include="*.py" .
   ```

5. **Fix any issues found**:
   - For each linting error, apply the appropriate fix
   - For placeholder statements, implement the functionality or remove if unnecessary
   - Ensure all changes pass linting after fixes

6. **Verify all checks pass**:
   ```bash
   ruff check .
   black --check .
   mypy . --ignore-missing-imports
   ```

## Output

Report all issues found and fixes applied. If any issues cannot be automatically fixed, list them with recommendations.
