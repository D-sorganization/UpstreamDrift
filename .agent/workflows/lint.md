---
description: Run linting tools (ruff, black, mypy) and fix placeholder/TODO statements
---

# Lint Workflow

Run comprehensive linting and code quality checks on the codebase.

## Steps

// turbo

1. **Run ruff with auto-fix**:

   ```bash
   ruff check . --fix
   ```

// turbo
2. **Run black formatter**:

   ```bash
   black .
   ```

// turbo
3. **Run mypy type checking**:

   ```bash
   mypy . --ignore-missing-imports
   ```

1. **Find placeholder statements** (review manually):

   ```bash
   grep -rn "TODO\|FIXME\|XXX\|HACK\|NotImplementedError\|pass$" --include="*.py" .
   ```

// turbo
5. **Verify all checks pass**:

   ```bash
   ruff check .
   black --check .
   ```

## Output

Report all issues found and fixes applied. If any issues cannot be automatically fixed, list them with recommendations.
