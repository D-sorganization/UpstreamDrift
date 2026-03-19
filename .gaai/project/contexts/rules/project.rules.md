# Project Rules (GAAI Fleet)

## Safety
1. All AI work on `staging` branch. Never commit directly to `main`.
2. PRs target `staging`. No auto-merge. Human approval required.
3. No destructive git history operations.
4. No secret commits (.env, API keys, credentials).

## Code Quality
5. `ruff check` must pass on modified Python files before PR creation.
6. No new `print()` calls in `src/` (use logging).
7. TDD required for new modules and functions.

## Escalation
8. If a story requires modifying CI pipelines in a breaking way — escalate.
9. If a story touches shared/core modules affecting multiple subsystems — escalate.
