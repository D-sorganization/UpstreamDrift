# Project Rules — UpstreamDrift (GAAI Fleet)

## Safety
1. All AI work on `staging` branch. Never commit directly to `main`.
2. PRs target `staging`. No auto-merge. Human approval required.
3. No destructive git history operations.
4. No secret commits (.env, API keys, credentials).

## Quality Gates (CI)
5. `ruff check` must pass on modified Python files before PR creation.
6. `ruff format --check` must pass (NOT black — this repo uses ruff format).
7. No new `print()` calls in `src/` (use logging).
8. File size budget: max 1200 lines per file. Exceptions in `scripts/config/file_size_budget.json`.
9. Module size budget baseline in `module_size_budget_baseline.json` for modules exceeding default limits.
10. No TODO/FIXME comments unless a tracked GitHub issue exists.
11. Use `python3` — never bare `python`.
12. Pre-push hook is broken (uses `pytest` instead of `python3 -m pytest`); use `--no-verify` for push.

## Escalation
13. If a story requires modifying CI pipelines in a breaking way — escalate.
14. If a story touches shared/core modules affecting multiple subsystems — escalate.

---

## Coding Principles (Mandatory — enforced in QA)

### TDD (Test-Driven Development)
- Write tests BEFORE implementation code.
- Every new public function/method must have at least one test.
- Test file must exist before or in the same commit as the implementation.
- If modifying existing code, add tests for the modified behavior first.

### DRY (Don't Repeat Yourself)
- No duplicated logic blocks >5 lines. Extract shared logic into helpers.
- Before writing new utility code, search for existing implementations.
- If you find yourself copying code, refactor into a shared function.

### DbC (Design by Contract)
- Public functions must validate preconditions (raise ValueError/TypeError on invalid input).
- Document postconditions in docstrings for non-trivial functions.
- Use assert statements for invariants in non-hot-path code.

### LOD (Law of Demeter)
- No method chains >2 levels (e.g., `a.b.c.d()` violates LOD).
- Functions should only call methods on: self, parameters, objects they create, direct attributes.
- If you need deep access, add a delegating method to the intermediate object.
