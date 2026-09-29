# ADR-0016: Error-handling discipline and the ratchet pattern

- Status: Accepted
- Date: 2026-05-21
- Decision Makers: @D-sorganization/maintainers
- Related Issues/PRs: closes #5911 (adversarial-review epic #5907)

## Context

The 2026-05-21 adversarial A-N review graded UpstreamDrift's error-handling
**4 / 10** — the lowest UpstreamDrift score after Configuration. Concrete
findings:

- `pyproject.toml [tool.ruff.lint] extend-ignore` globally suppressed
  `BLE001` (blind-except), `F841` (unused-variable), `F401` (unused-import).
  The lint config tolerated exactly the patterns hurting the codebase most.
- 20+ bare `except Exception:` clauses, frequently `# pragma: no cover -
defensive`, swallowing failures silently.
- 30+ `raise RuntimeError("short message")` without context or `from exc`.
- `asyncio.gather(*tasks)` without `return_exceptions=True` — one failing
  task crashes the whole batch with no partial-result handling.
- `subprocess.Popen(...)` standalone calls with no `.wait()` / `.terminate()`
  → zombie-process accumulation.
- Files opened without `with` → file-handle leaks when iterators
  short-circuit.

A pre-existing exception hierarchy at `src/shared/python/core/error_utils.py`
(`GolfSuiteError` + 22 subclasses) already covers domain errors, and
`src/shared/python/core/error_decorators.py` provides `log_errors`,
`retry_on_error`, `ErrorContext`, `validate_args`, `safe_import`. The gap
was **not** missing primitives; it was (a) the lint config allowed the
violations and (b) the codebase had no zero-allocation primitives for the
three remaining patterns (subprocess cleanup, async gather, narrow
catching).

## Decision

1. **Tighten the lint config.** Remove `BLE001`, `F841`, `F401` from
   `extend-ignore` in `pyproject.toml`. Pre-existing violations are
   grandfathered via file-local `# noqa: <code>` comments (added in one
   commit via `ruff --add-noqa`) so that _new_ code triggers the rule.

2. **Add three resource-safety primitives** in a new module
   `src/shared/python/core/process_safety.py`:

   - `managed_popen(args, *, timeout, kill_timeout, **popen_kwargs)` —
     context manager that escalates `terminate()` → `kill()` and always
     reaps the process. Rejects `shell=True` and string-form args at
     construction (DbC precondition).
   - `safe_gather(*coros, raise_on_all_failed, log_partial)` — wrapper
     around `asyncio.gather` that defaults to `return_exceptions=True`,
     logs partial failures, and only raises (`AllTasksFailedError`) when
     opted in and **every** task failed.
   - `narrow_catch(*types, log_message)` — context manager that catches
     only the listed exception types. Rejects bare `Exception` at
     construction. Uses `logger.exception(...)` so tracebacks are
     preserved.

   All three validate inputs at the boundary (DbC), are self-contained
   (LOD: they wrap, return, re-raise — they never reach into caller
   internals), and reuse the existing `logging` and exception hierarchy
   (DRY).

3. **Add a ratcheting CI check.**
   `scripts/ci/check_error_handling_ratchet.py` counts five anti-pattern
   classes in `src/` and compares against
   `scripts/config/error_handling_baseline.json`. Counts may **decrease**
   freely; any **increase** fails the PR. Wired into `ci-standard.yml`
   after the existing file-size-budget check.

4. **Document the policy in CLAUDE.md** so new contributors learn the
   migration path before they hit a CI failure.

## Alternatives Considered

1. **Big-bang migration of all 120+ BLE001 sites in one PR.** Rejected:
   unbounded scope, would conflict with every in-flight PR, no rollback
   plan if one of the migrations breaks runtime behaviour.
2. **Leave the lint suppression in place and rely on code review.** The
   adversarial review demonstrated this hasn't worked. Reviewers don't
   block on patterns that pass CI.
3. **Use `flake8-bugbear` instead of `ruff`'s built-in BLE.** Adds a
   dependency for no incremental benefit; ruff already implements BLE001.
4. **Force every caller into the new helpers immediately.** Some
   long-lived process spawns (Docker tile launchers, persistent
   simulators) intentionally outlive the function call and don't fit the
   `managed_popen` lifetime model. The ratchet permits the existing
   sites and only blocks regressions.

## Consequences

- **Positive:**

  - The lint rule is enforced for all new code from this PR forward.
  - Three named primitives make the right pattern the easy pattern.
  - The ratchet provides quantitative evidence of improvement on every
    PR ("BLE001 down from 227 to 214 since this change merged").
  - DBC checks on the new helpers loud-fail on misuse rather than
    silently degrading.

- **Negative:**

  - Grandfathered `# noqa` comments add line noise. They're explicit and
    greppable (`# noqa: BLE001`), but they are still added everywhere.
  - The ratchet is text-based, not AST-based, so adversarial code (e.g.
    `subprocess . Popen(`) could evade it. We accept this — the goal is
    to make accidents costly, not adversaries impossible.

- **Follow-ups:**
  - Migrate the AI peer-review coordinator (`src/shared/python/ai/peer_review/coordinator.py:191`)
    to `safe_gather` — pending product decision on partial-failure
    semantics.
  - Migrate the legacy `subprocess.Popen` sites in `src/launchers/docker_manager.py`
    where the lifetime model permits.
  - Audit existing `raise RuntimeError("...")` sites and replace with
    domain-specific subclasses from `core.error_utils`.

## Validation

- `tests/unit/core/test_process_safety.py` — 21 tests covering the three
  primitives, all preconditions, and the cleanup invariants.
- `tests/unit/scripts/test_error_handling_ratchet.py` — 10 tests covering
  the ratchet (equal, decreasing, growing, missing baseline, malformed
  baseline, `--update-baseline` flag).
- `python3 scripts/ci/check_error_handling_ratchet.py` is required in
  `ci-standard.yml`.
- `ruff check src/` is already required; with `BLE001/F841/F401` removed
  from `extend-ignore`, those rules now block new violations.
