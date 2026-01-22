# Agent Guidance vs. Repo Automation Alignment Assessment

This document summarizes how the repository guidance (AGENTS.md and related agent directives) aligns with the actual GitHub workflows and CI/CD checks currently present in this repository. It also records notable gaps and recommendations for improving automation so that future agents can minimize manual intervention.

## Scope and Sources Reviewed

- `AGENTS.md` (repo-wide guidance)
- `docs/development/AGENTS.md` (docs/development scope)
- GitHub workflows under `.github/workflows/`
- `pyproject.toml` (lint/test/type config)
- `.pre-commit-config.yaml` (local checks)
- `tools/code_quality_check.py` (quality checks)

## Alignment Summary

### ✅ Strong Alignment

1. **Critical files protection**
   - Guidance explicitly protects root files like `AGENTS.md`, `README.md`, `CHANGELOG.md`, `LICENSE`, `pyproject.toml`, `conftest.py`, and key entry points.
   - `critical-files-guard.yml` enforces this on PRs to `main` and `develop`.

2. **Control Tower orchestration**
   - Guidance describes a “Control Tower” that dispatches worker workflows based on events/schedules.
   - `Jules-Control-Tower.yml` exists and dispatches workers on CI failure, PR events, scheduled runs, and workflow dispatch.

3. **Worker workflow presence**
   - Guidance lists Test Generator, Documentation Scribe, Conflict Fix, and Tech Debt Assessor workflows.
   - Corresponding workflows exist:
     - `Jules-Test-Generator.yml`
     - `Jules-Documentation-Scribe.yml`
     - `Jules-Conflict-Fix.yml`
     - `Jules-Tech-Debt-Assessor.yml`

### ⚠️ Misalignment / Gaps

1. **Auto-Repair is described but disabled**
   - Guidance lists Auto-Repair as active for CI failures.
   - The workflow `Jules-Auto-Repair.yml` is currently disabled (`if: false`), so the behavior does not match guidance.

2. **Workflow naming inconsistencies**
   - Guidance references lowercase workflow filenames (e.g., `jules-control-tower.yml`).
   - Actual workflows use capitalized filenames (e.g., `Jules-Control-Tower.yml`).

3. **Pre-commit expectations vs. actual hooks**
   - Guidance lists commands like `ruff format` and emphasizes broad lint/type checks.
   - `.pre-commit-config.yaml` currently includes only:
     - `black`
     - `ruff --fix`
     - `mypy` (with `--ignore-missing-imports`)
   - `ruff format` is not configured in pre-commit.

4. **CI configuration does not match user-stated CI expectations**
   - The repository does **not** contain `.github/workflows/ci.yml` or `pr-quality-check.yml`.
   - Actual CI is driven by `ci-standard.yml` and `ci-fast-tests.yml`.
   - `ci-standard.yml` runs:
     - `ruff check`
     - `black --check` (line length 88)
     - `mypy`
     - TODO/FIXME grep
     - `pip-audit`
     - MATLAB quality check (non-blocking)
     - pytest (+ cross-engine validation)
   - There is no isort job in CI and Ruff is not set to “select ALL.”

5. **Formatting and linting differences**
   - Repo enforces Black line length **88** in `pyproject.toml`.
   - Guidance and user-provided instructions reference line length **100**.
   - Ruff’s rule selection is a curated subset, not “ALL.”

6. **Bandit is not enforced in CI Standard**
   - `ci-standard.yml` uses `pip-audit` for dependency vulnerabilities.
   - Bandit appears in Tech Debt Assessor but is not a gating CI step.

## Current CI/CD Checks (Actual Behavior)

### CI Standard (`ci-standard.yml`)
- Linting: `ruff check .`
- Formatting: `black --check .`
- Type checking: `mypy . --config-file pyproject.toml`
- Placeholder detection: grep TODO/FIXME (blocking)
- Security: `pip-audit` (blocking)
- MATLAB quality check: `tools/matlab_utilities/scripts/matlab_quality_check.py` (non-blocking)
- Tests: pytest + cross-engine validation + Codecov (if token present)

### CI Fast Tests (`ci-fast-tests.yml`)
- Fast unit and integration test jobs with pytest-xdist/timeouts.
- Designed for quick signal on PRs/branches.

### Critical File Guard (`critical-files-guard.yml`)
- Enforces the protected file list specified in `AGENTS.md`.

## Recommendations to Improve Automation and Reduce Manual Oversight

1. **Update guidance to match actual CI**
   - Reflect real workflow names and the disabled Auto-Repair state.
   - Document that CI uses `ci-standard.yml` and `ci-fast-tests.yml`, not `ci.yml`.

2. **Align pre-commit with CI**
   - Add hooks for `pip-audit`, placeholder checks, and `tools/code_quality_check.py`.
   - This reduces CI failures after pushing.

3. **Integrate `tools/code_quality_check.py` into CI**
   - The repository already includes a robust quality checker.
   - Adding it to CI (and/or pre-commit) would enforce placeholder/magic-number rules automatically.

4. **Consider adding Bandit to CI Standard**
   - Bandit runs in Tech Debt Assessor but does not gate CI.
   - If security scanning is expected to gate merges, add Bandit to `ci-standard.yml`.

5. **Add a PR-specific quality workflow (optional)**
   - If you want a separate non-blocking PR workflow, add `pr-quality-check.yml`.
   - Alternatively, document that `ci-standard.yml` already runs on PRs.

## IDE + Agent Workflow Tips (Claude / Antigravity)

1. **Use Black line length 88 locally**
   - Match `pyproject.toml` to avoid formatting churn.

2. **Run Ruff + Mypy on save**
   - Aligns with CI Standard and avoids post-hoc rework.

3. **Add a “CI parity” task**
   - A single IDE task that runs:
     - `ruff check .`
     - `black --check .`
     - `mypy .`
     - `pytest -m "not slow"`

4. **Prompt the model for typed signatures + docstrings**
   - This aligns with mypy strictness and the code-quality checks.

## Notes for Future Agents

- The user-provided CI instructions differ from the repo’s actual workflows.
- Always confirm whether CI and lint/format rules are enforced by `ci-standard.yml`.
- Do **not** assume `scripts/quality_check.py` exists at the repo root; quality checks live under `tools/` and per-engine directories.

