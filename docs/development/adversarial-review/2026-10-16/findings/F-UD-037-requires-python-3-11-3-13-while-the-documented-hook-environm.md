# F-UD-037

**`requires-python = ">=3.11,<3.13"` while the documented hook environment is Python 3.13.3**

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| ⬜ Low   | governance | high       | S      | proposed |

**Locations:** `pyproject.toml:8`, `CLAUDE.md:356`, `.pre-commit-config.yaml:137`

## Finding

- Severity: Low - Category: governance
- Confidence: high
- Location: pyproject.toml:8; CLAUDE.md:356-365; .pre-commit-config.yaml:137-139
- Evidence: CLAUDE.md: "`default_language_version: python: python` … resolves to … Python 3.13.3 on the current workstation";
  pre-push `mypy`/`pytest-unit` hooks therefore run under an interpreter `pip install -e .` refuses.
- Failure scenario: hooks pass locally on 3.13 for code that fails on 3.11 (CI matrix 3.11/3.12), or vice versa (StrEnum,
  typing changes); the "Windows hook environment — Resolved" section documents the mismatch as resolved.
- Why a newer engineer misses this: `python` on PATH is whatever was installed last.
- Proposed change: either raise `requires-python` to `<3.14` (the test run here on 3.13.16 was fine) or pin the hook language
  version to 3.12 with `uv`/pyenv.
- Verification: `pre-commit run --all-files` under the pinned interpreter.
- Effort: S

## Draft Pull Request

**Title:** `build(python): reconcile requires-python with the hook interpreter`

### Why This Matters (The Pitch)

pyproject.toml says >=3.11,<3.13, but CLAUDE.md documents the pre-commit hook environment as Python 3.13.3, an interpreter pip install -e . refuses. Pre-push mypy and pytest-unit hooks can pass on 3.13 for code that fails on the 3.11/3.12 CI matrix, or the reverse. Either raise the upper bound to below 3.14 or pin the hook language version to 3.12.

### Proposed PR Body

## Problem

`pyproject.toml:8` sets `requires-python = ">=3.11,<3.13"`; `CLAUDE.md:356-365` and `.pre-commit-config.yaml:137-139` run hooks under Python 3.13.3, and the CLAUDE.md section calls the mismatch resolved.

## Change

- [ ] Raise `requires-python` to `<3.14` (the review's test run on 3.13.16 was fine), or pin the hook language version to 3.12 with `uv`/pyenv

## Verification

`pre-commit run --all-files` under the pinned interpreter.

## Risk and Rollback

Low: version metadata only; widening the range requires the CI matrix to cover 3.13. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-037

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
