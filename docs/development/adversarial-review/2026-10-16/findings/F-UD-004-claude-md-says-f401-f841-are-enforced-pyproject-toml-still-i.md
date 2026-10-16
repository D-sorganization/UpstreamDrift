# F-UD-004

**CLAUDE.md says `F401`/`F841` are enforced; `pyproject.toml` still ignores them (1,372 unused imports hidden)**

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| 🟧 High  | governance | high       | M      | proposed |

**Locations:** `pyproject.toml:529`, `CLAUDE.md:234`, `scripts/ci/check_error_handling_ratchet.py:43`, `scripts/config/error_handling_baseline.json:13`

## Finding

- Severity: High - Category: governance
- Confidence: high
- Location: pyproject.toml:529; CLAUDE.md:234-239; scripts/ci/check_error_handling_ratchet.py:43-48;
  scripts/config/error_handling_baseline.json:13-15
- Evidence:
  ```toml
  extend-ignore = ["E501", "C901", "E402", "I001", "SIM118", "SIM105", "SIM117", "PLE0604", "SIM102", "SIM113", "PERF403", "F841", "F401",
  ```
  CLAUDE.md: "Lint rules enforced (no longer in `extend-ignore`): BLE001 … F841 … F401". The ratchet regexes
  `#\s*noqa\s*:[^#]*F401` count _comments_.
- Failure scenario: `ruff --extend-select F401,F841` on the tree reports 1,372 unused imports and 123 unused locals today. The
  ratchet reports "IMPROVED noqa_F401: was 711, now 201 (−510)" — those were comments deleted alongside files, not imports fixed;
  the rule never fired. Agents reading CLAUDE.md add `# noqa: F401` to silence a warning that never appears.
- Why a newer engineer misses this: the ratchet produces green output with the word IMPROVED in it. Nobody checks whether the
  thing being counted can occur.
- Proposed change: remove `F841`, `F401` from `extend-ignore`, run `ruff --fix` (1,293 are auto-fixable), keep `__all__`
  re-exports via the redundant-alias idiom the doc already prescribes; delete the `noqa_F401`/`noqa_F841` counters from the ratchet.
- Verification: `ruff check . --statistics` shows 0 F401/F841; `check_error_handling_ratchet.py` no longer lists those keys.
- Effort: M

## Draft Pull Request

**Title:** `build(lint): stop ignoring F401/F841 and drop the dead noqa counters`

### Why This Matters (The Pitch)

CLAUDE.md tells every agent that F401 and F841 are enforced and no longer in extend-ignore. pyproject.toml:529 says otherwise, and turning them on reports 1,372 unused imports and 123 unused locals, 1,293 of them auto-fixable. Meanwhile the error-handling ratchet prints 'IMPROVED noqa_F401: was 711, now 201 (−510)', which sounds like progress but is a count of noqa comments for a rule that never fires; the comments left when files were deleted. Agents reading the doc add '# noqa: F401' to silence a warning that does not exist. The result is a green dashboard measuring nothing, plus 1,372 pieces of dead weight every reader must skim. This change is size M but mostly mechanical: ruff --fix does the bulk, the re-export idiom the doc already prescribes keeps **all** intact, and the counters get deleted. Net effect: the docs become true, the ratchet measures something real, and the tree gets lighter.

### Proposed PR Body

## Problem

`pyproject.toml:529` still lists `F841` and `F401` in `extend-ignore` while `CLAUDE.md:234-239` says they are enforced. `ruff --extend-select F401,F841` reports 1,372 unused imports and 123 unused locals. `scripts/ci/check_error_handling_ratchet.py:43-48` counts `# noqa: F401` comment text, so "IMPROVED noqa_F401: was 711, now 201" reflects deleted comments, not fixes.

## Change

- [ ] Remove `F841` and `F401` from `extend-ignore` in `pyproject.toml`
- [ ] Run `ruff --fix` (1,293 are auto-fixable); keep `__all__` re-exports via the redundant-alias idiom the doc prescribes
- [ ] Delete the `noqa_F401`/`noqa_F841` counters from `check_error_handling_ratchet.py` and its baseline

## Verification

`ruff check . --statistics` shows 0 F401/F841; `check_error_handling_ratchet.py` no longer lists those keys.

## Risk and Rollback

Medium: removing an import that has side effects or is a re-export could break callers; the large diff should be reviewed for those. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-010 (make ratchets actually shrink).

Refs: adversarial review 2026-10-16, F-UD-004

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
