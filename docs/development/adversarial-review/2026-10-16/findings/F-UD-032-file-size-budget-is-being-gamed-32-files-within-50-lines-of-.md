# F-UD-032

**File-size budget is being gamed: 32 files within 50 lines of the 1,200 cap, one at exactly 1,200**

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| ⬜ Low   | governance | high       | M      | proposed |

**Locations:** `src/tools/shot_pattern_analysis/matched_lean_control.py:1200`, `scripts/config/file_size_budget.json:1`, `.github/workflows/ci-standard.yml:1047`, `CLAUDE.md:130`

## Finding

- Severity: Low - Category: governance
- Confidence: high
- Location: src/tools/shot_pattern_analysis/matched_lean_control.py:1200; scripts/config/file_size_budget.json:1-6;
  .github/workflows/ci-standard.yml:1047; CLAUDE.md:130-131
- Evidence: `check_file_size_budget.py` prints `matched_lean_control.py: 1200 lines`, `matched_supplement_figures.py: 1195`;
  70 files are between 1,000 and 1,200 lines. CLAUDE.md says "1200 lines max"; CI runs `check_module_size_budget.py --max-lines
1500`.
- Failure scenario: files are split at the budget boundary into `_part2.py`-style siblings (see the `*_mixin.py`/`_*_ui.py`
  swarms under `src/launchers/`), which satisfies the counter while increasing coupling. Two different ceilings in doc vs CI.
- Why a newer engineer misses this: a size budget is a proxy for cohesion; agents optimise the proxy.
- Proposed change: budget per _package_ public surface (functions per module, imports per module) or use `radon mi`; align the
  two numbers.
- Verification: a histogram of file sizes in CI output; the spike at 1,150–1,200 should disappear.
- Effort: M

## Draft Pull Request

**Title:** `docs(governance): align file-size ceilings; measure cohesion`

### Why This Matters (The Pitch)

CLAUDE.md says 1200 lines max, CI runs the module-size check at 1500, and 32 files sit within 50 lines of the cap, one at exactly 1,200 (matched_lean_control.py). Files get split at the boundary into part2-style siblings, which satisfies the counter and increases coupling. Aligning the two numbers and measuring per-package public surface or maintainability index gives a budget that tracks cohesion.

### Proposed PR Body

## Problem

`src/tools/shot_pattern_analysis/matched_lean_control.py` is exactly 1200 lines and `matched_supplement_figures.py` is 1195; 70 files are between 1,000 and 1,200. `CLAUDE.md:130-131` says 1200 while `.github/workflows/ci-standard.yml:1047` passes `--max-lines 1500`; see `scripts/config/file_size_budget.json:1-6`.

## Change

- [ ] Align the CLAUDE.md and CI ceilings
- [ ] Budget per package public surface (functions per module, imports per module) or use `radon mi`

## Verification

A histogram of file sizes in CI output; the spike at 1,150–1,200 should disappear.

## Risk and Rollback

Low: governance change. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-032

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
