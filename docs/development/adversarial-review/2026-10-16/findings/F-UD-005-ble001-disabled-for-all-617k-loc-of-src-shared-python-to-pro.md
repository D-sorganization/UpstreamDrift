# F-UD-005

**`BLE001` disabled for all 617k LOC of `src/shared/python/**` to protect a 110k-LOC Tools copy\*\*

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| 🟧 High  | governance | high       | S      | proposed |

**Locations:** `pyproject.toml:572`, `scripts/config/error_handling_baseline.json:4`, `tests/unit/repo_hygiene/tools_child_copy_missing_counterparts.txt:1`

## Finding

- Severity: High - Category: governance
- Confidence: high
- Location: pyproject.toml:572-578; scripts/config/error_handling_baseline.json:4; tests/unit/repo_hygiene/tools_child_copy_missing_counterparts.txt:1-13
- Evidence:
  ```toml
  # Deferred: all of src/shared/python/ is a Tools-owned child copy tree
  "src/shared/python/**" = ["BLE001"]
  ```
  Measured against Tools HEAD: 631 of 2,377 files under `src/shared/python` have a Tools counterpart (sidekick 273, ai 125,
  model_generation 73, chat 57, calc_backend 44, humanoid_character_builder 40); the other 1,746 files (physics, motion_matching,
  simulation_backends, estimation, pose_interchange …) are UD-owned. `ruff --isolated --select BLE001 src/shared/python` → 63.
- Failure scenario: new blind `except Exception:` in the numerical core (motion_matching, physics) lands without a lint hit or a
  `narrow_catch`, contradicting the issue-#5911 policy the same file documents. The baseline comment admits the per-file ignore
  dropped the counted sites from 369 to 342 "net DECREASE" — the count went down because the detector was turned off.
- Why a newer engineer misses this: a glob is the fastest way to make CI green on a tree you are told not to touch; the fact
  that the glob covers 5× more code than the justification is invisible without counting.
- Proposed change: scope the ignore to the actual child-copy roots (`src/shared/python/{sidekick,ai,chat,calc_backend,
model_generation,humanoid_character_builder,gui_launcher}/**`) and annotate the remaining 63 sites inline.
- Verification: `ruff check src/shared/python --select BLE001` with the narrowed config returns only child-copy roots.
- Effort: S

## Draft Pull Request

**Title:** `build(lint): narrow the BLE001 ignore to the real Tools child-copy roots`

### Why This Matters (The Pitch)

To avoid touching a Tools-owned child copy, pyproject.toml disables BLE001 for the whole of src/shared/python/\*\*, which is 617k LOC. Measured against Tools HEAD, 631 of 2,377 files there have a Tools counterpart; the other 1,746, including physics, motion_matching, simulation_backends, estimation and pose_interchange, are owned by UpstreamDrift. So the numerical core can acquire a blind 'except Exception:' with no lint hit and no narrow_catch, contradicting the issue-#5911 policy the same file documents. The baseline comment even celebrates a 'net DECREASE' from 369 to 342 sites, a decrease that came from switching the detector off. Scoping the ignore to the child-copy roots (sidekick, ai, chat, calc_backend, model_generation, humanoid_character_builder, gui_launcher) re-arms the rule where the team owns the code; only 63 sites then need inline annotation. Size S, and it makes the error-handling policy true again.

### Proposed PR Body

## Problem

`pyproject.toml:572-578` sets `"src/shared/python/**" = ["BLE001"]` to protect Tools child copies. Only 631 of 2,377 files under that tree have a Tools counterpart (sidekick 273, ai 125, model_generation 73, chat 57, calc_backend 44, humanoid_character_builder 40); 1,746 are UD-owned. `ruff --isolated --select BLE001 src/shared/python` finds 63 sites.

## Change

- [ ] Scope the ignore to `src/shared/python/{sidekick,ai,chat,calc_backend,model_generation,humanoid_character_builder,gui_launcher}/**`
- [ ] Annotate the remaining 63 sites inline (narrow the catch or justify it)

## Verification

`ruff check src/shared/python --select BLE001` with the narrowed config returns only child-copy roots.

## Risk and Rollback

Low: lint-config change plus inline annotations; behaviour is unchanged. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-005

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
