# F-UD-039

**Binary research artifacts committed without LFS: 590 `.fig`, 114 `.mat`, 184 `.npz`, 256 `.gz`, 170 `.pdf`**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | M      | proposed |

**Locations:** `docs/research/shot_pattern_analysis/results/shots.csv:1`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: .gitattributes:1; docs/research/shot_pattern_analysis/results/shots.csv:1
- Evidence: `git ls-files | grep -E '\.(fig|mat|npz|gz|pdf)$'` counts above (643 `.fig/.mat` under
  `src/engines/Simscape_Multibody_Models/2D_Golf_Model`, 242 `.npz/.gz` under `docs/research/shot_pattern_analysis`);
  `grep -c lfs .gitattributes` → 0; the pack of a shallow clone is 460 MB.
- Failure scenario: every agent session (dozens per night) clones 460 MB; `.fig` files are opaque to review and to the
  "non-editable generated artifacts" rule the design-manual section imposes elsewhere.
- Why a newer engineer misses this: MATLAB figures and results feel like source until the repo is 1 GB.
- Proposed change: `git lfs track` or move to a `UpstreamDrift-artifacts` repo with a hash manifest; keep generators.
- Verification: fresh shallow clone < 100 MB.
- Effort: M

## Draft Pull Request

**Title:** `chore(repo): move binary research artifacts to LFS or artifacts repo`

### Why This Matters (The Pitch)

The repo tracks 590 .fig, 114 .mat, 184 .npz, 256 .gz and 170 .pdf files with no LFS configuration; a shallow clone's pack is 460 MB. Every agent session, dozens a night, pays for it, and .fig files are opaque to review. Move them to git LFS or a UpstreamDrift-artifacts repo with a hash manifest and keep the generators.

### Proposed PR Body

## Problem

`.gitattributes:1` has no LFS rules (`grep -c lfs .gitattributes` → 0); 643 `.fig/.mat` live under `src/engines/Simscape_Multibody_Models/2D_Golf_Model` and 242 `.npz/.gz` under `docs/research/shot_pattern_analysis`.

## Change

- [ ] `git lfs track` the binary types, or move them to a separate artifacts repo with a hash manifest
- [ ] Keep the generating scripts in this repo

## Verification

A fresh shallow clone is < 100 MB.

## Risk and Rollback

Medium: history rewriting or external storage affects anyone with local clones. Rollback is restoring from history.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-023.

Refs: adversarial review 2026-10-16, F-UD-039

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
