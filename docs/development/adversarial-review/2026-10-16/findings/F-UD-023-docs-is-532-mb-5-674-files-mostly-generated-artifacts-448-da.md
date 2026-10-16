# F-UD-023

**`docs/` is 532 MB / 5,674 files, mostly generated artifacts; 448 dangling source references**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | docs     | high       | M      | proposed |

**Locations:** `docs/research/shot_pattern_analysis/results/shots.csv:1`, `docs/development/DEVELOPMENT_LOG.md:1`, `docs/development/org01_capability_baseline/inventory-putting.md:12`

## Finding

- Severity: Medium - Category: docs
- Confidence: high
- Location: docs/research/shot_pattern_analysis/results/shots.csv:1; docs/development/DEVELOPMENT_LOG.md:1;
  docs/development/org01_capability_baseline/inventory-putting.md:12-14
- Evidence: file types in `docs/`: 2,064 json, 1,542 md, 505 txt, 299 png, 250 gz, 170 pdf, 156 npz, 140 py; 28 files >5 MB
  (ten `shots.csv` of 5.6 MB each in `corrected_results/*`); the shallow clone's pack is 460 MB. A scan of 1,542 markdown files
  finds 5,843 `src|scripts|tests/...` path references of which 448 (in 153 files) do not exist, e.g.
  `tests/config/feature_parity/test_feature_parity_registry.py` (10×), `src/shared/python/physics/impact_model.py` (14×).
  `DEVELOPMENT_LOG.md` is 6,183 lines; `SPEC.md` 343 kB with 395 change-log rows.
- Failure scenario: agents retrieve stale docs via RAG and act on module paths that moved; every clone and CI checkout pays for
  292 MB of CSV; `docs-governance.yml` validates structure, not currency.
- Why a newer engineer misses this: committing results next to the code that produced them feels like reproducibility; it is
  until the code changes.
- Proposed change: move `docs/research/**/results` and `*.npz/*.gz/*.csv > 1 MB` to a release asset / DVC remote with a manifest
  of hashes; add a dangling-reference check (the script I ran is 20 lines) to `docs-ci.yml`; archive `DEVELOPMENT_LOG.md`
  entries older than 90 days.
- Verification: `du -sh docs` < 50 MB; the dangling-ref count in CI is 0.
- Effort: M

## Draft Pull Request

**Title:** `chore(docs): move generated artifacts out of docs; check dangling refs`

### Why This Matters (The Pitch)

docs/ is 532 MB across 5,674 files, mostly generated artifacts. Ten shots.csv files of 5.6 MB each sit under corrected_results, 28 files exceed 5 MB, and the shallow clone's pack is 460 MB, so every clone and every CI checkout pays for roughly 292 MB of CSV and NPZ. A scan of 1,542 markdown files finds 5,843 src, scripts or tests path references, of which 448 in 153 files do not exist, for example tests/config/feature_parity/test_feature_parity_registry.py cited 10 times. Agents retrieving these via RAG act on module paths that moved, and docs-governance.yml checks structure, not currency. Moving results and files over 1 MB to a release asset or DVC remote with a hash manifest, adding a 20-line dangling-reference check to docs-ci.yml, and archiving DEVELOPMENT_LOG.md entries older than 90 days shrinks clones and stops the docs misleading the next reader. Size M.

### Proposed PR Body

## Problem

`docs/` is 532 MB / 5,674 files (2,064 json, 505 txt, 299 png, 250 gz, 170 pdf, 156 npz); 28 files exceed 5 MB (ten `shots.csv` in `corrected_results/*`). 448 of 5,843 path references in 153 markdown files do not exist (e.g. `src/shared/python/physics/impact_model.py` 14×). `DEVELOPMENT_LOG.md` is 6,183 lines.

## Change

- [ ] Move `docs/research/**/results` and `*.npz/*.gz/*.csv > 1 MB` to a release asset / DVC remote with a hash manifest
- [ ] Add a dangling-reference check to `docs-ci.yml`
- [ ] Archive `DEVELOPMENT_LOG.md` entries older than 90 days

## Verification

`du -sh docs` < 50 MB; the dangling-ref count in CI is 0.

## Risk and Rollback

Medium: reproducing research results needs the external store; the manifest of hashes keeps them verifiable. Rollback is restoring the files from history.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-039 (binary artifacts elsewhere in the repo).

Refs: adversarial review 2026-10-16, F-UD-023

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
