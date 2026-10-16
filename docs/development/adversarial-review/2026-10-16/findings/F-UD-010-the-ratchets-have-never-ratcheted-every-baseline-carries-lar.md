# F-UD-010

**The ratchets have never ratcheted; every baseline carries large headroom**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `scripts/config/error_handling_baseline.json:12`, `scripts/config/suppression_ratchet_baseline.json:3`, `scripts/ci/lod_baseline.txt:1`, `scripts/config/dry_duplication_quarantine.json:1`, `scripts/config/hardcoded_style_baseline.json:1`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: scripts/config/error_handling_baseline.json:12-18; scripts/config/suppression_ratchet_baseline.json:3-6;
  scripts/ci/lod_baseline.txt:1-5; scripts/config/dry_duplication_quarantine.json:1; scripts/config/hardcoded_style_baseline.json:1
- Evidence (this run): `noqa_F841 22→10`, `noqa_F401 711→201`, `raw_popen 42→27`, `bare_type_ignore 60→38`,
  hardcoded-style `165→162`, LOD `468 baseline, 81 reductions`, DRY `171 quarantined fingerprints now occur fewer times than
recorded`, suite markers `can shrink by 248 entries`. Baselines dated 2026-05-21/24; the `--update-baseline` flags exist.
- Failure scenario: a PR can add 510 `# noqa: F401`, 15 raw `Popen`, 22 bare `type: ignore`, 81 LoD chains and 171 duplicate
  blocks and every ratchet prints OK. A ratchet that is never tightened is a ceiling, and a ceiling that is 3× above the floor
  is decoration.
- Why a newer engineer misses this: the scripts print "IMPROVED" and exit 0; the one-way design puts the tightening step on a
  human who is never prompted.
- Proposed change: make each ratchet fail when `current < baseline - slack` ("you must lower the baseline in this PR"), or run
  `--update-baseline` in the collate step after merge. Delete counters for rules that cannot fire (F-UD-004).
- Verification: re-run the scripts; each should fail until baselines are regenerated.
- Effort: S

## Draft Pull Request

**Title:** `ci(ratchets): fail when baselines are loose so they must be lowered`

### Why This Matters (The Pitch)

Every ratchet in this repo prints IMPROVED and exits 0, and none has ever been tightened. This run: noqa_F841 22 to 10, noqa_F401 711 to 201, raw_popen 42 to 27, bare_type_ignore 60 to 38, hardcoded-style 165 to 162, LoD 81 reductions available, 171 stale DRY quarantine fingerprints, and 248 suite-marker entries that could shrink. That headroom means one PR can add 510 noqa comments, 15 raw Popen calls, 22 bare type-ignores, 81 LoD chains and 171 duplicate blocks while every gate stays green. A ratchet that is never tightened is a ceiling, and a ceiling three times above the floor is decoration. Making each script fail when the current count drops below baseline minus slack, or running --update-baseline in the existing post-merge collate step, converts the dashboard into an actual one-way valve. Everyone who relies on the gates gets gates that bite.

### Proposed PR Body

## Problem

`scripts/config/error_handling_baseline.json:12-18`, `suppression_ratchet_baseline.json:3-6`, `scripts/ci/lod_baseline.txt:1-5`, `dry_duplication_quarantine.json:1` and `hardcoded_style_baseline.json:1` carry large headroom (e.g. `noqa_F401 711→201`, `raw_popen 42→27`, `bare_type_ignore 60→38`, LOD `468 baseline, 81 reductions`). Baselines are dated 2026-05-21/24 and the `--update-baseline` flags exist but are never run.

## Change

- [ ] Make each ratchet fail when `current < baseline - slack` with a message to lower the baseline in the PR, or run `--update-baseline` in the collate step after merge
- [ ] Delete counters for rules that cannot fire (see F-UD-004)
- [ ] Regenerate all baselines once

## Verification

Re-run the scripts; each should fail until baselines are regenerated.

## Risk and Rollback

Low: a failing gate on loose baselines is the intended behaviour; slack avoids flakiness. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-004.

Refs: adversarial review 2026-10-16, F-UD-010

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
