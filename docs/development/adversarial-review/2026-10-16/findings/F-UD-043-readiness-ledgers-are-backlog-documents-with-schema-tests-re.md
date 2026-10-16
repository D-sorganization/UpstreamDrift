# F-UD-043

**Readiness ledgers are backlog documents with schema tests; `release_status` is prose, not a computed quantity**

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| ⬜ Low   | governance | medium     | M      | proposed |

**Locations:** `src/config/bunkershot3d_qualification.json:1`, `src/config/impact_zone_readiness.json:1`, `CLAUDE.md:177`

## Finding

- Severity: Low - Category: governance
- Confidence: medium
- Location: src/config/bunkershot3d_qualification.json:1; src/config/impact_zone_readiness.json:1; CLAUDE.md:177-219
- Evidence: `bunkershot3d_qualification.json`: 16 children all `open`, every acceptance criterion `unmet/partial`,
  `release_status: blocked`; `industrial_readiness.json` 4 children; `feature_parity.json` 62 features (42 parity, 16 exempt,
  4 gap). CLAUDE.md spends 43 lines (177–219) mandating updates to these for every PR that touches a child.
- Failure scenario: these are honest (all blocked), which is the good news; the cost is that the schema tests (and the
  regenerated `*.md` indexes, with freshness gates) add a mandatory step to every related PR without producing a measurement —
  `release_status` is hand-entered with a consistency rule, not computed. See F-UD-024 for the inconsistency this produced.
- Why a newer engineer misses this: machine-readable ≠ machine-verified.
- Proposed change: derive `release_status` from the children (one line of code) and derive each child's `status` from GitHub
  issue state at collate time; drop the prose `evidence` fields that duplicate PR descriptions.
- Verification: `python -m scripts.generate_industrial_readiness_index --check` recomputes status and fails on drift.
- Effort: M

## Draft Pull Request

**Title:** `refactor(config): compute readiness release_status from children`

### Why This Matters (The Pitch)

The readiness ledgers are honest, all blocked, but they are backlog documents with schema tests. CLAUDE.md spends 43 lines mandating their updates on every PR touching a child, while release_status is hand-entered rather than computed. That adds a mandatory step without producing a measurement, and it produced the inconsistency in F-UD-024. Deriving release_status from the children and each child's status from GitHub issue state at collate time removes the manual step.

### Proposed PR Body

## Problem

`src/config/bunkershot3d_qualification.json:1` (16 children all `open`, `release_status: blocked`), `impact_zone_readiness.json:1`, `industrial_readiness.json` (4 children), `feature_parity.json` (62 features); `CLAUDE.md:177-219` mandates updates. Confidence is medium.

## Change

- [ ] Derive `release_status` from the children (one line of code)
- [ ] Derive each child's `status` from GitHub issue state at collate time
- [ ] Drop prose `evidence` fields that duplicate PR descriptions

## Verification

`python -m scripts.generate_industrial_readiness_index --check` recomputes status and fails on drift.

## Risk and Rollback

Low to medium: the collate step needs GitHub access. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-024.

Refs: adversarial review 2026-10-16, F-UD-043

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
