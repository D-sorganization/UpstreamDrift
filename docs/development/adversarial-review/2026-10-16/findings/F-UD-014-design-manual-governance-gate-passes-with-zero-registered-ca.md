# F-UD-014

**Design-manual governance gate passes with zero registered calculations**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | M      | proposed |

**Locations:** `scripts/config/design_manual_governance.json:21`, `CLAUDE.md:13`, `AGENTS.md:69`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: scripts/config/design_manual_governance.json:21-24; CLAUDE.md:13-20; AGENTS.md:69-78
- Evidence: `python3 -m scripts.check_design_manual_governance` → "Design-manual governance verified: 68 QMD sources,
  0 registered calculations, release=blocked-inventory-required." exit 0.
- Failure scenario: CLAUDE.md tells every agent to read this JSON and run this gate "before changing calculations"; the gate
  cannot relate any calculation to any source because the inventory is empty, so it will pass for any change. It is a
  pre-commit hook (`design-manual-governance`) that costs a process start per commit and catches nothing.
- Why a newer engineer misses this: a passing gate is read as "governed"; the message says `blocked` and `verified` in the same
  line.
- Proposed change: either populate `manuals/upstreamdrift/calculation-registry.json` for the ≥5 pathways that matter
  (ZTCF/ZVCF, ball flight, GCI, rigid contact, GRF) or remove the hook and the CLAUDE.md mandate until it does.
- Verification: the gate fails when a governed source file changes without a registry/SPEC update.
- Effort: M

## Draft Pull Request

**Title:** `ci(governance): populate the calculation registry or retire the hook`

### Why This Matters (The Pitch)

CLAUDE.md instructs every agent to read design_manual_governance.json and run the gate before changing calculations. The gate reports 'verified: 68 QMD sources, 0 registered calculations, release=blocked-inventory-required' and exits 0, using 'blocked' and 'verified' in the same breath. With an empty inventory it cannot relate any calculation to any source, so it passes for any change, while running as a pre-commit hook that costs a process start per commit and catches nothing. A passing gate gets read as 'governed'. Two honest options: register the five pathways that matter (ZTCF/ZVCF, ball flight, GCI, rigid contact, GRF) so the gate can fail when a governed source changes without a registry or SPEC update, or remove the hook and the CLAUDE.md mandate until that exists. Either way agents stop spending prompt tokens on a ritual with no teeth.

### Proposed PR Body

## Problem

`python3 -m scripts.check_design_manual_governance` prints "Design-manual governance verified: 68 QMD sources, 0 registered calculations, release=blocked-inventory-required." and exits 0 (`scripts/config/design_manual_governance.json:21-24`), yet `CLAUDE.md:13-20` and `AGENTS.md:69-78` mandate it before calculation changes.

## Change

- [ ] Populate `manuals/upstreamdrift/calculation-registry.json` for the ≥5 pathways that matter (ZTCF/ZVCF, ball flight, GCI, rigid contact, GRF), or
- [ ] Remove the `design-manual-governance` hook and the CLAUDE.md mandate until it does

## Verification

The gate fails when a governed source file changes without a registry/SPEC update.

## Risk and Rollback

Low to medium: a populated registry will start failing PRs that touch governed files; a removal is safe. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-014

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
