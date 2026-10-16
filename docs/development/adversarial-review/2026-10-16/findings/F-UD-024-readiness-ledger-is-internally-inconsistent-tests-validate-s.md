# F-UD-024

**Readiness ledger is internally inconsistent; tests validate schema, not semantics**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `src/config/industrial_readiness.json:54`, `src/config/industrial_readiness.json:95`, `src/config/industrial_readiness.json:124`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: src/config/industrial_readiness.json:54-57; src/config/industrial_readiness.json:95; src/config/industrial_readiness.json:124
- Evidence: U3 (#8820) has `"status": "merged"` with a merge SHA, yet acceptance `children-reconciled` says "U3 and U4 were
  re-confirmed open" and `no-false-green` lists `8820` among its blockers. `tests/config` passes (ran: all green).
- Failure scenario: the ledger's purpose is to stop "readiness nobody measured"; it currently records readiness it contradicts.
  The contract test checks SHA length and path existence, so an agent that flips a status without touching evidence prose passes.
- Why a newer engineer misses this: JSON schema validation feels like semantic validation.
- Proposed change: add a consistency rule — a `merged` key may not appear in any acceptance `blockers`; evidence prose must not
  name a merged key as open (simple regex on `"U\d+"`).
- Verification: the new test fails on today's file, passes after the two edits.
- Effort: S

## Draft Pull Request

**Title:** `test(config): reject merged readiness keys named as open or blocking`

### Why This Matters (The Pitch)

The industrial readiness ledger exists to stop readiness nobody measured, and it currently records readiness it contradicts. U3 (#8820) has status merged with a merge SHA, yet the acceptance criterion children-reconciled says U3 and U4 were re-confirmed open, and no-false-green lists 8820 among its blockers. tests/config passes anyway because the contract test checks SHA length and path existence: an agent can flip a status without touching the evidence prose and sail through. Schema validation is feeling like semantic validation. A small consistency rule, a merged key may not appear in any acceptance blockers and evidence prose may not name a merged key as open (a simple regex on U followed by digits), fails on today's file and passes after two edits. Size S, and the ledger starts earning the trust its name asks for.

### Proposed PR Body

## Problem

`src/config/industrial_readiness.json:54-57`, `:95`, `:124`: U3 (#8820) has `"status": "merged"` with a merge SHA, yet `children-reconciled` says "U3 and U4 were re-confirmed open" and `no-false-green` lists `8820` as a blocker; `tests/config` passes.

## Change

- [ ] Add a test rule: a `merged` key may not appear in any acceptance `blockers`
- [ ] Add a rule: evidence prose must not name a merged key as open (regex on `"U\d+"`)
- [ ] Fix the two inconsistent entries

## Verification

The new test fails on today's file and passes after the two edits.

## Risk and Rollback

Low: test and data edit only. Rollback is a revert.

## Handoff

Follow-ups: F-UD-043 (derive status instead of hand-entering it).

Refs: adversarial review 2026-10-16, F-UD-024

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
