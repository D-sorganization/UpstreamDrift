# Implementation Handoff — Fabricated-Evidence Audit P0 Batch (#10960)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-10960-p0-batch`
- Branch: `claude/ud-10960-p0-batch`
- Baseline commit: origin/main at branch creation
- Implementation commit: `SELF`
- Pull request: draft, opened from this branch
- Governing issue: #10960 (development log DL-#10960)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: remove literal success values from seven P0 findings of the #10960 audit (P0-1, P0-3, P0-5/6, P0-7, P0-8, P0-24) plus P0-9 (FB receipts), P1-7 (parity report), P1-8 (OpenSim document IK), P1-9 (failed rollout audit) and P1-10 (sample strategy package).
- Status: slices executed by agy (Gemini 3.8 Flash) in isolated worktrees; orchestrator review removed
  a new self-qualifying checkpoint loader (P0-24), a summed-units selection score (P0-3), speculative Moco
  adapters (P0-8), zero-cost defaults (P0-6), and duplicated guards (DRY).

## Files and Decisions

- See DL-#10960 Paths. Decisions: unmeasured -> None/UNQUALIFIED/"unmeasured" or NotImplementedError; surrogate
  selection is lexicographic on measured errors (no invented weights); contact receipts versioned
  `matched-swing-contact-id/v2` with `kind` synthetic vs measured.

## Validation

- 113/113 tests pass in the nine touched test files (junit). artifact_audit x2 and event_alignment x8 fail
  identically on origin/main. OpenSim capability test needs the OpenSim runtime (absent locally).

## Blockers and Risks

- No training / native-replay / measured-GRF pipeline exists, so the affected matrices now report
  unqualified everywhere. This is intended, but downstream dashboards will show fewer "passing" cells.

## Next Steps

1. CI green on the draft PR; review; merge.
2. Remaining #10960 findings: P0-10, P1-11 and P1-1..6 wait for #10973 (shared files).
3. Follow-ups: `CandidateSession.rms_error` returns 0.0 when no receipt metric exists; `ledger.extract_horizon_s` assumes 360 Hz when a receipt has frames but no `rate_hz`.

## Change Log

- `SELF` — #10960 CI gates: issue refs on fail-closed `NotImplementedError` stubs, one LOD chain removed, and seven over-budget functions split (`MeasuredBaseline`, `IkMeasurement`, per-gate helpers) with no exceptions added.
- `1b9ae9a1e` — #10960 P0 batch: fail-closed receipts for body fit, NM-07/08/09/10/12, MS-20 contact id and OpenSim full swing (DL-#10960).
- `SELF` — Reconcile child-copy convergence, divergence inventory, and agent context views (#10944).
