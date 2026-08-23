# Agent Handoff — Proximal-Distal Program

Last updated: 2026-08-23

## Governing Scope

- Canonical epic: UpstreamDrift #8557.
- This worktree owns #8752 constitutive headline uncertainty:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\goal-8752-uncertainty`.
- Branch: `research/8752-articulated-uncertainty`; remote head before the
  terminal-evidence commit is `2cccde3df6c6fd2aa0c3158749d766041ee588cb`.
- #8800 is the dependency-ordered structural-propagation continuation on
  `research/8800-headline-structural-propagation`, remote head
  `1fddf84a65f976dee7300951b64dcca64dd5ba9e`.
- Do not close #8752 until #8800 is completed or its external boundary is
  explicitly adjudicated. Do not promote either synthetic campaign to a human,
  population, equipment-calibration, or coaching claim.

## Verified Remote-Main Baselines

- UpstreamDrift: `3ecd8c2be0ad25da1548a4b948a93fbfa2268179`.
- AffineDrift: `f7cf332faf57efccf55b708b700c0dc6ce0df056`.
- Tools: `193822fb808047e6f87d06b45a78bad6f5cc0360`.
- UpstreamDrift is the scientific authority. AffineDrift publishes an immutable
  pinned projection. Tools consumes governed records and must not fork physics.

## #8752 Terminal Campaign Evidence

- The single qualified ControlTower container finished successfully:
  `state=complete`, `exit_code=0`, 8 workers under a 4-core cap.
- Started `2026-08-23T16:28:52Z`; finished `2026-08-23T21:59:22Z`.
- Exact source launch commit:
  `13146cdcece879e7156e06e2dca6626c1a54e045`.
- Final record contains 19 registered corners and 38 pathway outcomes.
- Final record SHA-256:
  `e8a7e53701217e4de875a370f7483172f3cfbfb167416b5133ba269b8fef689b`.
- Final archive SHA-256:
  `c54e021eb5ad1e8270ee2a6b473c2cb6d9799583fd41c190b19df9581a6f6d1a`.
- Final manifest SHA-256:
  `0465ee3e91a96b7724d8def0f0f415cfdd8ee7d5cf0b16f89dcbea696b9d144a`.
- The final manifest verifies 1,329 files: 1,327 atomic branch checkpoints,
  the terminal record, and the evidence test. The earlier 1,162-file archive
  was a valid but stale in-progress snapshot and was not promoted.
- Local verified transfer:
  `C:\Users\diete\Campaigns\UpstreamDrift-8752-transfer`.
- Isolated verified extraction:
  `C:\Users\diete\Campaigns\UpstreamDrift-8752-final-extract-20260823-150543`.
- Recovery checkpoints remain ignored and are not reviewer evidence.

## Qualified Scientific Result

- Nominal shaft support: 126/384 matched cells.
- Nine completed nonnominal shaft corners span 80--182 matched cells, or
  -46 to +56 from nominal.
- Shaft grip-damping low and high are retained failures.
- All 18 completed ground corners remain 0/384 matched cells.
- Ground grip-damping high is a retained failure.
- Interpretation: the shaft comparability set is sensitive to declared
  one-at-a-time constitutive bounds. No eligible ground comparison emerged in
  completed corners. These are matching-set results, not speed effects or
  evidence that ground compliance has no dynamical effect.
- Limits: one-at-a-time bounds do not estimate interactions or a joint
  distribution; bounds are not measured participant/equipment properties.

## Current Uncommitted Integration Work

- Added the terminal JSON and its completion-only evidence test.
- Added an immutable execution-provenance companion that binds the exact Git
  source blobs, runtime, resource caps, terminal state, and transfer hashes.
- Extended the record auditor to accept an independently validated archived
  source set, preventing later refactors from relabeling or invalidating the
  historical computation.
- Fixed the evidence test so registered `not_affected` pathways correctly omit
  computed-source hashes while every evaluated pathway remains hash-bound.
- Regenerated the headline PDF/SVG.
- Fixed low/high overplotting with deterministic vertical offsets; both shaft
  failures and every zero-change ground pair are now visible.
- Added exact result language to the headline chapter and conclusions.
- Added completed result claim `PD-CLAIM-306`; converted `PD-CLAIM-304` from
  in-progress to completed-with-retained-failures; retained claim boundaries.
- Rebuilt the 1,094-candidate inventory. All 1,094 candidates are adjudicated;
  305 claims and 40 release claims validate with no open release claims.

## Current Verification

```powershell
python -m pytest tests/research/test_articulated_headline_uncertainty_evidence.py `
  tests/research/test_articulated_headline_record_audit.py `
  tests/research/test_articulated_headline_uncertainty_figure.py `
  tests/research/test_articulated_uncertainty_claim_evidence.py -q
python -m scripts.research.proximal_distal_energy.claim_audit validate
```

- Result: 32 tests passed.
- Claim audit: 1,094/1,094 reviewed, 305 registered claims, no open release
  claims. The harmless Windows pytest temporary-symlink cleanup warning remains.
- The regenerated one-page figure was rendered to PNG and visually inspected;
  labels, markers, retained failures, and coincident ground pairs are legible.

## Immediate Next Actions

1. Run focused Ruff/format, title-case, document, claim-evidence, and record
   audits; then commit and push the terminal #8752 integration slice.
2. Reconcile the branch with current `origin/main`. Preserve the exact #8752
   computation sources and terminal record hashes; prefer current-main versions
   for unrelated manufactured-solution, claim, release, and handoff conflicts.
3. Open a protected PR for the qualified #8752 slice. Do not force-push,
   bypass review, rename required checks, or auto-merge if repository policy
   prohibits it.
4. Rebase/merge #8800 on the integrated authority, remove its remaining
   expiring long-function exception, execute all structural corners, and retain
   the low-height failure and every dynamic failure.
5. Only after #8800: regenerate the full paper/release bundle, visually inspect
   every PDF page, propagate the immutable AffineDrift projection, and update
   the Tools reviewer surface.

## External Boundaries That Must Remain Open

- #8556: no governed participant dataset with synchronized bilateral six-axis
  grip wrenches. Synthetic traces cannot substitute for human validation.
- #9004 and archival/PID/equipment-calibration work remain separate external
  evidence or publication boundaries.
- #8796 remains the stronger finite-static-friction trajectory extension; the
  qualified impulsive perfect-stick bound is not a stick-slip trajectory law.

## Protected Workflow Rules

- Run the repository gate commands in `CLAUDE.md` before pushing.
- Frozen evidence sources may only change through the governed qualification
  workflow; never hand-edit generated registries or release hashes.
- Required check is the aggregate `quality-gate`; do not create another job
  with that name. Do not add debt-ledger entries or weaken scientific/runtime
  thresholds to make CI pass.
