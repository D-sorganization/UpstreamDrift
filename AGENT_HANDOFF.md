# Agent Handoff — UpstreamDrift

Last updated: 2026-08-23

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the single proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns scientific sources, models, evidence registers, and the
  release bundle. AffineDrift is a generated, immutable, revision-pinned public
  projection. Tools owns reusable consumers; do not copy its solver or UI
  implementations into this repository or `vendor/ud-tools`.
- UpstreamDrift remote `main` is
  `3ecd8c2be0ad25da1548a4b948a93fbfa2268179`, the protected squash merge of
  PR #9019. It includes PR #9018 and the post-adjudication handoff correction.
- The current computational publication is 239 pages with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`,
  194 URI links, and 247 outline entries. All pages render and were inspected.
  Archival qualification remains false because the PDF is untagged and retains
  Type 3 and unembedded font resources.

## Completed Constitutive Uncertainty Campaign (#8752)

- The single ControlTower campaign completed at
  `2026-08-23T21:59:22Z` with exit code 0. It used eight workers under a
  four-core/96 GB cap at exact source revision
  `13146cdcece879e7156e06e2dca6626c1a54e045`.
- The final 1,329-file transfer manifest verifies 1,327 atomic checkpoints,
  the terminal record, and its evidence test. Record SHA-256 is
  `e8a7e53701217e4de875a370f7483172f3cfbfb167416b5133ba269b8fef689b`;
  archive SHA-256 is
  `c54e021eb5ad1e8270ee2a6b473c2cb6d9799583fd41c190b19df9581a6f6d1a`.
  The earlier in-progress snapshot was verified but not promoted.
- Nine completed nonnominal shaft corners span 80--182 matched cells, or
  -46 to +56 from nominal 126/384. Both grip-damping corners are retained
  failures. All 18 completed ground corners remain 0/384; high grip damping is
  a retained ground failure.
- This rejects invariance of the shaft comparability set over the registered
  one-at-a-time constitutive bounds. It is not a speed effect, joint parameter
  distribution, participant/equipment calibration, or coaching result. The
  empty ground matching set does not mean support compliance has no dynamics.
- `articulated_headline_execution_provenance.json` binds the exact Git blobs,
  runtime, resource caps, terminal state, and transfer hashes. The auditor can
  validate that archived source set after behavior-preserving refactors without
  relabeling the historical computation.
- The corrected PDF/SVG offsets coincident low/high markers so both retained
  failures and every zero-change ground pair remain visible. Claim
  `PD-CLAIM-314` records the bounded result. #8800 still blocks final closure of
  #8752/#8668 and the full publication/release regeneration.

## Headline Structural Propagation (#8800)

- Worktree `UpstreamDrift-worktrees\8800-headline-structural-propagation`
  wires height, body-mass, and joint-limit authorities into both headline
  atlases. Commits `d972db8ea` and `4bccef56c` contain the governed authority,
  atomic branch executors, campaign CLI, preflight, and focused tests; merge
  `c36bcd223` reconciles the branch with remote main without weakening controls.
- The seven-corner plan retains 84 planned states as 83 feasible states plus
  the declared low-height infeasible state. Shaft and ground checkpoints cover
  48 and 72 atomic branch packs respectively and restore without recomputation.
- Terminal #8752 integration is complete. Ground execution, scaled-authority
  regeneration, shaft record assembly, claim registration, and the #8800
  preflight builder are decomposed under the repository budgets. All seven
  temporary #8752/#8800 architecture exceptions are removed and the changed-file
  architecture gate passes without an exception.
- All seven structural authorities and the standalone nominal authority were
  regenerated from the refactored source. Their numerical arrays are unchanged;
  the JSON authority/source digests and the governed propagation plan now bind
  the current implementation. A stale-resume regression prevents a terminal
  authority row from being reused when its source map differs. Fifty-two focused
  authority, campaign, plan, checkpoint-contract, and shaft tests pass locally;
  the broader 192-test structural/headline suite also passes.
  The real short native-engine checkpoint restart passed on ControlTower at
  commit `f276e779fb1ed28950ae42442ead4f852a3e7c76` in the pinned #8752
  scientific image. DeskComputer still resolves the unrelated PyPI `pinocchio`
  package, so native execution qualification remains a ControlTower gate.
- ControlTower is the approved execution host and is reachable noninteractively
  from DeskComputer over direct Tailscale SSH. On 2026-08-23 it reported an
  Alienware Aurora Ryzen Edition R14 with a 16-core/32-thread Ryzen 9 5950X,
  128 GB RAM, an idle 24 GB RTX 3090, and adequate disk space. Codex does not
  need to run interactively on ControlTower: orchestration, monitoring, and
  evidence retrieval can occur from DeskComputer over SSH.
- Do not execute in ControlTower's primary `UpstreamDrift` checkout: it is old,
  on an unrelated branch, and dirty. Preserve the detached #8752 worktree and
  its untracked terminal evidence. After the #8800 implementation is committed,
  pushed, and passes preflight, create a new uniquely named worktree pinned to
  that exact remote commit. Run it detached with atomic checkpoints, separate
  stdout/stderr and exit-code files, a conservative initial CPU-worker cap, and
  periodic SSH status collection. The current model path is CPU-bound unless a
  separately verified GPU backend is introduced; the RTX 3090 alone does not
  make the existing solver GPU-accelerated.
- The first #8800 launch attempt exposed two fail-closed portability defects
  before scientific execution: the #8752 image lacked Matplotlib, and Windows
  text-mode authority generation hashed CRLF working-tree bytes that Git stored
  as LF. The failed container log is preserved under
  `C:\Users\diete\Campaigns\UpstreamDrift-8800-f276e779f\logs`. A derived
  runtime image pins Matplotlib 3.10.8 at image ID
  `sha256:b40d91fe2326c5fae288e4a853377fb164aa0a6ba1de62cb28aba15d65500a1e`.
  Authority and campaign writers now emit canonical LF bytes, with regression
  assertions. All seven authorities, the plan, and the 313-claim evidence
  manifest were regenerated and locally validated at execution commit
  `8a20df8fe340c4c8b9e65ca5e575bbbcabce0f96`.
- The exact-commit campaign in container `upstreamdrift-8800-8a20df8fe`
  completed all 48 nominal shaft branch checkpoints, then failed retained during
  record assembly because the checkpoint adapter called the refactored
  single-context shaft record contract with nine positional arguments. Docker
  exhausted its three declared retries and exited 1 without OOM. The status,
  logs, and source-bound checkpoints remain under
  `C:\Users\diete\Campaigns\UpstreamDrift-8800-8a20df8fe`; no result, figure,
  or release evidence was emitted.
- The active worktree now constructs the exact `shaft._RecordContext` and has a
  regression that rejects variadic mocks masking future interface drift. The
  governed propagation plan was regenerated because its source/test digests
  changed. Consequently, the 48 historical checkpoints are retained failure
  evidence but cannot be relabeled or resumed under the corrected plan. Commit,
  push, validate in Linux, and prepare a uniquely named clean detached worktree
  before any relaunch. Do not restart the failed container or start a corrected
  run until the worker/CPU plan is explicitly selected.
- This is execution infrastructure, not a scientific result. One incomplete
  #8800 attempt has run, but no complete atlas path or headline estimate has
  propagated, and no paper or release claim may be promoted until all 14 paths
  and common-support controls pass.

## Normalized Claim Adjudication (#8724)

- Protected PR #9018 is merged on remote `main` as
  `9e220712025564caf0ac5201a0ddcf69dd98299e`; #8724 is closed.
- The authority contains 1,100 reviewed narrative candidates and 303 material
  claims: 283 supported only within declared estimands and boundaries, five
  inconclusive, 15 untested, and zero contradicted. Supported claims may report
  null, mixed, or adverse findings; the count does not imply theory survival.
- The schema and snapshot-locked migration require explicit normalized
  outcomes, typed locators, reciprocal mappings, falsifiers, source digests,
  reasons, reviewer identity, dates, and supported-scope contradiction checks.
  Unfamiliar claims fail closed.
- Reviewer JSON, CSV, and paper tables separately report normalized outcome,
  evidence tier, source independence, model tier, unresolved replication, and
  claim-family source concentration. Model evidence is never promoted to human
  validation.
- Deterministic release evidence covers 2,130 evidence references, 319 local
  artifacts, 78 external URLs, and a 592-artifact release bundle. The merged
  head passed 63 focused tests plus standard, optional-stack, publication,
  security, title-case, file-size, architecture, Ruff, MyPy, and Bandit gates.

## Measured-Trajectory and Human-Evidence Boundaries

- PR #9017 is merged at
  `ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`. It provides fail-closed
  participant split, processing, frame-transform, and event-detector
  authorities for #9004.
- #9004 remains open because no qualifying governed participant trajectory
  dataset or held-out human outcome is registered. Simscape exports, fixtures,
  tutorials, GolfDB labels, and launch-monitor records are not substitutes.
- #8556 remains externally blocked by the absence of governed synchronized
  bilateral six-axis grip-wrench participant data. Synthetic traces and
  paper-level curves must never substitute for human validation.

## Other Active Dependencies

- The current constitutive subcampaign is the active #8752 slice. #8800 then
  propagates height, body-mass, and joint-limit bounds through both headline
  atlases and blocks the final #8752/#8668 audit.
- #8443, #8448, #8449, #8450, #8595, #8668, #8684, and #8796 remain open.
  Verify each issue's exact acceptance evidence before changing state.
- Tools #4142 remains open for requirement-level R10–R15 qualification and
  immutable UpstreamDrift consumption. Tools #4430 is complete.
- AffineDrift #3930 remains downstream of the qualified UpstreamDrift release;
  do not project a moving or partial campaign.
- #8963 architecture debt remains separate from the frozen campaign source.
  Do not merge or regenerate source-bound spatial evidence until campaign
  integration removes the source-lock conflict.

## Scientific Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- Distinguish energy transfer, momentum redistribution, joint work, constraint
  forces, and clubhead speed. Preserve falsifiers, adverse cases,
  identifiability limits, uncertainty, countermodels, and unavailable states.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Human review is required.
- Never force-push, admin-merge, bypass hooks/checks, add quarantine debt, or
  edit `vendor/ud-tools`.
- Use TDD, DbC, DRY, and LoD. Edit canonical sources and regenerate governed
  artifacts. Use title case for document headings and captions.
- Verify exact PR head, reviews, checks, merge SHA, remote-main ancestry, and a
  clean worktree before reporting completion.

## Focused Validation

```powershell
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close a scientific child whose narrower evidence
or governed external-data requirement remains incomplete.
