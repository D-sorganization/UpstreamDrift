# Agent Handoff — UpstreamDrift

Last updated: 2026-08-23

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the single proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns the scientific sources, models, evidence registers, and
  release bundle. AffineDrift is a generated, immutable, revision-pinned public
  projection. Tools owns typed reusable consumers; do not copy its solver or UI
  implementations into this repository or `vendor/ud-tools`.
- UpstreamDrift PR #9017 is protected-squash-merged at remote-main commit
  `ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`. It completes the current
  fail-closed measured-trajectory ingestion boundary for issue #9004,
  including immutable participant split, processing, frame-transform, and
  event-detector authorities. Issue #9004 remains open because no qualifying
  governed participant trajectory dataset or held-out human outcome exists.
- The current computational candidate is a 239-page PDF with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`.
  All pages render, with 194 URI links and 247 outline entries. Archival
  publication remains false because the PDF is untagged and retains Type 3 and
  unembedded font resources.

## Active Articulated Uncertainty Campaign (#8752)

- ControlTower worktree:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\goal-8752-uncertainty`,
  detached at exact scientific launch HEAD `13146cdcece879e7156e06e2dca6626c1a54e045`.
- Container `upstreamdrift-8752-campaign` retains eight worker processes and
  atomic branch checkpoints. At 11:21 PDT its reversible container CPU cap was
  reduced from eight cores to four over Tailscale, without a restart or source
  change, to coexist with protected CI. The running container then used about
  four cores and 589 MiB. Do not start a duplicate.
- At 2026-08-23 12:53 PDT one registered ground corner had completed all 72
  branches and the next corner had atomically completed branch 21/72;
  `status.json` still reported `running`. The container remained within its
  four-core cap at about 394% CPU and 585 MiB. The container reports an
  `on-failure` restart policy, and its writable workspace and campaign outputs
  are bind-mounted from the ControlTower C: drive. Remote control is available
  through the pinned Tailscale SSH route into the `ControlTower-SSD` WSL
  distribution; a separate Codex session on ControlTower is not required. Do
  not infer total-campaign
  completion from one corner's branch counter.
- Runtime: Ubuntu 22.04, Python 3.10.12, NumPy 2.2.4, SciPy 1.15.2, MuJoCo
  3.8.0, and Pinocchio 3.8.0. The cross-CPU canary preserves every discrete
  decision and registered gate at `rtol=2e-8`, `atol=1e-9`.
- Status/logs: `C:\Users\diete\Campaigns\UpstreamDrift-8752`. Inspect those
  records and the named container before recovery. Never restart a zero-exit
  completed container. Partial checkpoints are execution evidence, not release
  evidence.
- At terminal completion: audit every expected branch and digest, integrate
  `fix/8752-atomic-campaign-checkpoint`, execute #8800, regenerate claims,
  figures, paper, and release bundle, then refresh the AffineDrift projection.

## Normalized Claim Adjudication (#8724)

- Worktree:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\claim-adjudication-8724`;
  branch `research/8557-claim-adjudication-8724`.
- A normal, non-rebased merge of current remote `main`
  (`ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`) is in progress with all merge
  conflicts resolved and staged. Do not abort, rebase, force-push, or replace
  the staged current-main changes.
- The current paper authority contains 1,100 fully reviewed narrative
  candidates and 303 material claims: 283 supported only at their declared
  estimands and boundaries, five inconclusive, 15 untested, and zero
  contradicted. A zero contradicted count does not erase supported claims that
  accurately report null, mixed, or adverse findings.
- `migrate_claim_adjudication_v2.py` is locked to paper digest
  `7407e8f00842ecdf95769d65ac7d2fe3f8d495cb0d11d405640e7582e6b8560a`
  and contains an exhaustive explicit outcome set for all 303 claim IDs. Any
  unfamiliar claim fails rather than defaulting to supported. The 22 generated
  reviewer-table candidates are explicitly enumerated as editorial projections
  of existing claims.
- The generated JSON, CSV, and paper chapter separate normalized outcome,
  evidence tier, source independence, model tier, unresolved replication, and
  claim-family source concentration. No axis promotes model evidence to human
  validation. The registry validator also rejects a supported outcome whose
  detailed state leaves human validation, reimplementation, or a hypothesis
  open without an explicitly narrower adjudication reason.
- Current focused evidence: the claim registry, reviewer summary, release
  review, 2,130-reference evidence manifest, and 592-artifact release bundle
  validate deterministically. All 63 focused claim, migration, evidence,
  publication, PDF, release, and document-governance tests pass. Ruff, title
  capitalization, file-size, and changed-source architecture gates pass with
  no exception or quarantine. The 239-page PDF was inspected in full and
  passes the computational publication profile; the documented archival gaps
  remain fail-closed. Protected full PR #9018 is open for #8724; preserve its
  review and required-check gates, fix only actionable failures, and verify the
  squash commit on remote `main` before closing the issue.

## Measured-Trajectory Qualification (#9004)

- PR #9016 is protected-squash-merged at remote-main commit
  `a2c093aa1478961db20483a4ee89805a132f4ef1`. It
  preregisters eleven primary metrics, four coordinate-frame authorities, two
  events, participant-level holdout, training-only threshold freezing, four
  negative controls, six uncertainty analyses, and missing-as-unavailable
  semantics.
- Its initial optional-stack run exposed an ambient interactive Matplotlib
  backend on the headless runner. The merged correction forces
  `MPLBACKEND=Agg` at the optional-stack job boundary and enforces that contract
  in `tests/ci/test_ci_infrastructure.py`. All API, Pinocchio, and unit steps
  passed on the corrected run; its large post-job cache upload was still
  finalizing at 11:11 PDT. Do not replace this with a blind rerun or an
  application-wide backend override.
- The metric contract is deliberately fail-closed: `execution_ready=false`,
  `human_inference=false`, `bilateral_wrench=false`, and
  `results_status=not_run_no_authority` until a governed participant dataset is
  registered. Synthetic controls qualify software discrimination only.
- The local census found no governed participant golf motion capture. Simscape
  exports are circular simulation evidence; pipeline fixtures, OpenSim
  tutorials, GolfDB labels, and launch-monitor records cannot substitute for
  body-and-club trajectories.
- KIT motion 1319 and GolfPose remain candidate sources only. Neither currently
  has a registered local digest plus verified reuse and required calibration,
  club, participant, and event fields. Never infer authority from a filename,
  screenshot, or visual resemblance.
- PR #9017 protected-squash-merged at
  `ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`; all required exact-head checks
  and human review passed. The boundary must remain unusable for human
  inference until the source registry qualifies an actual dataset. The
  subsequent #9004 slice is the deterministic coordinate/event mapping and
  replay runner.

## Other Active Dependencies

- #8556 remains externally blocked: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Synthetic traces
  must never substitute for human validation.
- #8752 precedes #8800. #9004 precedes #8450. The canonical dependency and
  acceptance ledger remains issue #8557.
- #8724, #8443, #8448, #8449, #8450, #8595, #8668, #8684, and #8796 remain
  open. Verify exact acceptance evidence before changing issue state.
- AffineDrift must remain pinned to its existing immutable UpstreamDrift release
  until terminal scientific changes are merged and fully regenerated.
- Tools PR #4646 remains a separate visual-evidence timeout repair. Do not alter
  shared runners or weaken its assertions to obtain a pass.

## Scientific Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- Distinguish energy transfer, momentum redistribution, joint work, constraint
  forces, and clubhead speed. A sequence, correlation, or model optimum does not
  establish a human mechanism.
- Preserve falsifiers, adverse cases, identifiability limits, uncertainty,
  countermodels, and explicit unavailable states in every new result.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Human review is required.
- Never force-push, admin-merge, bypass hooks/checks, add quarantine debt, or
  edit `vendor/ud-tools`.
- Use TDD, DbC, DRY, and LoD. Generate manifests and publication artifacts with
  governed scripts; do not hand-edit generated outputs.
- Use title case for document headings and captions.
- Verify exact PR head, review, checks, merge SHA, remote-main ancestry, and a
  clean worktree before reporting protected completion.

## Focused Validation

```powershell
python -m pytest tests/research/test_measured_trajectory_source_registry.py `
  tests/research/test_measured_trajectory_metric_registration.py `
  tests/research/test_measured_trajectory_ingestion.py -q
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close a scientific child issue whose acceptance
evidence remains unavailable or incomplete.
