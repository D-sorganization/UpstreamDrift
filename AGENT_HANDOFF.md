# Agent Handoff — UpstreamDrift

Last updated: 2026-08-23

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the single proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns the scientific sources, models, evidence registers, and
  release bundle. AffineDrift is a generated, immutable, revision-pinned public
  projection. Tools owns typed reusable consumers; do not copy its solver or UI
  implementations into this repository or `vendor/ud-tools`.
- UpstreamDrift PR #9005 is protected-squash-merged at remote-main commit
  `46683586ea37fb742d80f0fbe47d25f2acea2cb4`. It supplies the fail-closed
  measured-trajectory source registry for issue #9004.
- The current computational candidate is a 235-page PDF with SHA-256
  `ce51e6fe4f3d9033bf730c0fe2538c72bf88b1b9707f77a7b6385923a1b5fdcf`.
  All pages render, with 194 URI links and 246 outline entries. Archival
  publication remains false because the PDF is untagged and retains Type 3 and
  unembedded font resources.

## Active Articulated Uncertainty Campaign (#8752)

- ControlTower worktree:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\goal-8752-uncertainty`,
  detached at exact scientific launch HEAD `13146cdcece879e7156e06e2dca6626c1a54e045`.
- Container `upstreamdrift-8752-campaign` runs eight CPU-active workers with a
  CPU cap of eight and atomic branch checkpoints. Do not start a duplicate.
- At 2026-08-23 10:08 PDT the container was healthy at approximately 792% CPU
  and 582 MiB RAM. It completed the active 72-branch ground corner and began
  branch 1/72 of the next registered corner.
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

## Measured-Trajectory Qualification (#9004)

- PR #9016 is protected-squash-merged at remote-main commit
  `a2c093aa1478961db20483a4ee89805a132f4ef1`. It
  preregisters eleven primary metrics, four coordinate-frame authorities, two
  events, participant-level holdout, training-only threshold freezing, four
  negative controls, six uncertainty analyses, and missing-as-unavailable
  semantics.
- Its initial optional-stack run exposed an ambient interactive Matplotlib
  backend on the headless runner. The merged correction forces `MPLBACKEND=Agg` at the
  optional-stack job boundary and enforces that contract in
  `tests/ci/test_ci_infrastructure.py`; do not replace this with a blind rerun
  or an application-wide backend override.
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
- Branch `research/9004-governed-ingestion` adds the typed acquisition
  manifest and no-pickle governance gateway. Implementation commit `c9da9b9ca`
  additionally binds each trial to a digest-frozen, source-specific participant
  split and verifies disjoint training, held-out, and adverse cohorts,
  participant membership, and intended use before parsing. Follow-up commit
  `1b7e87c42` also requires the split freeze time to precede artifact creation.
  Source-package and trajectory digests, SI units, processing, four frames, two
  events, channel coverage, and six uncertainty records remain required.
  Thirty-eight focused tests and the 587-artifact release gate pass.
- It is being reconciled with the protected #9016 squash without rewriting
  history. Regenerate the governed manifests, validate, push through every
  hook, and open a full PR. It must remain unusable for human inference until
  the source registry qualifies an actual dataset. The subsequent #9004 slice
  is the deterministic coordinate/event mapping and replay runner.

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
  tests/research/test_measured_trajectory_metric_registration.py -q
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close a scientific child issue whose acceptance
evidence remains unavailable or incomplete.
