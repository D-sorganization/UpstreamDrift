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

## Active Articulated Uncertainty Campaign (#8752)

- ControlTower worktree:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\goal-8752-uncertainty`,
  running the exact scientific launch revision
  `13146cdcece879e7156e06e2dca6626c1a54e045`.
- Container `upstreamdrift-8752-campaign` uses eight workers within a reversible
  four-core CPU cap. Its workspace and campaign records are bind-mounted from
  the ControlTower C: drive, and restart policy is `on-failure`. Do not start a
  duplicate or modify its frozen source.
- The latest non-mutating probe still reports `running`. Its log completed one
  72/72 ground-atlas pass and reached branch 56/72 of the next atomic pass.
  Treat checkpoint counts as progress only, not as a terminal scientific result.
- Status and logs:
  `C:\Users\diete\Campaigns\UpstreamDrift-8752`. Remote inspection uses the
  pinned Tailscale SSH route into `ControlTower-SSD`; a separate Codex session
  on ControlTower is unnecessary. Loss of this task or Tailscale connectivity
  does not stop the container.
- Runtime: Ubuntu 22.04, Python 3.10.12, NumPy 2.2.4, SciPy 1.15.2, MuJoCo
  3.8.0, and Pinocchio 3.8.0. The cross-CPU canary preserves registered
  discrete decisions and gates at `rtol=2e-8`, `atol=1e-9`.
- At terminal status, audit every expected branch, retained failure, and
  digest. Then integrate `fix/8752-atomic-campaign-checkpoint`. The constitutive
  subcampaign precedes execution of #8800; #8800 still blocks final closure of
  #8752 and #8668. Regenerate claims, figures, paper, and release only after
  both authorities pass.

## Headline Structural Propagation (#8800)

- Worktree `UpstreamDrift-worktrees\8800-headline-structural-propagation`
  wires height, body-mass, and joint-limit authorities into both headline
  atlases. Commits `d972db8ea` and `4bccef56c` contain the governed authority,
  atomic branch executors, campaign CLI, preflight, and focused tests; merge
  `c36bcd223` reconciles the branch with remote main without weakening controls.
- The seven-corner plan retains 84 planned states as 83 feasible states plus
  the declared low-height infeasible state. Shaft and ground checkpoints cover
  48 and 72 atomic branch packs respectively and restore without recomputation.
- This is execution infrastructure, not a scientific result. No #8800 campaign
  has run, no headline estimate has propagated, and no paper or release claim
  may be promoted until all 14 atlas paths and common-support controls pass.

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
