# Agent Handoff — UpstreamDrift

Last updated: 2026-08-24

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the single proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns scientific sources, models, evidence registers, and the
  release bundle. AffineDrift is a generated, immutable, revision-pinned public
  projection. Tools owns reusable consumers; do not copy its solver or UI
  implementations into this repository or `vendor/ud-tools`.
- UpstreamDrift remote `main` is
  `228b9a5df130ac46954dd2c9431d795525003c58`. The normalized-claim authority
  from PR #9018 remains an ancestor; issue #8724 is closed.
- The current computational publication is 239 pages with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`,
  194 URI links, and 247 outline entries. All pages render and were inspected.
  Archival qualification remains false because the PDF is untagged and retains
  Type 3 and unembedded font resources.

## Active Structural Campaign and Recovery Boundary (#8800)

- ControlTower prepared clean source commit
  `1bd4d57da7bd257b76b42b3cc19524b283b5f748`, image
  `sha256:b40d91fe2326c5fae288e4a853377fb164aa0a6ba1de62cb28aba15d65500a1e`,
  plan file SHA-256
  `2bddb125492e907acc827e1dbf4cb43b9724e73087679cbf4113bcf96824b120`,
  and plan-contract SHA-256
  `c5cfba35ecafa96054ef8cc872f2e91a9f7855db0b93cfa491f9b18ee3db80f4`.
- The campaign used two workers and two CPUs. Shaft is complete at 48/48
  atomic checkpoints; ground has 45/48. The 93 checkpoint files and completed
  shaft artifacts remain on the ControlTower C: drive under
  `C:\Users\diete\Campaigns\UpstreamDrift-8800-1bd4d57da`.
- Status remains `running`, has no retained execution failures, and explicitly
  reports `release_evidence=false`. Partial checkpoints are resume evidence,
  not scientific release evidence.
- WSL cannot currently mount
  `F:\WSL\ControlTower-SSD\ext4.vhdx`; Windows reported the file corrupted and
  unreadable (`0x80070570`). Both WSL distributions are stopped. Tailscale SSH
  to ControlTower remains available.
- Do not retry WSL, run CHKDSK, repair or mount the VHDX, restart Windows
  services, create a replacement campaign, or alter frozen source or plans
  without an explicit recovery and recoverability decision. Preserve the VHDX
  and C: checkpoints. After recovery, verify identities and resume only the
  three missing ground branches before complete-set and digest qualification.

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

- #8800 propagates height, body-mass, and joint-limit bounds through both
  headline atlases and blocks the final #8752/#8668 audit.
- #8443, #8448, #8449, #8450, #8595, #8668, #8684, and #8796 remain open.
  Verify each issue's exact acceptance evidence before changing state.
- Tools PR #4669 is squash-merged and verified on remote `main` at immutable
  commit `f9730033fd279ba8b4abe03bab2aadd950400b47`. Its downstream run exposed
  an UpstreamDrift provider-mode test-isolation defect; protected PR #9022 at
  exact head `a00e29228c382950015c5eb29c0bd24f8bc2ab08` corrects it and records the
  frozen campaign boundary. Substantive checks pass; runner-dependent checks
  and human review remain pending. Auto-merge is disabled.
- #8358 is leased as `proximal-distal-20260824-r15`. An unpublished detached
  worktree at `UpstreamDrift-worktrees/8358-tools-variation-adapter-prep`
  contains local commits `7e7da8512` (canonical schema gateway) and
  `a55653858` (typed hit, no-impact, failure, and partial-trace evidence). The
  current local work executes every deterministic Tools sample through injected
  serial or batched engine/evidence adapters, preserve row order, and convert
  only declared numerical failures into trial evidence. Thirty-one combined
  focused tests pass, including serial/batched row-equivalence controls.
  Do not push until #9022 merges; then transplant the commits onto refreshed
  `origin/main`, repin Tools, and continue batched and cross-engine parity.
- Tools #4142 remains open until immutable UpstreamDrift consumption and
  requirement-level R10–R15 qualification are complete.
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
