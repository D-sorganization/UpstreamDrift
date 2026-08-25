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
  `c6fc49a7c63549fa6b60afb43594d3c76a226d50`. The normalized-claim authority
  from PR #9018 remains an ancestor; issue #8724 is closed.
- The current computational publication is 239 pages with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`,
  194 URI links, and 247 outline entries. All pages render and were inspected.
  Archival qualification remains false because the PDF is untagged and retains
  Type 3 and unembedded font resources.

## Active Structural Campaign and Recovery Boundary (#8800)

- Separate #8752 constitutive uncertainty completed on ControlTower and was
  transferred at branch commit `2fa6cf886`; it is not yet on remote `main`.
- #8800 uses clean source commit
  `1bd4d57da7bd257b76b42b3cc19524b283b5f748`, image
  `sha256:b40d91fe2326c5fae288e4a853377fb164aa0a6ba1de62cb28aba15d65500a1e`,
  plan file SHA-256
  `2bddb125492e907acc827e1dbf4cb43b9724e73087679cbf4113bcf96824b120`,
  and plan-contract SHA-256
  `c5cfba35ecafa96054ef8cc872f2e91a9f7855db0b93cfa491f9b18ee3db80f4`.
- #8800 used two workers/two CPUs. Nominal shaft is 48/48 and nominal ground
  45/72; the full seven-corner plan is only 93/830, with 737 checkpoints absent.
  The checkpoints and nominal shaft artifact remain under
  `C:\Users\diete\Campaigns\UpstreamDrift-8800-1bd4d57da`.
- The persisted status saying `running` is stale; no campaign process is
  running and `release_evidence=false`. Partial checkpoints are resume
  evidence, not scientific release evidence.
- WSL cannot currently mount
  `F:\WSL\ControlTower-SSD\ext4.vhdx`; Windows reported the file corrupted and
  unreadable (`0x80070570`). Both WSL distributions are stopped. Tailscale SSH
  to ControlTower remains available.
- Do not retry WSL, run CHKDSK, repair or mount the VHDX, restart Windows
  services, create a replacement campaign, or alter frozen source or plans
  without an explicit recovery and recoverability decision. Preserve the VHDX
  and C: checkpoints. After recovery, verify identities, finish 27 nominal
  ground branches, then qualify the remaining registered structural corners.

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
  headline atlases and blocks the final #8752/#8668 audit. It was reopened on
  2026-08-24 after a manual closure lacked a complete campaign, protected PR,
  or remote-main merge.
- #8443, #8448, #8449, #8450, #8595, #8668, #8684, and #8796 remain open.
- PR #9022 is merged as current remote-main commit `76bf6ab1d`. Tools PR #4674
  is merged at immutable commit
  `17474249b9267d0e73a779c1d72f231e7b8de39c`; this is the #8358 gitlink and
  canonical JSON/CSV/HDF5 analysis authority.
- Active #8358 PR #9039 is on `feat/8358-tools-variation-adapter`, updated to
  current `main`. Its Tools gateway, typed trial evidence, serial/batched
  execution, analytical and articulated MuJoCo adapters, fail-closed
  cross-engine comparison, lossless bundles, and canonical scalar, rank/OAT,
  dispersion, and quiet-zone analysis are present. Protected head `1bdc73c60`
  exposed a headless `--version` GUI-import failure after 11,922 other tests
  passed; the current candidate defers GUI/diagnostics imports until launch and
  adds a forced-import-block regression. All five launcher tests pass on Python
  3.11 and 3.13; a new protected run is pending. Keep #8358 open because its
  broader UI, localized-perturbation, and presentation criteria remain unaudited.
- PR #9032 is open at `5edacb13c`; Standard CI passes, but optional-stack has
  a headless `DISPLAY` failure and human review remains absent.
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
python -m pytest -n 0 -q tests/unit/perturbation tests/shared_contracts/test_tools_vendoring.py tests/launchers/test_tools_vendor_authority.py
python scripts/ci/check_architecture_budget.py
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close a scientific child whose narrower evidence
or governed external-data requirement remains incomplete.
