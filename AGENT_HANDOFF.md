# Agent Handoff — UpstreamDrift

Last updated: 2026-08-24

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns scientific sources, models, evidence registers, and the
  release bundle. AffineDrift is an immutable revision-pinned public
  projection. Tools owns reusable analysis authorities; do not copy its
  algorithms into this repository or edit `vendor/ud-tools` by hand.
- Remote `main` is `a5714721a2d94b07ef75fc18c0a092ce141fa1f8`.
- The computational paper is 239 pages with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`.
  Archival qualification remains false because the PDF is untagged and retains
  Type 3 and unembedded font resources.

## Numeric Claim Authority (#8918)

- Protected PR #9042 merged as current remote `main`; issue #8918 is closed.
- All 303 material claims are covered. The 124 numeric claims contain 380
  numeric literals, each bound to a reviewed JSON Pointer, transform, scope,
  and tolerance. Exact statement digests and literal inventories fail closed.
- Evidence scopes distinguish 172 semantically matched local JSON values, 144
  registered values not independently recomputed, 57 reported external values,
  and seven protocol or notation values.
- Four representative headline tests independently recompute planar, spatial,
  articulated-shaft, and finite-ground quantities from committed arrays.
  Pointer agreement is not physical validation, and cross-engine controls must
  remain close but nonidentical.
- Release validation covers 2,232 evidence references, 321 local artifacts, 78
  external URLs, and a 600-artifact computational bundle.

## Tools Variation Integration (#8358 / Tools #4142)

- Tools PR #4674 is merged at immutable commit
  `17474249b9267d0e73a779c1d72f231e7b8de39c`; it owns canonical
  JSON/CSV/HDF5 variation persistence and reusable analysis.
- UpstreamDrift PR #9039 is open with human-requested squash auto-merge. Its
  remote branch consumes that exact Tools pin through a thin gateway and adds
  typed trial evidence, serial/batched execution, analytical double-pendulum
  and bilateral articulated-MuJoCo adapters, fail-closed cross-engine
  comparison, lossless bundles, and canonical scalar, rank/OAT, dispersion,
  geometry, and quiet-zone consumption.
- Reconciliation worktree:
  `UpstreamDrift-worktrees/9039-protected-reconcile`; local branch:
  `fix/9039-protected-reconcile`; push destination:
  `feat/8358-tools-variation-adapter` by normal fast-forward only.
- The current-main merge is resolved locally. Shared adapter contracts remove
  the reported LoD and DRY growth; 56 focused adapter/executor tests, Ruff,
  LoD, DRY, architecture, and file-size gates pass. The earlier pendulum
  entry-point failure is fixed by current `main`. Push only after focused MyPy
  and the pre-push hooks pass; then inspect the exact protected CI state once.
- This integration is synthetic, model-conditional infrastructure. It does not
  establish participant mechanics, anatomy, physiology, equipment calibration,
  injury risk, coaching strategy, or a universal speed benefit.

## Completed and Incomplete Campaigns

- #8752 completed on ControlTower at source commit
  `13146cdcece879e7156e06e2dca6626c1a54e045`; terminal evidence is committed on
  `origin/research/8752-articulated-uncertainty` at
  `2fa6cf8861eeaf7ae111dd8dd18c4053a9f82e65`. The computation completed, but
  terminal publication integration is not yet an ancestor of remote `main`.
- #8800 did not complete. The registered campaign requires 830 atomic
  checkpoints across 83 feasible states: 332 shaft and 498 ground. Current
  evidence is 93/830: nominal shaft 48/48 and nominal ground 45/72. No campaign
  process is running; persisted `state=running` is stale and
  `release_evidence=false`.
- Preserve #8800 source, plan identities, 93 checkpoints, and the corrupted
  ControlTower WSL VHDX. Do not claim three missing branches or a 45/48 ground
  denominator; 737 registered checkpoints remain absent.

## Human-Evidence Boundaries

- #9004 remains open because no governed participant trajectory dataset or
  held-out human outcome is registered. Simscape exports, fixtures, tutorials,
  GolfDB labels, and launch-monitor records are not substitutes.
- #8556 remains externally blocked by the absence of governed synchronized
  bilateral six-axis grip-wrench participant data.
- Synthetic evidence never substitutes for unavailable human validation.

## Scientific and Review Rules

- Distinguish energy transfer, momentum redistribution, joint work, constraint
  forces, and clubhead speed. Preserve falsifiers, adverse cases,
  identifiability limits, uncertainty, countermodels, and unavailable states.
- PRs target `main`; use full PRs, never drafts. Human review is required.
- Never force-push, admin-merge, bypass hooks/checks, add quarantine debt, or
  edit generated/vendor authority by hand.
- Use TDD, DbC, DRY, and LoD. Edit canonical sources, regenerate governed
  artifacts, and use Title Case for document headings and captions.
- Run resource-intensive tests serially on DeskComputer. The local runner fleet
  is deliberately drained; do not restart it while reconciling #9039.
- Verify exact PR head, reviews, checks, merge SHA, remote-main ancestry, and a
  clean worktree before reporting completion.

## Focused Validation

```powershell
python scripts/ci/check_lod.py src --baseline scripts/ci/lod_baseline.txt
python scripts/ci/check_dry_duplication_gate.py
python -m pytest -n 0 -q tests/unit/perturbation tests/shared_contracts/test_tools_provider_contracts.py tests/launchers/test_pendulum_simulator_entrypoint.py
python -m mypy src/shared/python/perturbation --follow-imports=skip
python -m ruff check src/shared/python/perturbation tests/unit/perturbation
python -m ruff format --check src/shared/python/perturbation tests/unit/perturbation
python scripts/ci/check_architecture_budget.py
python scripts/ci/check_file_size_budget.py
python scripts/check_document_title_case.py --changed-from origin/main
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate --source-revision (git rev-parse HEAD) --publication-profile computational
```

Passing shared gates does not close a scientific child whose narrower evidence
or governed external-data requirement remains incomplete.
