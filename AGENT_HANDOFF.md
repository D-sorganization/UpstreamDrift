# Agent Handoff — UpstreamDrift

Last updated: 2026-08-21

This file is current operational state, not a changelog. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Verified Cross-Repository Delivery

- UpstreamDrift PR #8954 merged through protection at
  `81cc731d0dd19367b00cd819be5677ab157ce125`, verified on remote `main`.
- It pins Tools revision
  `1664d806df8a2c7b184d2d3fbcea93b714caaee5` and verifies the qualified
  18-case rotating-base contract without copying its solver or catalog logic.
- Standard CI, shared consumer contracts, docs, SPEC freshness, unit
  aggregation, and title capitalization passed. The wheel build passed; two
  downstream smoke jobs were cancelled after merge and received one targeted
  failed-job rerun request. Inspect that exact run rather than duplicating it.
- Tools PR #4619 remains an ancestor of current Tools `main`; Tools issue #4430
  closed with the verified UpstreamDrift consumer evidence.

## Critical Cross-Engine Evidence Repair (#8909)

- Worktree: `UpstreamDrift-worktrees/8909-real-pinocchio-parity`.
- Branch: `fix/8909-real-pinocchio-parity`; active Codex lease is renewed.
- The distributed-grip release falsely recorded `pinocchio: 0.1` and exact-zero
  parity because `native_dynamics_operator` silently substituted MuJoCo when
  the unrelated PyPI `pinocchio` package was imported.
- The repair removes that fallback, requires robotics Pinocchio version 2.6 or
  newer plus its native API, rejects an identically zero engine comparison,
  and audits every committed cross-engine artifact's recorded version.
- The first genuine 576-trajectory rerun exposed a real failed numerical gate:
  twelve stick rows had rank ten, the normal-equation condition number exceeded
  1e17, and the Pinocchio residual reached 1.35e-10 m/s. A mass-whitened
  rank-revealing SVD reduces the full 144-cell residual to 4.61e-15 m/s without
  relaxing the registered tolerance.
- Genuine MuJoCo 3.8.0 and Pinocchio 3.8.0 inertia, contact projection, forward
  contact, and distributed-grip evidence have been regenerated. Every gate
  passes with nonzero cross-engine differences. The distributed maxima are
  3.84e-13 trajectory, 1.44e-10 force, and 3.99e-12 stick velocity relative.
- Twenty-nine focused tests pass. Claim governance reports 1,068/1,068
  candidates adjudicated, 295 claims, zero open release claims, and a valid
  2,103-reference evidence manifest.
- Commit `898033064c3f2c45930bbea722a744406423af51` checkpoints the coherent
  repair. It is being merged with current `origin/main`; resolve only current
  handoff/SPEC chronology, then regenerate source-revision-bound release files.
- Remaining work: rebuild and inspect the PDF; rewrite release manifests and
  checksums; run publication/repository gates; push a full PR; shepherd human
  review and protected merge; verify remote-main ancestry before closing #8909.

## Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` is an active checkpointed 19-corner campaign, not an
  orphan. Seven corners were recorded at the 2026-08-21 05:03 checkpoint.
- It may own up to 20 workers. Do not kill workers individually, edit campaign
  sources, or start a second campaign. Retained numerical failures are data.
- #8910 remains critical: current manufactured inverse-dynamics,
  action-reaction, and conservation controls are tautological or hardcoded.
  Address #8910 after #8909 is protected.

## Scientific Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- #8556 remains externally data-gated: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Never substitute
  synthetic traces for human validation.
- #8910 invalidates manufactured-solution closure language until independent
  operators and actual drift are qualified.
- Exact zeros require an analytic identity or explicit degeneracy control; they
  are not automatically evidence of unusually good cross-engine parity.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Codex must not enable
  auto-merge; human approval is mandatory. Preserve explicit owner actions
  without treating them as permission to bypass protection.
- Use TDD, DbC, DRY, and LoD. Never force-push, admin-merge, bypass hooks or
  checks, add quarantine debt, or edit `vendor/ud-tools`.
- Research evidence is hash-pinned. Use governed generators rather than
  hand-editing JSON, NPZ, figures, manifests, checksums, or claim records.
- Use title case for document headings and captions.

## Focused #8909 Validation

```bash
python3 -m pytest -q -n 0 \
  tests/research/test_articulated_inertia_cross_engine.py \
  tests/research/test_articulated_contact_projection.py \
  tests/research/test_articulated_forward_contact.py \
  tests/research/test_articulated_distributed_atlas.py
python3 -m ruff check \
  scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py \
  scripts/research/proximal_distal_energy/articulated_forward_integration.py \
  scripts/research/proximal_distal_energy/articulated_distributed_atlas.py \
  scripts/research/proximal_distal_energy/register_articulated_inertia_claims.py \
  tests/research/test_articulated_inertia_cross_engine.py \
  tests/research/test_articulated_distributed_atlas.py
python3 -m scripts.research.proximal_distal_energy.claim_audit validate
python3 -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
```

## Release and Repository Gates

```bash
python3 -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate \
  --source-revision "$(git rev-parse HEAD)" --publication-profile computational
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/check_doc_size_budget.py
python3 scripts/ci/check_architecture_budget.py
python3 -m ruff check .
python3 -m ruff format --check .
```

Passing common gates does not close a child issue whose scientific criteria
remain unmet. Verify exact PR head, review decision, checks, merge SHA, and
remote-main ancestry before reporting protected completion.
