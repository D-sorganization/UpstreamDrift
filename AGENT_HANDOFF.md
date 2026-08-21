# Agent Handoff — UpstreamDrift

Last updated: 2026-08-21

This file is current operational state, not a changelog. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Protected Delivery in Progress

### Tools Rotating-Base Consumer

- UpstreamDrift PR #8954 merged through protection at
  `81cc731d0dd19367b00cd819be5677ab157ce125`, verified on remote `main`.
- It pins Tools remote-main revision
  `1664d806df8a2c7b184d2d3fbcea93b714caaee5` and verifies the qualified
  18-case rotating-base contract without copying its solver.
- Standard CI, shared consumer contracts, documentation, SPEC freshness, unit
  aggregation, and title-capitalization checks passed. The package artifact
  job was still running after merge and was not a required protection gate.
- Tools PR #4619's merge remains an ancestor of current Tools `main`; companion
  issue Tools #4430 closed with the verified UpstreamDrift consumer evidence.
- Worktree: `UpstreamDrift-worktrees/4430-rotating-base-consumer`.

### Critical Cross-Engine Evidence Repair (#8909)

- Worktree: `UpstreamDrift-worktrees/8909-real-pinocchio-parity`.
- Branch: `fix/8909-real-pinocchio-parity`; lease held by Codex.
- The distributed-grip release falsely recorded `pinocchio: 0.1` and exact-zero
  parity because `native_dynamics_operator` silently substituted MuJoCo when
  the unrelated PyPI `pinocchio` package was imported.
- The in-flight repair removes that fallback, requires robotics Pinocchio
  version 2.6 or newer plus its native dynamics API, rejects an identically
  zero numerical comparison, and audits every committed artifact's version.
- Four focused regression tests pass on Windows, where the impostor is now
  rejected. A genuine WSL Pinocchio 3.8.0 versus MuJoCo 3.8.0 smoke atlas
  passed with nonzero trajectory, force, and stick-projection discrepancies.
- The first genuine 576-trajectory regeneration exposed a real failed gate:
  redundant stick rows made the Schur system condition number exceed 1e17 and
  left a 1.35e-10 m/s projection residual. A mass-whitened rank-revealing SVD
  now reduces the full 144-case stick residual to 4.61e-15 m/s without relaxing
  the gate; focused analytic, redundancy, and degeneracy tests pass.
- The corrected genuine-engine atlas now passes every gate with nonzero maximum
  trajectory, force, and stick-velocity discrepancies of 3.84e-13, 1.44e-10,
  and 3.99e-12 relative. Inertia and contact-projection authorities also pass
  after regeneration; forward contact is running in WSL session `90002`.
- After forward-contact regeneration: regenerate its figure, finish the
  chapter's exact metrics, rerun claim registration and release qualification,
  rebuild/inspect the PDF, update SPEC, commit, push, open a full PR, and
  shepherd protected review.

### Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` is an active checkpointed 19-corner campaign, not an
  orphan. Seven corners were recorded at the 2026-08-21 05:03 checkpoint.
- It may own up to 20 workers. Do not kill workers individually, edit campaign
  sources, or start a second campaign. Retained numerical failures are data.
- #8910 remains critical: the current manufactured-solution inverse-dynamics,
  action-reaction, and conservation controls are tautological or hardcoded.
  Do not close #8752 on those controls. Address #8910 after #8909 is protected.

## Scientific Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- #8556 remains externally data-gated: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Never substitute
  synthetic traces for that human validation.
- #8909 invalidates the current distributed-grip parity claim until regenerated
  with two genuine engines. #8910 invalidates current manufactured-solution
  closure language until independent operators and actual drift are qualified.
- Exact zeros require an analytic identity or an explicit degeneracy check;
  they are not automatically evidence of unusually good cross-engine parity.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Codex must not enable
  auto-merge; human approval is mandatory. Preserve explicit owner actions
  already recorded on GitHub without treating them as permission to bypass.
- Before issue work, run the Repository_Management claim check and post a lease.
- Use TDD, DbC, DRY, and LoD. Never force-push, admin-merge, bypass hooks or
  checks, add quarantine debt, or edit `vendor/ud-tools`.
- Research evidence is hash-pinned. Regenerate through governed commands rather
  than editing source digests, figures, JSON, NPZ, manifests, or claims by hand.
- Use title case for document headings/captions and run the title-case audit.

## Focused #8909 Validation

```bash
python3 -m pytest \
  tests/research/test_articulated_inertia_cross_engine.py \
  tests/research/test_articulated_distributed_atlas.py -q -n 0
python3 -m ruff check \
  scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py \
  scripts/research/proximal_distal_energy/articulated_forward_integration.py \
  scripts/research/proximal_distal_energy/articulated_distributed_atlas.py \
  tests/research/test_articulated_inertia_cross_engine.py \
  tests/research/test_articulated_distributed_atlas.py
python3 -m ruff format --check \
  scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py \
  scripts/research/proximal_distal_energy/articulated_forward_integration.py \
  scripts/research/proximal_distal_energy/articulated_distributed_atlas.py \
  tests/research/test_articulated_inertia_cross_engine.py \
  tests/research/test_articulated_distributed_atlas.py
```

## Release and Repository Gates

```bash
python3 -m scripts.research.proximal_distal_energy.claim_audit validate
python3 -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python3 -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate \
  --source-revision "$(git rev-parse HEAD)" --publication-profile computational
python3 scripts/check_document_title_case.py
python3 scripts/check_doc_size_budget.py
python3 scripts/ci/check_architecture_budget.py
python3 -m ruff check .
python3 -m ruff format --check .
```

Passing common gates does not close a child issue whose scientific acceptance
criteria remain unmet. Verify the exact PR head, review decision, checks, merge
SHA, and remote-main ancestry before reporting protected completion.
