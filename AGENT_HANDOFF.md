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

## Verified Cross-Engine Evidence Repair (#8909)

- PR #8960 merged through protection at
  `fdf2eb0d1e37db8f5b58109dbbf224a519538170`, verified as current remote
  `main`; issue #8909 is closed.
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
- Ninety-five focused scientific/publication tests pass. Claim governance
  reports 1,068/1,068 candidates adjudicated, 295 claims, zero open release
  claims, and a valid 2,103-reference evidence manifest.
- Commit `898033064c3f2c45930bbea722a744406423af51` checkpoints the coherent
  repair; merge commit `f86e8f8d6e8405c51c1e484d9ac1a36dcf75f732`
  reconciles current `origin/main` without unresolved conflicts.
- The optimized 233-page PDF was visually inspected on every page. The
  computational profile passes with 192 valid URI links, 246 outline entries,
  fast-web linearization, and no render errors. The archival profile remains
  fail-closed on the disclosed untagged/Type 3/unembedded-font gaps.
- The open-release validator passes for 571 artifacts with no mismatches.
- Unqualified repository-wide `pytest` currently fails during collection in
  the pre-existing `src/shared/python/sidekick/tests/conftest.py` because it
  imports nonexistent `utils.path_helpers`. Do not hide that baseline defect;
  use the dependency-scoped protected lanes and track a separate repair.

## Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` is an active checkpointed 19-corner campaign, not an
  orphan. Eight corners and 15 of 30 pathway evaluations were recorded by the
  2026-08-21 08:46 checkpoint; the next ground pathway was active when last
  inspected. Completed pathway rows and atomic ground-branch checkpoints are
  restartable. Do not infer completion from the partial campaign record.
- It may own up to 20 workers. Do not kill workers individually, edit campaign
  sources, or start a second campaign. Retained numerical failures are data.
- #8910 repair worktree:
  `UpstreamDrift-worktrees/8910-real-manufactured-solution`; branch:
  `fix/8910-real-manufactured-solution`, based on verified #8960 remote main.
- PR #8961 is open; implementation commit
  `e47e17050c74b2dae1dd0eb3eb3058a173763be7` contains the coherent scientific
  repair. Inspect the live PR head before review because handoff-only commits
  may follow. Human review and protected CI remain required. Do not enable
  auto-merge or bypass a required check.
- The local repair replaces self-defined torque residuals with analytical
  Lagrange--Christoffel, MuJoCo `mj_inverse`, and robotics Pinocchio RNEA;
  replaces hardcoded conservation zeros with an unforced gravity-free rollout;
  uses adjacent three-level Richardson estimates; and manufactures exact
  constrained closure by coordinating pelvis yaw with club translation.
- Five focused tests pass, including a corruption killswitch that adds 10 Nm
  to the MuJoCo result and requires the release gate to fail. Twenty-nine
  claim/release-governance tests also pass. A prior 24-test neighboring
  articulated run passed; the current combined rerun completed nine tests and
  then reached the inherited 180-second finite-difference timeout while the
  separate 20-worker uncertainty campaign was active, with no assertion
  failure. Ruff format and lint, architecture and document-size budgets,
  explicit title-case checks, and scoped mypy pass. Genuine relative residuals are
  3.09e-11 Lagrange/MuJoCo,
  3.07e-11 Lagrange/Pinocchio, and 3.54e-13 MuJoCo/Pinocchio. Richardson orders
  are 1.0011 and 1.0007. The conservation control excites only the genuinely
  free-floating club subtree; measured gravity-free drift is 1.01e-6 linear
  momentum, 1.51e-6 angular momentum, and 8.67e-7 kinetic energy. Recovered
  constrained-load residuals are 3.39e-13 multiplier, 3.25e-13 cross-engine,
  and 5.51e-13 equilibrium relative. The governed record, five new atomic
  claims, release review, evidence manifest, article source, and 233-page PDF
  are regenerated. Claim governance reports 1,073/1,073 candidates
  adjudicated, 300 claims, 41 reviewed release claims, zero open release
  claims, and a valid 2,128-reference evidence manifest. The computational
  release validator passes for 574 artifacts, 233 rendered pages, 192 valid URI
  links, and 246 outline entries. Archival tagging and font findings remain
  disclosed and fail closed. Next: shepherd PR #8961 through human review and
  protected CI, then verify its merge commit as an ancestor of remote `main`.

## Scientific Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- #8556 remains externally data-gated: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Never substitute
  synthetic traces for human validation.
- #8910 continues to invalidate release-level manufactured-solution closure
  language until the independent controls and regenerated evidence merge to
  remote `main` through protection.
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

## Focused #8910 Validation

```bash
python3 -m pytest -q -n 0 \
  tests/research/test_articulated_manufactured_solution.py
python3 -m ruff check \
  scripts/research/proximal_distal_energy/articulated_manufactured_solution.py \
  tests/research/test_articulated_manufactured_solution.py
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
