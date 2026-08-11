# Agent Handoff — UpstreamDrift

Last updated: 2026-08-11

Update this current-state file with every PR and every push to `main`. Keep it
under 150 lines; use Git history and linked issues for completed chronology.

## Primary Active Program

- **Epic #8426 — Proximal-to-Distal Model Completion and Falsification.** All
  phase PRs target `integration/proximal-distal-completion`; only that
  consolidated branch will later target protected `main`.
- **Phase 0:** PR **#8488** merged into integration at `7fdb0fe3d`. It defines
  engine-neutral spatial-wrench, matched-state attribution, prediction,
  tolerance, migration, and claim/falsifier contracts.
- **Phase 1:** PR **#8491** merged into integration at `8a6d9c20d`. Its
  seven-coordinate model advances two two-link arms and a floating planar club
  under four independent grip constraints. All substantive checks are green;
  unchanged post-merge runner-selection/optional jobs remain queued. Do not
  rerun them.
- **Phase 1 evidence:** deterministic baseline plus exact same-state
  zero-command branches; separate force-generated couple and direct wrist
  torque; common/differential force modes; contact-power identity; zero-grip-
  moment-arm negative control; timestep and projection sensitivity; JSON/NPZ
  outputs; three paired PDF/SVG figures; and a report chapter.
- **Bounded result:** baseline negative-couple onset is 0.19825 s. The exact
  0.200 s zero-command branch starts at -2.57 N m, remains negative for 50 ms,
  and reaches -7.43 N m. Removing both grip moment arms makes the force couple
  exactly zero. This is fixed-shoulder, rigid-club, planar model evidence, not
  muscle, human, or coaching evidence.
- **Phase 1 gate:** 30 evidence/forward/figure tests plus 20 dependent tests,
  Ruff, mypy, title, governance, and size gates passed. The 113-page PDF
  preserved 110 URI links and 131 outlines and passed visual review.
- **Phase 2:** branch `feat/8426-phase-2-moving-base-flex`, commit `ea335ee59`,
  is locally complete and based on merged Phase 1. Its PR waits for the
  remaining #8491 post-merge capacity jobs to settle so hosted waves remain
  serialized.
  The ten-coordinate forward model couples a finite-mass translating base, two
  closed-loop arms, two solved grip reactions, and a two-segment compliant club.
  It records base and shaft energy, force-generated grip couple, direct wrist
  torque, shaft moment, common-wrench samples, contact-power identity, and all
  constraint/projection residuals.
- **Phase 2 bounded result:** the declared baseline moves the base by 26.9 mm,
  flexes 7.08 degrees, and crosses to a -4.15 N m force-generated minimum. The
  exact 0.200 s zero-command branch remains negative for 50 ms and reaches
  -2.17 N m. Coincident grips remove the force couple exactly. Work--energy
  residual decreases 0.0483 -> 0.0254 -> 0.0130 J under 2/1/0.5 ms refinement.
  This is planar mechanism evidence, not anatomical, equipment, or human
  validation.
- **Phase 2 gate:** the 121-page PDF, focused and dependent tests, Ruff, mypy,
  title, governance, size, and visual checks passed before `ea335ee59`.
- **Phase 3:** branch `feat/8426-phase-3-spatial-cross-engine`, commit
  `cc77898b6`, adds one canonical
  reduced full-body 3D model evaluated by native MuJoCo inverse dynamics and an
  independently assembled Lagrange--Christoffel formulation. It is a
  cross-formulation common-state experiment, not yet a two-engine forward
  contact simulation.
- **Phase 3 bounded result:** the 20-coordinate, 32-inertia model has 34.96 mm
  out-of-plane club motion. Prescribed action--reaction hand loads create a
  -4.32 N m couple; reversing the moment arm reverses its sign to numerical
  precision and coincident hands remove it exactly. Across 61 states, the two
  formulations differ by at most `1.26e-9` generalized-force units and
  `2.14e-11` relative. This supports common-state implementation transport,
  not passive load origin, forward spatial contact, or human inference.
- **Phase 3 gate:** 67 adjacent scientific tests, Ruff, mypy, title,
  governance, size, error-ratchet, source-hash, and visual gates passed. The
  optimized 129-page PDF preserves 115 URI links and 150 outlines.
- **Phase 4:** branch `feat/8426-phase-4-uncertainty-control` adds a deterministic
  12-input global screen and separate six-case training and held-out ensembles
  to the coupled moving-base/flexible-club model. It includes pure delay,
  first-order activation, torque-rate and torque--velocity limits, an impedance
  proxy, eight preselected command programs, five objectives, PRCC screening,
  and rank/null-space identifiability audits.
- **Phase 4 bounded result:** global delivery speed spans 4.005/5.188/5.840 m/s
  at the 5th/median/95th percentiles. Net planar wrench has rank 3 for four hand-
  force components (nullity 1), while six summary observables leave a lower-
  bound nullity of six for 12 parameters. Early restraint improves held-out
  lower-tail speed over late drive (4.706 vs 4.417 m/s) but worsens the planar
  face/path proxy (5.44 vs 1.58 degrees). All eight programs are held-out
  nondominated: no universal optimum, physiological, human, or coaching claim.
- **Phase 4 gate:** 19 focused tests and 308 other research tests, Ruff, mypy,
  title, governance, size, error-ratchet, source-hash, and visual gates pass.
  The optimized 135-page PDF preserves 120 URI links and 158 outlines. One
  unrelated legacy Wave 6 test remains at its baseline smoothing-factor
  assertion; do not mix that engine repair into this scientific phase.
- **Next action:** finish and commit Phase 4 locally. When #8491 capacity-only
  jobs settle, publish and merge Phase 2 against integration; then rebase and
  publish the unpushed Phase 3 and Phase 4 branches sequentially.

## Scientific Architecture Pointers

1. `docs/research/proximal_distal_energy_transfer/EVIDENCE_SCHEMA_V2.md` —
   versioned interfaces, attribution, prediction, and tolerance contract.
2. `docs/research/proximal_distal_energy_transfer/MODEL_COMPLETION_FALSIFICATION_MATRIX.md`
   — claim, alternative, tier, falsifier, and status register.
3. `scripts/research/proximal_distal_energy/forward_two_arm.py` — forward
   constrained integrator and exact same-state branch API.
4. `scripts/research/proximal_distal_energy/run_forward_two_arm_study.py` —
   deterministic evidence and negative-control protocol.
5. `scripts/research/proximal_distal_energy/moving_base_flexible_club.py` —
   coupled finite-mass base, closed arms, flexible club, and energy contracts.
6. `scripts/research/proximal_distal_energy/run_moving_base_flexible_study.py` —
   deterministic branch, sensitivity, negative-control, and convergence study.
7. `scripts/research/proximal_distal_energy/spatial_full_body.py` — common
   reduced spatial model, wrench intervention, and two-formulation inverse
   dynamics comparison.
8. `scripts/research/proximal_distal_energy/uncertainty_control.py` — coupled
   design, delayed actuator, PRCC, identifiability, and control comparison.
9. `scripts/research/proximal_distal_energy/two_arm_closed_loop.py` — canonical
   planar KKT, contact-force, wrench, and mode primitives; do not duplicate.
10. `docs/research/proximal_distal_energy_transfer/WSCG_2024_LEGACY_EVIDENCE_AUDIT.md`
    — registered source claims and pointwise/forward interpretation boundary.
11. `CLAUDE.md` and `AGENTS.md` — binding gates plus discovery-first shared-
    infrastructure rules.

## Remaining Epic Order

1. Full-body forward cross-engine contact remains an explicit open tier.
2. Preregistered experimental falsification and human held-out evaluation
   protocol.
3. Research workbench, AffineDrift publication integration, archival release,
   complete visual QA, and protected merges.

## Other Current Repository Context

- **#8458 hand-path attribution** is merged into current `main` through #8473;
  it provides the pointwise ZTCF/control/ZVCF terminology reused here.
- **#8443 mechanism work** is merged through #8452–#8457: interaction forces,
  killswitch ensemble, WSCG two-hand audit, shaft surrogate, model ladder, and
  compact article foundation.
- **#8430 private launch data** keeps real shot data behind the authenticated
  `LAUNCH_MONITOR_DATA_ROOT` authority; public tests use synthetic fixtures.
- **#8345 putting dynamics/UI**, **#8344 impact physics**, and launch-monitor
  flexible-analysis work are independent; do not mix them into #8426 branches.

## Gate Commands

```bash
python3 -m ruff format --check .
python3 -m ruff check .
python3 -m pytest -n auto --timeout=60
python3 -m pytest -m "not slow and not live_simulation" -n auto --timeout=60
python3 scripts/ci/check_file_size_budget.py
python3 scripts/ci/check_error_handling_ratchet.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/check_docs_governance.py
```

For the article, regenerate source data and figures, render Quarto PDF, run
`python3 -m scripts.research.proximal_distal_energy.optimize_article_pdf`, then
inspect every changed/full-document page rather than relying on compilation.

## Do Not

- Do not edit `vendor/ud-tools/`; fix the canonical Tools repository.
- Do not import the deprecated `upstream_drift_tools`; use `sidekick`.
- Do not bypass hooks, reviews, required checks, leases, or branch protection.
- Do not open drafts, force-push, or create redundant CI reruns.
- Do not infer biological effort from contact force or promote coupled planar
  support into spatial or human tiers.
- Do not call pointwise ZTCF a forward trajectory or count projection energy as
  physical work.
- Do not edit generated `.tex`; edit Quarto source and regenerate it.
