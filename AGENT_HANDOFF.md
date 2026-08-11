# Agent Handoff — UpstreamDrift

Last updated: 2026-08-11

Update this current-state file with every PR and every push to `main`. Keep it
under 150 lines; use Git history and linked issues for completed chronology.

## Primary Active Program

- **Epic #8426 — Proximal-to-Distal Model Completion and Falsification.** All
  phase PRs target `integration/proximal-distal-completion`; only that
  consolidated branch will later target protected `main`.
- **Phase 0:** branch `feat/8426-phase-0-evidence-contracts`, commit
  `c39a0ab1c`, full PR **#8488**. It defines engine-neutral spatial wrench,
  matched-state attribution, prediction, tolerance, migration, and
  claim/falsifier contracts. Required substantive checks are green; only
  unchanged runner-selection and Rust Quickstart jobs are queued. Do not create
  redundant reruns.
- **Phase 1:** branch `feat/8426-phase-1-forward-two-hand`, stacked on the Phase
  0 commit until #8488 is merged. The seven-coordinate model advances two
  two-link arms and a floating planar club under four independent grip
  constraints. The KKT solve and mass-metric position/velocity projections
  fail closed on singular or out-of-tolerance states.
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
- **Focused gate:** 30 evidence-contract/forward-model/figure tests and 20
  dependent hand-path/mechanism-ladder tests pass. Ruff, Python 3.12 mypy,
  document-title, documentation-governance, and file-size gates pass. The final
  113-page, 855,042-byte PDF preserves 110 URI links and 131 outline entries;
  all changed pages and generated figure PDFs were visually checked. The
  pre-push repository gate remains before opening the Phase 1 PR.
- **Next action:** finish report/source registers and gates; render, compact,
  and inspect the complete article; commit/push Phase 1; merge #8488 when its
  capacity-only jobs complete; rebase Phase 1 onto the consolidated branch;
  then open the full Phase 1 PR against that branch.

## Scientific Architecture Pointers

1. `docs/research/proximal_distal_energy_transfer/EVIDENCE_SCHEMA_V2.md` —
   versioned interfaces, attribution, prediction, and tolerance contract.
2. `docs/research/proximal_distal_energy_transfer/MODEL_COMPLETION_FALSIFICATION_MATRIX.md`
   — claim, alternative, tier, falsifier, and status register.
3. `scripts/research/proximal_distal_energy/forward_two_arm.py` — forward
   constrained integrator and exact same-state branch API.
4. `scripts/research/proximal_distal_energy/run_forward_two_arm_study.py` —
   deterministic evidence and negative-control protocol.
5. `scripts/research/proximal_distal_energy/two_arm_closed_loop.py` — canonical
   planar KKT, contact-force, wrench, and mode primitives; do not duplicate.
6. `docs/research/proximal_distal_energy_transfer/WSCG_2024_LEGACY_EVIDENCE_AUDIT.md`
   — registered source claims and pointwise/forward interpretation boundary.
7. `CLAUDE.md` and `AGENTS.md` — binding gates plus discovery-first shared-
   infrastructure rules.

## Remaining Epic Order

1. Dynamic moving-base and compliant-club energy closure.
2. Spatial full-body common-observable experiments in two independent engines.
3. Identifiability, coupled uncertainty, delayed actuation, and robust control.
4. Preregistered experimental falsification and held-out evaluation protocol.
5. Research workbench, AffineDrift publication integration, archival release,
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
- Do not infer biological effort from contact force or promote planar support
  into moving-base, compliant, spatial, or human tiers.
- Do not call pointwise ZTCF a forward trajectory or count projection energy as
  physical work.
- Do not edit generated `.tex`; edit Quarto source and regenerate it.
