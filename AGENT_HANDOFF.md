# Agent Handoff — UpstreamDrift

Last updated: 2026-08-11

Keep this current-state file under 150 lines. Update it with every PR and push
to `main`; use Git history and linked issues for completed chronology.

## Primary Active Program

- **Epic #8426:** all phase PRs target
  `integration/proximal-distal-completion`; only the consolidated branch later
  targets protected `main`. Use full PRs, serialize hosted CI, and preserve
  protected review/check requirements.
- **Phase 0:** PR #8488 merged into integration at `7fdb0fe3d`; it freezes the
  prediction, evidence-v2, wrench/twist, tolerance, migration, and falsifier
  contracts.
- **Phase 1:** PR #8491 merged into integration at `8a6d9c20d`; it executes the
  forward planar two-hand model. Its unchanged post-merge optional/capacity
  jobs remain queued; do not rerun them or publish another hosted wave until
  they settle.
- **Phase 1 result:** a same-state zero-command branch remains negative for
  50 ms and reaches -7.43 N m; coincident grips remove the force couple exactly.
  This is fixed-shoulder, rigid-club, planar model evidence, not muscle or human
  evidence.
- **Phase 2:** commit `ea335ee59` on
  `feat/8426-phase-2-moving-base-flex` couples a translating finite-mass base,
  both arms, solved grip reactions, and one lumped club-flex coordinate. Its PR
  waits on the Phase 1 capacity wave.
- **Phase 2 result:** base motion reaches 26.9 mm, flex reaches 7.08 degrees,
  and the force-couple minimum is -4.15 N m. The 0.200 s zero-command branch
  stays negative for 50 ms; work--energy residual decreases monotonically with
  timestep. This remains a planar mechanism case.
- **Phase 3:** commit `cc77898b6` adds a 20-coordinate reduced full-body
  common-state inverse-dynamics comparison. Native MuJoCo and an independent
  Lagrange--Christoffel formulation agree to `2.14e-11` relative error;
  reversing/coinciding hand moment arms gives the registered sign/zero controls.
  Prescribed loads mean passive forward spatial contact remains untested.
- **Phase 4:** commit `784679ff1` adds a 12-input global screen, delayed
  actuator, rank/null-space identifiability, eight preselected programs, five
  objectives, and separate training/held-out ensembles. The hand-load map has
  nullity one, the parameter map has lower-bound nullity six, and all eight
  controllers are held-out nondominated; no universal strategy is supported.
- **Phase 5:** commit `5cb5b46b7` freezes six synchronized measurement streams,
  four falsifiers, participant-level holdout, identity-safe provenance, and
  fail-closed intake. The committed dry run is synthetic only; EXP-H1--H4 are
  `untested_no_governed_human_data` pending governed measurements and approval.
- **Phase 6:** commit `12c32f489` adds model-tier CLI presets, deterministic
  manifest/checksums, citation/data/license records, read-only validation, and
  a claim-first reviewer workbench. The visually inspected 139-page optimized
  PDF is 1,012,137 bytes with 122 URI links and 171 outlines.
- **Current Phase 7:** `feat/8426-phase-7-beam-reference` reuses the canonical
  Euler--Bernoulli finite-element shaft, adds declared tip inertia, identifies
  a synthetic two-parameter modal case, and compares one-mode and six-mode
  responses. It includes deterministic JSON/NPZ evidence and three paired
  PDF/SVG figures.
- **Phase 7 result:** 24-to-48-element modal change is below 0.090%; the first
  three frequencies are 5.240, 62.931, and 137.909 Hz. Slow-load RMS reduction
  error is 0.000687 mm versus 0.0413 mm under a short force/moment pulse.
  Work--energy residual is at most `2.88e-7` J. Identification recovers only a
  declared synthetic truth; it is not equipment calibration, and the beam is
  not yet coupled into the two-hand KKT solve.

## Remaining Scientific Gates

1. Couple an equipment-calibrated distributed/modal shaft into the forward
   moving-base two-hand solve and repeat contact/energy interventions.
2. Execute independently solved forward spatial two-hand contact in two actual
   engine adapters, including constraints, interface wrenches, energy, events,
   long-axis club rotation, swing-plane evolution, and ground pathways.
3. Evaluate the frozen protocol on governed participant-held-out human data;
   never substitute synthetic readiness for empirical evidence.
4. Deliver synchronized 3-D reviewer views and golden tutorials, synchronize
   the pinned AffineDrift review surface, archive a qualified release with a
   persistent identifier, and verify protected merges on both remote mains.

## Scientific Architecture Pointers

1. `docs/research/proximal_distal_energy_transfer/EVIDENCE_SCHEMA_V2.md`
2. `docs/research/proximal_distal_energy_transfer/MODEL_COMPLETION_FALSIFICATION_MATRIX.md`
3. `scripts/research/proximal_distal_energy/forward_two_arm.py`
4. `scripts/research/proximal_distal_energy/moving_base_flexible_club.py`
5. `scripts/research/proximal_distal_energy/shaft_beam_reference.py`
6. `src/shared/python/physics/flexible_shaft.py`
7. `scripts/research/proximal_distal_energy/spatial_full_body.py`
8. `scripts/research/proximal_distal_energy/uncertainty_control.py`
9. `scripts/research/proximal_distal_energy/experimental_protocol.py`
10. `scripts/research/proximal_distal_energy/release_bundle.py`
11. `CLAUDE.md` and `AGENTS.md`

## Other Current Context

- #8458 and #8443 are merged foundations. Unrelated launch-data, putting,
  impact, and flexible-analysis work must stay out of #8426 phase branches.
- One unrelated Wave 6 test retains its baseline smoothing-factor failure; do
  not mix that engine repair into the scientific epic.

## Gate Commands

```bash
python3 -m ruff format --check .
python3 -m ruff check .
python3 -m pytest -m "not slow and not live_simulation" -n auto --timeout=60
python3 scripts/ci/check_file_size_budget.py
python3 scripts/ci/check_error_handling_ratchet.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/check_docs_governance.py
```

Regenerate evidence and figures, render the Quarto PDF, run
`python3 -m scripts.research.proximal_distal_energy.optimize_article_pdf`, and
visually inspect changed and representative full-document pages.

## Do Not

- Do not edit `vendor/ud-tools/`, generated `.tex`, or generated evidence by
  hand; edit canonical sources and regenerate.
- Do not bypass hooks, reviews, checks, leases, or branch protection; do not
  open drafts, force-push, or create redundant CI reruns.
- Do not infer biological effort from contact force, call pointwise ZTCF a
  forward trajectory, count projection energy as physical work, or promote a
  planar/synthetic result into spatial, equipment, physiological, or human
  evidence.
