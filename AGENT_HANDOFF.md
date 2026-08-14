# Agent Handoff — UpstreamDrift

Last updated: 2026-08-13

Update this current-state file with every PR and push to `main`; history belongs
in git and GitHub.

## Active Research Program

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal research program. Critical-question issue
  [#8595](https://github.com/D-sorganization/UpstreamDrift/issues/8595) remains
  open until its spatial, subject-scaled, and human falsification stages are
  complete.
- Canonical question sources are
  `docs/research/proximal_distal_energy_transfer/MOMENTUM_TRANSFER_QUESTION_PROGRAM.md`,
  `data/momentum_transfer_question_registry.json`, and
  `data/momentum_transfer_experiment_registry.json`.
- #8556 remains open: no qualifying governed participant dataset with
  synchronized bilateral six-axis grip wrenches has been acquired. Synthetic
  traces must not substitute for the registered human test.

## Current Branch

- `research/8595-observer-recovery` adds a paired 15-case delayed-observer and
  trajectory-recovery screen for Q4/Q5. Clock, perfect-state, delayed/noisy,
  and higher-impedance policies receive common nuisance parameters and sensor
  realizations.
- Recovery requires normalized state error to fall below half its initial
  value and remain there through delivery. Fractions are 0.20 for clock and
  delayed/noisy policies and 0.13 for perfect-state and higher-impedance
  policies; median terminal error remains 0.79–0.84. No policy or human
  self-correction advantage is established.
- The study reports recovery trajectories, times, viable-set return, terminal
  delivery, individual-hand force, and effort. The paper includes the result
  and a visually checked figure. The claim audit is complete at 937/937
  candidates and 245 atomic claims.
- The rebuilt, losslessly optimized paper is 206 pages and 1,423,668 bytes,
  with 186 URI links and 241 outline entries. Pages 185–186 were visually
  checked for clipping, legibility, captioning, and figure layout.

## Recently Merged #8595 Slices

- #8596/#8597: question registry, specification, and initial handoff.
- #8598: prospective experiment registry MT-E01 through MT-E06 and MT-H01.
- #8599: reader-facing critical-question chapter and paper integration.
- #8600: participant-held-out human preregistration with fail-closed data gate.
- #8601/#8602: typed-slack constitutive screen and document-budget repair.
- #8603: 27-case timing/casting factorial; angle and rate casting definitions
  disagreed in all cases, so no universal casting event or optimum is claimed.

## Remaining Scientific Work

1. Complete the geometry atlas with null, orthogonal, coincident-grip, and
   reversed-moment-arm controls across planar and spatial tiers.
2. Add matched-work and matched-delivery controls to the timing factorial.
3. Expand observer recovery into a phase-volume and attraction-region study,
   including adverse external loads and identified observer alternatives.
4. Build matched rate/acceleration dose-response surfaces for proximal motion.
5. Extend each typed-slack law into higher-order delivery models without
   conflating contact, backlash, preload, series compliance, or deadband.
6. Repeat the registry in subject-scaled spatial models and independent engines.
7. Execute the frozen participant-held-out protocol only after qualifying data
   acquisition and governance approval.

## Required Gates

```powershell
python -m pytest tests/research/test_observer_recovery_study.py `
  tests/research/test_momentum_transfer_experiment_registry.py -q
python -m ruff check scripts/research/proximal_distal_energy/run_observer_recovery_study.py `
  tests/research/test_observer_recovery_study.py
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/check_doc_size_budget.py
```

For publication, render the Quarto PDF, run
`optimize_article_pdf`, inspect affected pages as PNGs, write and validate the
open-release manifest, and verify the protected squash merge commit on remote
`main`.

## Do Not

- Do not infer human technique, muscle action, injury benefit, or neural timing
  demand from the synthetic studies.
- Do not call terminal dispersion, a transient threshold crossing, or open-loop
  repeatability “self-correction.”
- Do not collapse pointwise drift, forward killswitch persistence, and
  statistical mediation into one estimand.
- Do not close #8556 or #8595 without their declared evidence boundary.
- Do not bypass reviews/checks, force-push, admin-merge, or rerun unchanged
  runner-capacity failures.
