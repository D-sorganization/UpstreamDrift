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

- `research/8595-proximal-dose` executes Q6 rate and acceleration controls.
  The velocity atlas now has 126 pointwise cases: 90 relative/absolute-rate
  cases plus 36 exact total-kinetic-energy matches. The zero-energy transition
  dose is correctly non-identifiable rather than energy injected.
- No energy-matched phase is monotonic. The pre-impact drift-power slope falls
  from 283.74 to 5.70 W/(rad/s) when total kinetic energy is held fixed; maximum
  energy residual is `5.7e-14 J`.
- The 60-program forward ledger now includes proximal/distal, positive/negative,
  and net actuator work. Eight reused pairs match net and positive work within
  5%; every higher-rate member is 4.31–9.28 m/s slower, but no pair also matches
  peak interface force within 10%, so the joint causal comparison remains open.
- A separate 45-case identical-state acceleration intervention holds state,
  energy, and distal torque fixed. Before impact, interface-power response is
  positive while club-angular-acceleration response reverses; required proximal
  torque spans -69 to +189 N m. This is pointwise model evidence, not a human
  acceleration strategy.
- The rebuilt paper is 208 pages and 1,489,234 bytes with 186 URI links and
  243 outline entries. Figures and paper pages 53–57 and 185–186 were visually
  checked. The claim audit is complete at 945/945 candidates and 249 claims;
  the qualified open release contains 412 checksum-verified artifacts.

## Recently Merged #8595 Slices

- #8596/#8597: question registry, specification, and initial handoff.
- #8598: prospective experiment registry MT-E01 through MT-E06 and MT-H01.
- #8599: reader-facing critical-question chapter and paper integration.
- #8600: participant-held-out human preregistration with fail-closed data gate.
- #8601/#8602: typed-slack constitutive screen and document-budget repair.
- #8603: 27-case timing/casting factorial; angle and rate casting definitions
  disagreed in all cases, so no universal casting event or optimum is claimed.
- #8605: 15-case observer/recovery screen; no policy establishes a recovery
  advantage and sustained half-error recovery occurs in only 13–20% of cases.
- #8606: analytical geometry atlas plus planar and independent spatial null,
  reversal, and proper-frame power controls.

## Remaining Scientific Work

1. Add jointly work-, load-, and delivery-state-matched forward rate and
   acceleration controls; the current work screen is deliberately not load matched.
2. Add matched-work and matched-delivery controls to the timing factorial.
3. Expand observer recovery into a phase-volume and attraction-region study,
   including adverse external loads and identified observer alternatives.
4. Extend each typed-slack law into higher-order delivery models without
   conflating contact, backlash, preload, series compliance, or deadband.
5. Repeat the registry in subject-scaled spatial models and independent engines.
6. Execute the frozen participant-held-out protocol only after qualifying data
   acquisition and governance approval.

## Required Gates

```powershell
python -m pytest tests/research/test_shoulder_velocity_transfer.py `
  tests/research/test_shoulder_velocity_transfer_study.py `
  tests/research/test_shoulder_velocity_strategy_search.py `
  tests/research/test_shoulder_velocity_strategy_study.py `
  tests/research/test_proximal_acceleration_transfer.py `
  tests/research/test_proximal_acceleration_transfer_study.py `
  tests/research/test_momentum_transfer_experiment_registry.py -q
python -m ruff check scripts/research/proximal_distal_energy/proximal_acceleration_transfer.py `
  scripts/research/proximal_distal_energy/run_proximal_acceleration_transfer_study.py
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
