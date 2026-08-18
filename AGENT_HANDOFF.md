# Agent Handoff — UpstreamDrift

Last updated: 2026-08-14

Update this current-state file with every PR and push to `main`; history belongs
in git and GitHub.

## Program Authority

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal research program. Issue
  [#8595](https://github.com/D-sorganization/UpstreamDrift/issues/8595) retains
  the photographed nine-point momentum-transfer agenda.
- Canonical question, experiment, and readiness registries live under
  `docs/research/proximal_distal_energy_transfer/data/`. MT-E01 through MT-E07
  are model studies; MT-H01 is the governed participant-held-out stage.
- [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556) remains
  open. No qualifying participant dataset with synchronized bilateral six-axis
  grip wrenches is available; synthetic traces cannot substitute.
- NotebookLM Biomechanics and Nonlinear Control review remains blocked on
  manual Google reauthentication. The 2026-08-14 token check failed closed;
  credentials and authentication dialogs were not automated.

## Active Bilateral-Wrench Sensor-Qualification Slice

- Branch `research/8557-nonlinear-biomechanics-gap` adds MT-E07 and advances Q2
  without claiming a human result.
- Two separated three-axis point forces map to one net club wrench with rank 5
  and nullity 1. The exact invisible mode is equal and opposite axial force
  along the hand-separation direction.
- One independently measured internal axial scalar gives point-force rank 6.
  It does not solve the full problem: two six-axis hand wrenches map to net
  wrench with rank 6 and nullity 6.
- A 49-case 0.06–0.30 m grip-span sweep preserves rank while changing
  normalized nonzero conditioning. Three proper rotations preserve singular
  values to `2.22e-16` under declared 1 N / 1 N m scaling.
- The trajectory-level synthetic qualification exercises 301 samples across 32
  deterministic trials. Net-wrench-only inversion retains 11.86 N allocation
  RMSE and 29.05 N axial-mode RMSE despite numerical resultant-wrench closure.
- In the registered combined synthetic case, the augmented point-force
  estimator has 1.02 N allocation RMSE, 3.87 N 95th-percentile error, 0.0142
  normalized net-wrench RMSE, and 0.351 N axial-mode RMSE.
- Exact cross-talk calibration reduces the declared 1% normalized cross-talk
  case from 0.942 N to 0.153 N allocation RMSE. Exact contact tracking reduces
  the declared 8 mm migration case from 2.025 N to numerical closure.
- These are synthetic point-force estimator results, not device calibration or
  human evidence. Full bilateral moments, distributed contact, compliance,
  muscle/scapular action, intentionality, and strategy remain unestablished.
- Evidence is in `data/bilateral_wrench_sensor_qualification.json`; both
  identifiability and sensor-qualification backends, runners, tests, and figures
  are release artifacts.

## Current Program State

- The claim inventory is adjudicated at 975/975 candidates and 258 atomic
  claims, including structural and synthetic bilateral-wrench qualification.
- The handwritten agenda retains bounded model answers for eight points and a
  construct-level unresolved boundary for human intentional slack.
- Typed slack remains separated into five classes; no global benefit,
  necessity, intentionality, or delivery advantage is established.
- The common-phase timing screen has no sustained half-error recovery in 60
  cases and does not support a state-trigger timing-volume advantage.
- Higher proximal rate is not a universal release rule. Matching choice and
  exact torso/arm/wrist killswitches retain favorable and adverse outcomes.

## Remaining Scientific Work

1. Qualify MT-E07 with a traceably calibrated bilateral six-axis device,
   distributed time-varying contact, compliance, drift, and synchronization.
2. Add full-delivery-state-matched forward rate/acceleration and timing controls
   across spatial and subject-scaled tiers.
3. Estimate continuous attraction regions with identified observers, external
   contact loads, spatial impact, saturation, and subject scaling.
4. Embed each typed-slack class separately in higher-order delivery models.
5. Couple calibrated grip, distributed shaft, articulated arms/scapula, ground,
   impact, and an independent dynamics engine.
6. Execute MT-H01 only after governed bilateral-wrench acquisition and the
   frozen participant split.

## Required Gates

```powershell
python -m scripts.research.proximal_distal_energy.run_bilateral_wrench_identifiability_study
python -m scripts.research.proximal_distal_energy.run_bilateral_wrench_sensor_qualification
python -m pytest tests/research/test_bilateral_wrench_identifiability.py `
  tests/research/test_bilateral_wrench_identifiability_evidence.py `
  tests/research/test_bilateral_wrench_sensor_qualification.py `
  tests/research/test_bilateral_wrench_sensor_qualification_evidence.py `
  tests/research/test_momentum_transfer_experiment_registry.py `
  tests/unit/research/test_momentum_question_readiness.py `
  tests/research/test_proximal_distal_release_bundle.py -q
python -m ruff check scripts/research/proximal_distal_energy/bilateral_wrench_identifiability.py `
  scripts/research/proximal_distal_energy/run_bilateral_wrench_identifiability_study.py `
  scripts/research/proximal_distal_energy/register_bilateral_wrench_identifiability_claims.py `
  tests/research/test_bilateral_wrench_identifiability.py `
  tests/research/test_bilateral_wrench_identifiability_evidence.py
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/check_doc_size_budget.py
```

Render the Quarto PDF, optimize it, inspect affected and boundary pages, write
and validate the release manifest, then verify the protected squash merge on
remote `main`.

## Do Not

- Do not infer human technique, muscle action, injury benefit, or neural timing
  demand from synthetic or structural-identifiability studies.
- Do not treat one net club wrench as a substitute for bilateral sensing.
- Do not collapse pointwise drift, forward persistence, and statistical
  mediation into one estimand.
- Do not close #8556 or #8557 without their declared evidence.
- Do not bypass reviews/checks, force-push, admin-merge, or rerun unchanged
  runner-capacity failures.
