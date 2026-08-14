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

## Subject-Scaled Spatial-Geometry Slice

- PR #8637 merged MT-E08 as
  `04e03b9248dc483737f7e96b7dc0523e63860284` and advances Q2, Q4, Q5, and Q7
  without claiming an anatomically feasible or human result. Branch
  `docs/8557-spatial-spec` records the corresponding SPEC contract required by
  the separate freshness gate.
- Six deterministic de Leva engineering profiles (female and male at 1.55,
  1.75, and 1.95 m) are evaluated at three grip spans and 61 states per case.
- Anatomical hand points miss the prescribed grip contacts by 0.171--0.616 m
  (median 0.405 m); none meets the preregistered 5 mm closure tolerance.
- Every local 6 x 20 bilateral contact Jacobian nevertheless has rank six and
  condition number 5.35--6.40. Local rank is therefore not proof that an open
  prescribed configuration satisfies its declared contact constraints.
- Two separated three-axis point forces map to one net club wrench with rank 5
  and nullity 1. The exact invisible mode is equal and opposite axial force
  along the hand-separation direction.
- One independently measured internal axial scalar gives point-force rank 6.
  It does not solve the full problem: two six-axis hand wrenches map to net
  wrench with rank 6 and nullity 6.
- The prescribed force-generated couple scales linearly with grip span, but
  this algebraic control does not rescue contact closure or anatomical validity.
- Evidence is in `data/subject_scaled_spatial_geometry.json` and `.npz`; its
  backend, runner, tests, claim registration, and figure are release artifacts.

## Current Program State

- The MT-E08 closed-contact follow-up solves all 234 combinations of six
  synthetic profiles, three grip spans, and 13 phase samples while fixing the
  six club coordinates. Worst bilateral closure error is
  `1.15817e-10 m`; every achieved constraint Jacobian has rank six.
- The minimum broad engineering-limit margin is `0.103452 rad`, the minimum
  coarse nonadjacent-body bounding-sphere clearance is `0.0308571 m`, and the
  maximum adjacent-sample configuration change is `0.0255737 rad`.
- Those screens do not establish clinical range of motion, mesh-level anatomy,
  grip force, passive transfer, timing benefit, slack benefit, or human use.
  The next spatial gate is calibrated compliant forward contact initialized
  from the closed configurations.
- The claim inventory is adjudicated at 987/987 candidates and 263 atomic
  claims, including the adverse subject-scaled contact-closure result.
- The handwritten agenda retains bounded model answers for eight points and a
  construct-level unresolved boundary for human intentional slack.
- Typed slack remains separated into five classes; no global benefit,
  necessity, intentionality, or delivery advantage is established.
- The common-phase timing screen has no sustained half-error recovery in 60
  cases and does not support a state-trigger timing-volume advantage.
- Higher proximal rate is not a universal release rule. Matching choice and
  exact torso/arm/wrist killswitches retain favorable and adverse outcomes.

## Remaining Scientific Work

1. Replace the executed broad engineering bounds and bounding-sphere collision
   screen with subject-specific joint geometry, clinical ranges where governed,
   and mesh-level collision/contact qualification.
2. Integrate calibrated compliant contact from the closed states, then repeat
   coincident, reversal, killswitch, power, and work--energy controls in two
   independent forward engines.
3. Qualify MT-E07 with a traceably calibrated bilateral six-axis device,
   distributed time-varying contact, compliance, drift, and synchronization.
4. Add full-delivery-state-matched forward rate/acceleration and timing controls
   across spatial and subject-scaled tiers.
5. Estimate continuous attraction regions with identified observers, external
   contact loads, spatial impact, saturation, and subject scaling.
6. Embed each typed-slack class separately in higher-order delivery models.
7. Couple calibrated grip, distributed shaft, articulated arms/scapula, ground,
   impact, and an independent dynamics engine.
8. Execute MT-H01 only after governed bilateral-wrench acquisition and the
   frozen participant split.

## Required Gates

```powershell
python -m scripts.research.proximal_distal_energy.run_subject_scaled_spatial_geometry
python -m scripts.research.proximal_distal_energy.make_subject_scaled_spatial_geometry_figures
python -m scripts.research.proximal_distal_energy.run_subject_scaled_closed_contact
python -m scripts.research.proximal_distal_energy.make_subject_scaled_closed_contact_figures
python -m pytest tests/research/test_subject_scaled_spatial_geometry.py `
  tests/research/test_subject_scaled_closed_contact.py `
  tests/research/test_momentum_transfer_experiment_registry.py `
  tests/unit/research/test_momentum_question_readiness.py `
  tests/research/test_proximal_distal_release_bundle.py -q
python -m ruff check scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py `
  scripts/research/proximal_distal_energy/run_subject_scaled_spatial_geometry.py `
  scripts/research/proximal_distal_energy/make_subject_scaled_spatial_geometry_figures.py `
  scripts/research/proximal_distal_energy/register_subject_scaled_spatial_geometry_claims.py `
  tests/research/test_subject_scaled_spatial_geometry.py `
  scripts/research/proximal_distal_energy/subject_scaled_closed_contact.py `
  scripts/research/proximal_distal_energy/run_subject_scaled_closed_contact.py `
  scripts/research/proximal_distal_energy/make_subject_scaled_closed_contact_figures.py `
  tests/research/test_subject_scaled_closed_contact.py
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
