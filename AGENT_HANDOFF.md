# Agent Handoff — UpstreamDrift

Last updated: 2026-08-15

This is a current-state handoff. Historical detail belongs in git and GitHub.

## Program Authority

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal research program. Issue
  [#8595](https://github.com/D-sorganization/UpstreamDrift/issues/8595) retains
  the photographed nine-point momentum-transfer agenda.
- PR [#8664](https://github.com/D-sorganization/UpstreamDrift/pull/8664)
  merged the closed-state forward bridge as
  `eff94297743d0612075f11648f60419044e2067c`. PR
  [#8665](https://github.com/D-sorganization/UpstreamDrift/pull/8665) merged its
  architecture isolation as `9f0c18a4d50b8c30ce3bafbb36a2e45197ffe1d3`,
  verified as exact remote `main` before the present slice.
- Issue [#8666](https://github.com/D-sorganization/UpstreamDrift/issues/8666)
  is the current protected slice: the cross-engine forward-contact validity
  horizon and adverse-load map.
- Issue [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556)
  remains open. No qualifying participant dataset with synchronized bilateral
  six-axis grip wrenches is available; synthetic traces cannot replace it.
- NotebookLM review remains blocked on manual Google reauthentication. Do not
  automate credentials, authentication dialogs, CAPTCHA, or two-factor steps.

## Current Scientific State

- The question, experiment, claim, evidence, and readiness registries live in
  `docs/research/proximal_distal_energy_transfer/data/`. The claim inventory is
  fully adjudicated at 1,004 candidates and 272 atomic claims; all 33 release
  claims have deterministic dispositions.
- Evidence integrity covers 1,743 support references, 216 hash-pinned local
  artifacts, and 78 external URLs representing 56 works. Link availability is
  not treated as independent validation.
- The photographed agenda has bounded model answers or partial answers for
  eight of nine points. MTQ-06—whether passive or drift-mediated transfer
  reduces timing precision beyond the adverse planar comparison and in
  people—remains unresolved.
- Typed slack remains separated into five classes. No global slack benefit,
  necessity, intentionality, delivery advantage, or coaching rule is claimed.
- Higher proximal rate is not a universal release rule. Matching choice and
  torso, arm, and wrist killswitches retain favorable and adverse outcomes.

## Subject-Scaled Geometry and Closed-State Bridge

- Six deterministic engineering profiles, three grip spans, and 13 phase
  samples form 234 prescribed states. Anatomical hand points miss prescribed
  contacts by 0.171–0.616 m (median 0.405 m); none meets 5 mm closure.
- Both contact Jacobians can still be rank six. Local rank therefore does not
  prove that an open prescribed configuration satisfies its contacts.
- The closed-contact solve reaches `1.15817e-10 m` worst closure with rank-six
  achieved constraints, but broad joint bounds and bounding spheres are not
  clinical range-of-motion or mesh-level anatomy evidence.
- A scapula-on-ellipsoid surrogate closes 31 of 54 paired states to 0.5 mm and
  passes optimizer termination in 16. This exposes geometry sensitivity and
  allocation non-identifiability; it does not establish anatomy or muscle use.
- The engine-neutral closed-state bridge preserves inertial velocities, creates
  zero Kelvin–Voigt preload at exact closure, and gives every state a unique
  digest. Native MuJoCo and robotics Pinocchio receive identical digests.

## Validity-Horizon Result

- The #8666 study evaluates 4, 10, 25, and 50 ms for all 54 selected
  profile–span–phase states under nominal conditions and nine one-factor/null
  variants. Ten variants, two engines, and four horizons produce 1,080 engine
  traces and 2,160 paired horizon cases.
- All cases pass the registered trajectory, wrench, normalized-energy, and
  work–energy gates through 50 ms. No first failure is observed, so the result
  is right-censored at 50 ms rather than evidence for a full downswing.
- At 50 ms, worst nominal club-position difference is 0.503 micrometres,
  relative wrench RMS is 0.0252%, and normalized-energy difference is
  `3.20e-06`.
- Across all variants, worst position difference is 1.02 micrometres, wrench
  RMS is 0.0506%, and energy-closure residual is 0.327%. Timestep halving
  improves worst closure to 0.0818%, the expected refinement direction.
- The reduced model uses finite-mass hand carriages rather than articulated
  arms. It does not establish anatomy, equipment behavior, full delivery,
  passive human transfer, timing benefit, or coaching strategy.

## Next Scientific Gates

1. Add subject-scaled articulated arms and scapulae with independently
   qualified joint geometry, clinical limits where governed, and mesh contact.
2. Calibrate distributed grip and shaft compliance, damping, friction, contact
   loss, ground coupling, and equipment uncertainty.
3. Repeat the registered horizon, reversal, killswitch, adverse-load,
   refinement, power, and energy-ledger controls in two independent engines.
4. Add full-delivery state-matched rate, acceleration, and timing controls;
   estimate continuous attraction regions with identified observers.
5. Embed each typed-slack class separately rather than treating slack as one
   binary state.
6. Qualify MT-E07 using a traceable bilateral six-axis measurement device with
   synchronization, drift, distributed contact, and uncertainty controls.
7. Execute MT-H01 only after governed participant acquisition and the frozen
   participant-held-out split.

## Reproduction and Release Gates

```powershell
python -m scripts.research.proximal_distal_energy.run_forward_contact_validity_horizon
python -m scripts.research.proximal_distal_energy.make_forward_contact_validity_horizon_figure
python -m pytest tests/research/test_forward_contact_validity_horizon.py `
  tests/research/test_closed_state_forward_bridge.py `
  tests/research/test_proximal_distal_release_bundle.py -q
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/check_doc_size_budget.py
pre-commit run --hook-stage pre-push --all-files
```

Render the Quarto PDF, optimize it, inspect affected and boundary pages, and
validate the release manifest before the protected squash merge.

## Do Not

- Do not infer human technique, muscle action, injury benefit, neural timing
  demand, or coaching advice from synthetic or identifiability studies.
- Do not treat one net club wrench as a substitute for bilateral sensing.
- Do not collapse pointwise drift, forward persistence, and statistical
  mediation into one estimand.
- Do not close #8556 or #8557 without their declared evidence.
- Do not bypass reviews or checks, force-push, admin-merge, or rerun unchanged
  runner-capacity failures.
