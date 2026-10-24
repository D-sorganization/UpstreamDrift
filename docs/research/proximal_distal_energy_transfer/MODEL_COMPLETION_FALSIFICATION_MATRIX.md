# Model Completion Falsification Matrix

## Interpretation

This register connects every principal mechanism claim to the evidence that
currently supports it, the strongest competing explanations, and the next test
that can contradict it. `Supported` is reserved for an executed prediction at
the declared model tier. A finding does not automatically inherit support when
it is transported to a higher tier.

The machine-readable prediction definitions are stored in
`data/model_completion_predictions.json`. Status changes require a reviewed
evidence bundle generated after its tolerance and intervention were registered.

The complementary
[`data/release_claim_review.json`](data/release_claim_review.json) covers all 49
claims exposed by the qualified release. It records the supporting atomic
claims, controls, falsifiers, uncertainty boundaries, and the next scientific
gate for each one. Its completed review state means that no release claim is
awaiting adjudication; it does not mean that model, equipment, anatomical,
archival, or human-validation gates have been completed.

The normalized reviewer projections in
[`data/claim_adjudication_summary.json`](data/claim_adjudication_summary.json)
and [`data/claim_adjudication_summary.csv`](data/claim_adjudication_summary.csv)
cover all 303 atomic claims without inferring an outcome from prose or evidence
count. They expose outcome, evidence tier, independent-work count, model tier,
unresolved replication class, and claim-family source concentration. These
axes describe the support boundary; they do not promote project-only model
evidence into independent or human validation.

The complementary
[`data/external_source_review.json`](data/external_source_review.json)
prevents URL count from being mistaken for independent-source count. It
consolidates mirrors into underlying works, assigns a claim-specific evidence
role and uncertainty boundary, records scholarly-update checks, and rejects
broken or unchecked links. Its passing state qualifies source traceability and
scope only; it does not promote an observational association, review,
simulation, or method paper into causal human validation.

## Claim–Evidence–Falsifier Register

The claim–evidence–falsifier register is maintained in [`MODEL_COMPLETION_CLAIM_REGISTER.md`](MODEL_COMPLETION_CLAIM_REGISTER.md). The model-tier discrepancy register is in [`MODEL_COMPLETION_MODEL_DISCREPANCY.md`](MODEL_COMPLETION_MODEL_DISCREPANCY.md).

## Status Rules

- **Supported:** the registered expected result passes its preoutcome decision
  rule and applicable negative controls.
- **Contradicted:** the registered falsifier occurs outside numerical and
  declared uncertainty bounds.
- **Inconclusive:** the observable is unavailable, non-identifiable, singular,
  or too uncertain to distinguish the alternatives.
- **Untested:** the required experiment or model tier has not been executed.

Publication language must preserve these statuses. A visually similar curve,
successful trajectory fit, or higher force magnitude is not a substitute for
the registered estimand and intervention.

## #8556-Conditional Parameters

Issue [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556) serves as the human-validation gate requiring synchronized bilateral six-axis grip wrenches. In addition to serving as a validation gate, several key _model-tier_ parameters and claims are conditionally anchored to the missing #8556 experimental dataset:

1. **Grip Stiffness and Damping (`k_grip`, `c_grip`):**

   - The top-tier Kelvin-Voigt grip interface currently assumes a nominal total stiffness of \(k*{\text{grip}} = 1800\,\text{N/m}\) and damping of \(c*{\text{grip}} = 18\,\text{N}\cdot\text{s/m}\) (`articulated_distributed_grip.py`), setting the overall force scale of the unactuated articulated response (#8912).
   - _Outcome sensitivity:_ In one-factor stiffness sweeps across the ladder (\(1.8\times 10^3\), \(1.8\times 10^4\), \(1.8\times 10^5\,\text{N/m}\)), higher stiffness dramatically scales peak station forces toward realistic downswing magnitudes (from \(\sim 4\,\text{N}\) up into \(\mathcal{O}(100\text{--}500\,\text{N})\)), altering the relative free-flight response of the club relative to hand carriages.

2. **Coulomb Friction Cone Anchor Normal Force (\(\mu\)):**

   - The friction cone (\(\mu = 0.35\)) is currently anchored to the internal spring tension of the grip fibers rather than measured physiological compressive/squeeze normal force (#8915).
   - _Outcome sensitivity:_ Anchoring friction to fiber tension means that as radial fiber displacement approaches zero, the cone collapses to zero, producing false-positive slip indications even under negligible shear force. True frictional grip stability requires empirical normal pressure distribution from #8556.

3. **Tension-Only Unilaterality Direction:**
   - The distributed contact model (`articulated_distributed_grip.py:186-200`) assumes unilateral tension-only registration where fibers can pull but not push the club, which represents the opposite physical unilaterality from compressive hand-shaft contact.
   - _Outcome sensitivity:_ Club stabilization under compressive deceleration relies directly on the contact unilaterality orientation.

As established by the analytical rank deficiency proof in `bilateral_wrench_identifiability.py` ("net-wrench closure does not establish force allocation", `README.md:200-204`), net resultant external forces and moments measured at the overall club or wrists cannot mathematically substitute for synchronized bilateral six-axis grip wrench measurements. Consequently, parameters (a), (b), and (c) remain uncalibrated model assumptions until #8556 data acquisition is completed.
