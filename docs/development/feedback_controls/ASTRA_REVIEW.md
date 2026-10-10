# Astra Architecture Review

## Astra Architecture Review: Binding Implementation Clarifications

These instructions refine the earlier plan; they preserve the full all-model/six-engine and muscle-driven OpenSim scope. Planning publication does not complete this epic.

1. **Acyclic delivery and early replay.** Tools T03 implementation depends on T01/T02 and F01's frozen resource/preview contract. Its downstream acceptance integrates with F09, which is not an implementation prerequisite. F06 has an early native replay-contract/smoke-test milestone after F01/T01 and final integration after F02/F03. Every F03 candidate must pass the early independent validator. Begin F07 native state/muscle/contact probes early; final qualification retains F03.
2. **Qualification firewall.** Every receipt identifies dynamics, contact, integration, actuator and restart implementations. Shared rigid-body/common-contact engine tests, native model simulation and native muscle/model-contact simulation are distinct evidence classes. Restarted, externally forced, torque-only or shared-contact results cannot satisfy uninterrupted native muscle-driven own-contact gates. Required capabilities are frozen separately from availability/support and remain in the completion denominator. Physical inapplicability needs a scientific rationale, not a missing implementation.
3. **Versioned state and stepping.** Existing same_input v1 and MOSAIC local-policy fixtures assume Euclidean equal configuration/velocity dimensions. Reuse them within that scope; versioned adapters must support configuration retraction/difference, tangent dimensions, complete continuous/discrete/actuator/plugin/filter state and actual native stepping before claiming broader capability. Declare physical versus numerical integration state or cold-start both reference and replay identically. Never flatten quaternion/muscle state into a torque-only contract.
4. **Four separate measurements.** Record applied-input identity, same-policy within-engine reproduction, independently integrated transcription feasibility and observation agreement separately. Collocation nodes are not a native rollout; require uninterrupted forward integration and predeclared mesh/integration refinement. Score frozen nominal feedforward replay separately from recorded total closed-loop input replay. Replaying recorded feedback corrections does not establish nominal open-loop sufficiency or identify the feedback law.
5. **Actual input boundary and integrity.** Declare actuator command, actuator force/torque, generalized effort, excitation or external-load injection. ZOH command need not mean ZOH force when transmission/state dynamics vary. Bind executed policy, time grid, complete initial-state/input payloads and model/provider identities; reject tampering. A native time-only input player (including PrescribedController) is allowed; state-feedback/observation callbacks and measured-state resets are forbidden in independent replay.
6. **Contact and assistance physics.** Rigid contact uses formulation-appropriate unilateral/complementarity checks. Compliant contact permits law-consistent bounded deformation with required nonadhesion, friction/dissipation and numerical convergence. Reserve/root gates include channel-wise peak/RMS force or torque, integrated absolute effort and separate positive/negative work, with declared normalization; zero signed work cannot establish muscle-only actuation. Audit passive/limit/constraint/external contributions without double counting.
7. **Controller/recruitment identifiability.** Tune on frozen nominal trajectories and predeclared held-out perturbations; disclose nonuniqueness/regularization if feedforward and gains are jointly adjusted. Separate frozen-controller sensitivity from compensation after reoptimization. Declare exact simulated versus estimated/noisy/delayed information and validate the claimed information pattern. Recruitment remains objective/prior dependent; physiological interpretation requires parameter/objective sensitivity or ensembles, beyond kinematic agreement.
8. **Real cost and scope.** Time-to-accepted-match includes preparation, derivatives, failed solves/retries, validation and export. Use bounded independent trial/candidate parallelism and complete cache identity. Small native muscle/contact fixtures qualify architecture only and do not reduce final full-model/capture coverage. NMPC/contact-implicit/synergy/learned approaches remain benchmarked options; no unsupported algorithm winner is promised.

Required regression cases: cycle detection; unavailable required engine; common-model/restarted/externally forced receipt falsely promoted; quaternion nq!=nv; actuator internal state; nonzero activation/tendon state; missing native initialization state; altered actuator order; tampered policy/input digest; nonunit/configuration-dependent transmission; bounded-excitation interpolation overshoot; poisoned observation callback; coarse collocation failing native replay; total-feedback replay passing while nominal feedforward fails; hidden static residual and opposing zero-net-work assistance; valid compliant-contact deformation; gain nonuniqueness and held-out perturbation response.

Existing source boundaries motivating the firewall: `motion_matching/same_input/bundle.py` and `integrator.py`, `estimation/mosaic/local_policy.py`, OpenSim and MyoSuite `full_body_parity.py`. Their current valid common-model/Euclidean scope is retained, not silently generalized. MuJoCo state/reproducibility and OpenSim forward-integration references are primary method guidance; native test evidence remains required.

## Milestone Delivery

milestone_dependencies.json records implementation readiness separately from scientific acceptance. D02 consumes T02 schema readiness; D03 consumes runnable F07-F09 providers and supplies their native evidence; D04 feeds F10.

## October 10 Endpoint Review

The renewed review of integrated native platform head `45ad4e372c5f32e12bd15a183ef09468b2024bd1`
puts declared constrained cold-start reconstruction and source-chart-bound
independent replay before additional optimization. Child #12136 must distinguish
model-owned lock targets from named State values, reject out-of-domain default
humerus initialization, and test the actual retained mechanical candidate.
Source-coordinate and branch-aware muscle path/work admission follows, retaining
`r = a + b` and the corresponding force covectors. Whole-body bilateral/contact
assembly and frozen private reference geometry precede full-horizon Moco inference
and fresh saved-excitation replay. Numerical seed availability does not remove
the six original-source scientific/preparation blockers. The original model's
large passive loads cannot be corrected by an unsupported donor-pose choice.
All seventeen variants and six ecosystems remain required; no new optimizer or
schema is justified by this review. The detailed read-only reasoning is retained
as `ASTRA_RESUMED_ENDPOINT_REVIEW_20261010.md` in the fleet planning directory.

## Primary Method References

- [MuJoCo Simulation State](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)
- [MuJoCo Numerical Reproducibility](https://mujoco.readthedocs.io/en/3.3.5/computation/)
- [OpenSim Forward Integration Utilities](https://opensim-org.github.io/opensim-moco-site/docs/0.2.0/group__mocomodelutil.html)
