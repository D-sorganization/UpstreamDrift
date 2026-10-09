# Implementation Plan

Governing epic: https://github.com/D-sorganization/UpstreamDrift/issues/11784

All items are planned and unqualified. IDs and dependencies resolve to real issue URLs; existing MOSAIC and parity work remain authoritative.

F09 implementation slices include [F09a: Output Sampling](https://github.com/D-sorganization/UpstreamDrift/issues/11831), [F09b: Native Observation Scoring](https://github.com/D-sorganization/UpstreamDrift/issues/11837), [F09c: Strict Native Replay Execution](https://github.com/D-sorganization/UpstreamDrift/issues/11898), [F09d: Native Marker Forward Kinematics](https://github.com/D-sorganization/UpstreamDrift/issues/11907), and [F09e: Pinocchio Native Marker Forward Kinematics](https://github.com/D-sorganization/UpstreamDrift/issues/11914). F09e depends on F09d's marker contract and the published Pinocchio frozen-replay adapter in #11906; it reuses both authorities without copying provider source. These bounded children preserve F09's required six-engine denominator and leave the parent epic open for remaining native consumers and qualification evidence.

| Task                                                                 | Deliverable                                                                 | Dependencies                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785) | Freeze Six-Engine Model/Actuation Parity and Benchmark Contracts            | Ready; coordinate existing #11533 and #11605 owners                                                                                                                                                                                                            |
| [F02](https://github.com/D-sorganization/UpstreamDrift/issues/11786) | Implement Distributed Task/Phase Feedback With Constrained Allocation       | [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785); MOSAIC reference/gain provider; Tools [T01](https://github.com/D-sorganization/Tools/issues/5461)                                                                                        |
| [F03](https://github.com/D-sorganization/UpstreamDrift/issues/11787) | Benchmark Constrained Optimal Feedforward and Tracking Backends             | F01; early F06 replay validator; MOSAIC/contact providers; D02 frozen protocol before real-data fitting. Native candidate evaluation precedes final integration acceptance.                                                                                    |
| [F04](https://github.com/D-sorganization/UpstreamDrift/issues/11788) | Tune Interconnected Control Loops and Quantify Coupling                     | [F02](https://github.com/D-sorganization/UpstreamDrift/issues/11786), [F03](https://github.com/D-sorganization/UpstreamDrift/issues/11787); private split/uncertainty contract                                                                                 |
| [F05](https://github.com/D-sorganization/UpstreamDrift/issues/11789) | Evaluate Robust NMPC With Bounded Runtime and Verified Fallback             | [F02](https://github.com/D-sorganization/UpstreamDrift/issues/11786)-[F04](https://github.com/D-sorganization/UpstreamDrift/issues/11788); [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785) benchmark protocol                             |
| [F06](https://github.com/D-sorganization/UpstreamDrift/issues/11790) | Integrate Exact Applied-Torque Export and Independent Controller-Off Replay | Early replay contract/native smoke tests: F01 and T01. Final integrated acceptance: F02/F03 runnable providers and D03 evidence. Reuse #11605/#11607.                                                                                                          |
| [F07](https://github.com/D-sorganization/UpstreamDrift/issues/11791) | Qualify OpenSim Muscle/Contact Model and Dynamic Tracking Baseline          | Early native state/muscle/contact probe: F01/T01 contracts. Runnable qualified-model candidate: F03 provider, D02 protocol and private inventory. Scientific acceptance consumes D03 evidence; F08 provider development does not wait for campaign completion. |
| [F08](https://github.com/D-sorganization/UpstreamDrift/issues/11792) | Match Mocap With Muscle Excitations and Verify Pure OpenSim Replay          | Runnable excitation matching/replay: F04 and F07 model/provider readiness, T01/T02 schema readiness, D02 protocol. Scientific acceptance consumes D03 evidence, not a prerequisite for D03 execution.                                                          |
| [F09](https://github.com/D-sorganization/UpstreamDrift/issues/11793) | Maintain All-Model Six-Engine Parity and Matching Workflow                  | Runnable parity/workflow: F06-F08 runnable providers and T02/T03 implemented contracts. Final integration/scientific acceptance consumes D03 evidence. T03 downstream consumer acceptance is not a workflow prerequisite.                                      |
| [F10](https://github.com/D-sorganization/UpstreamDrift/issues/11794) | Run Qualification Campaign and Deliver Reproduction/Turnover Packet         | F01-F04 and F06-F09 final acceptance; explicit optional F05 disposition; T01-T03 final acceptance; D01-D04 evidence. F10 consumes D04 and is never its prerequisite.                                                                                           |

## Engineering and Evidence Contract

- TDD: record a meaningful failing behavioral test, minimal green implementation and refactor; use independent analytic/synthetic truth and native integration where physics is claimed.
- DbC: validate finite states, units/frames/clocks, manifold and actuator dimensions, bounds, capability and provenance. Use existing public facades (LoD) and authoritative shared kernels/contracts (DRY).
- Update design calculations/manual QMD and registries when applicable, SPEC/change fragment and canonical turnover in the implementation PR. Include exact commands, provider versions, failures and receipt evidence.
- Preserve existing gates; new tolerances must be ratified from noise/baseline/convergence evidence before fitting. No weakened acceptance to rescue a result.
- Respect private-data boundaries. All models/engines remain represented in the capability/parity matrix, with unavailable or unqualified states explicit. No simulation result is implied by this issue.

## Review Refinements

**F01:** Adopt the existing authoritative model/variant/capability registry after source discovery, with stable engine, model-variant and drive-mode keys; do not create a parallel registry. This task inventories/freezes policy; #11605/#11607 retain replay implementation authority and F09 owns ongoing conformance execution. Make each comparison level executable with common states, horizons and channel masks.

**F02:** Independent analytic pendulum/manipulator tests must show a deliberately reversed feedback sign diverges or fails a restoring/stability bound, and the correct controller satisfies that bound within its stated local domain. Include saturation and actuator-permutation negative cases; marker objective alone is not a stability test.

**F06:** Expose a replay boundary that accepts only a frozen bundle and a native plant; it cannot receive a controller or observation provider. Tests prove no measurement callback, controller evaluation or measured-state-reset path is available.

**F07:** Publish an explicit F08 handoff: pinned muscle/contact model and provider, parameter/uncertainty set, complete initial activation/fiber/tendon states, contact/external-load policy, version/capability probe, reserve bounds and tested native forward API. Missing fields block F08.

**F08:** Use the same isolated replay boundary principle as F06: saved excitation bundle plus native plant only, without controller/observation providers. Reject hidden measurement callbacks and state resets. Excitation equality across engines is claimed only when muscle definitions, actuation dynamics and mappings are identical.

**F09:** Consume the registry and frozen comparison policy adopted by F01; do not define a second matrix. Keep per-cell applicability masks and unsupported/unavailable/unqualified states in receipts; no aggregate all-models-passed claim unless every applicable gate is satisfied. Equal-state/torque parity applies only to the same mapped dynamics/actuation model.

**F10:** Optional NMPC F05 is not a requirement to ship an accepted simpler controller: retain a benchmark/report or explicit not-selected disposition. Preserve every unqualified/unsupported cell and applicable mask in final reporting; do not collapse missing evidence to pass.

## Planning Delivery

[F00: Publish This Planning Packet](https://github.com/D-sorganization/UpstreamDrift/issues/11797) is the documentation-only delivery child. Its PR may close that child; all implementation/qualification children and the parent epic remain open.

## Milestone Dependency Semantics

Dependencies identify named deliverables, not automatic closure of entire issues.
Early schema/replay/probe readiness enables private protocol freezing and native
provider development. Runnable F07-F09 providers enable D03, which supplies their
scientific acceptance evidence. D04 consumes D03 artifacts and T03 resources;
F10 consumes D04. No private task waits for F10. This section and the corrected
table supersede earlier whole-issue dependency descriptions.
