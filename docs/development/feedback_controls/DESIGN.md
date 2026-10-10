# Feedback-Controlled Motion Matching and Muscle-Driven OpenSim

## Status and Expanded Goal

Planning specification, 2026-10-08. Governing epic: [UpstreamDrift #11784](https://github.com/D-sorganization/UpstreamDrift/issues/11784); shared provider epic: [Tools #5460](https://github.com/D-sorganization/Tools/issues/5460). No capture, control algorithm or physiological model is qualified by this document.

The program covers **all six current engines and their applicable model variants**: MuJoCo, Drake, Pinocchio/Crocoddyl, OpenSim, MyoSuite and Simscape. Track torque-driven, muscle-driven, rigid/flexible-club and reduced/full-body variants explicitly; a reduced-model success does not qualify a full-body model. New registered models inherit capability and parity conformance. The final product target is **muscle-driven OpenSim matching the available mocap**, with full-state, controller-off excitation replay and best defensible cross-engine biomechanical parity. Saved torque replay remains an essential intermediate gate.

## Success Criteria and Existing Authorities

Extend, rather than replace, [MOSAIC #11532](https://github.com/D-sorganization/UpstreamDrift/issues/11532), [Same-Input Parity #11605](https://github.com/D-sorganization/UpstreamDrift/issues/11605), [Matched Swing #10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363), [Simscape #9921](https://github.com/D-sorganization/UpstreamDrift/issues/9921) and [Fast Pinocchio #10430](https://github.com/D-sorganization/UpstreamDrift/issues/10430). Their acceptance authorities remain binding. The 2026-09-28 MMR board packet is historical proposal evidence, not implementation approval. This epic adds distributed controllers, tuning/coupling, robust-control orchestration and the muscle endpoint, while reusing estimator and replay infrastructure.

Success is a native, uninterrupted full-horizon rollout with declared actuator/contact modes, frozen acceptance metrics, measured time-to-accepted-match, reproducible artifacts and a clean-host rerun. Both driver and iron are required where the existing program requires them; missing observations or model capabilities stay explicit blockers. A closed-loop match, a low collocation defect or a rendered movie alone does not satisfy independent forward dynamics.

## Governing Equations and Assumptions

Use generalized configuration q on its manifold, velocity v, muscle activation a, muscle-tendon states z, input u and physical parameters theta. Let x=(q,v,a,z) for muscle models and x=(q,v) for torque-only models. Rotations use model-native manifold operations; nq need not equal nv.

`qdot = N(q) v`

`M(q;theta) vdot + h(q,v;theta) = B(q) tau_act(x,u;theta) + J_c(q)^T lambda + tau_external`

`adot = f_activation(a,e;theta), zdot = f_tendon(q,v,a,z;theta)`

Torque models use bounded applied actuator torques; muscle models use excitations `0 <= e <= 1`, with activations and tendon/fiber states evolving through the chosen model. Muscle generalized forces depend on moment arms, force-length-velocity and passive forces. Validate whether an engine reports actuator-space inputs, generalized efforts, reserve drives or constraint forces; never count a constraint reaction twice.

Observations are `y_i = H_i(x,theta)+epsilon_i`, with declared frames, SI units, clock alignment, occlusion masks and covariance. Subject mass/inertia/marker attachment uncertainty is distinct from tracking error. Kinematics alone generally cannot identify absolute mass, contact-force distribution, muscle recruitment or a unique feedback law. Record anchors, priors, rank/observable subspaces and uncertainty; do not fit gains to compensate for arbitrary geometry errors.

Rigid contact uses formulation-appropriate unilateral and complementarity checks. Compliant contact permits law-consistent bounded deformation and requires nonadhesive behavior where appropriate, friction/dissipation and numerical convergence checks. Grip closure and club compliance must be declared and checked. Begin with known/reviewed contact phases and existing validated contact laws. Contact-implicit methods are a benchmarked research option, not an assumed faster default. A measured-GRF-driven replay is labeled **externally forced**; it cannot pass autonomous-contact prediction.

## Optimization and Control Architecture

1. **Calibrate and Freeze Evidence.** Inventory available channels; align clocks/events; calibrate marker attachments/anthropometry on a separate split where possible. Preserve original observations and report filtering and differentiated-signal uncertainty. Whiten only valid measurements with a documented covariance model.
2. **Warm Start.** Reuse MOSAIC/reference IK and dynamically consistent inverse dynamics. Preserve the mass gauge, physical inertias, unactuated root and explicit residual wrenches. Do not claim an inverse-dynamics reconstruction is a forward prediction.
3. **Optimize Forward-Dynamic Tracking.** Compare existing multiple shooting/FDDP with sparse implicit inverse-dynamics collocation and MocoTrack for muscle models. Minimize robust marker/club/GRF residuals plus effort, control-rate, residual-drive and contact terms, subject to dynamics and actuation/physiology/closure constraints. Report hard-constraint feasibility separately from cost. Adapt mesh only with bounded, recorded rules; validate gradients against independent differences on synthetic smooth fixtures.
4. **Add Distributed Feedback.** Use nominal feedforward `u_ff` plus local joint/task feedback. In a fixed reference sign convention, `u = allocate_and_bound(u_ff - K_t * difference(x_ref,x) + u_task)`. Define the difference so the signed restoring response is tested. Gain dimensions follow tangent-state coordinates; gains must respect actuator mapping and saturation. TVLQR/Riccati gains reuse MOSAIC where valid. Handle task priority and actuator competition through one allocation boundary, not independent additive commands with hidden clipping.
5. **Tune and Reconcile Loops.** Configure feet/pelvis, trunk, shoulders/arms, wrists/hands and club tasks. Block-coordinate tuning is an initialization strategy: iterate blocks with explicit trust regions, track objective/constraint degradation, then jointly refine. The grouping and order are hypotheses, not assumed independent human control systems. Optimize gains/weights over multiple trials with held-out evaluation; record robust feasibility under perturbation.
6. **Optional NMPC.** Add warm-started constrained receding-horizon control only if its tracking/robustness benefit justifies measured latency. Fixed-budget timeout, infeasibility and failed-step behavior are public contracts; a verified stabilizing fallback is explicit. Compare against TVLQR/task impedance and predictive sampling where existing backends support it. Do not promise real time before hardware-specific evidence.
7. **Translate to Muscles.** Use MocoInverse/CMC only as prescribed-motion diagnostics or warm starts. Use MocoTrack or MocoStudy for dynamically free muscle-driven tracking and joint excitation optimization over time. Per-frame torque-to-muscle allocation can be dynamically infeasible and is insufficient. Check the installed Moco version's Controller support: some versioned guides disallow OpenSim Controllers inside optimization, so use compatible problem terms or an outer forward-simulation controller.
8. **Verify Pure Replay.** Independently instantiate the engine, load only the frozen initial state, model and applied-input bundle, disable feedback and measurement access, and integrate continuously. Do not reset states at windows, overwrite q/v, stabilize by measured trajectories or rerun the controller offline during replay.

The optional bounded search implementation has one numerical boundary: a finite candidate plan is evaluated for a finite cost and a vector of signed hard margins (nonnegative means feasible). The existing F02 torque rollout supplies the first evaluator; native muscle/motor adapters must supply their own state scaling, exact post-mapping command ordering, physics, source identity and hard criteria. The shared SLSQP loop enforces plan bounds, counts prediction and vector-constraint preparation calls against one cooperative budget, then independently re-evaluates a proposed solution before returning it. A returned plan is only a numerical proposal. The caller separately admits a safe fallback and must promote a candidate through fresh native execution and independent saved-input replay before making a physical matching claim. The loop does not supply marker attachments, physiology, contact validity or a hard real-time guarantee.

## OpenSim Muscle Endpoint

Before any controller or muscle parameter tuning, run the F07f native source
reference observer on each exact source artifact and a declared post-initSystem
or complete named continuous state at the initialized clock. Record achieved
versus serialized coordinate defaults because OpenSim `initSystem()` itself
assembles. Preserve frame transforms, joint/coupler coordinate paths, moving and
conditional muscle path points, native wrap curves, concrete law/options and
source/loaded/runtime identities. Cross-source diagnostics require explicit
candidate path pairs and a declared rigid frame registration; scalar coordinate
sign/axis equivalence and anatomical correspondence require separate evidence.
This observer is read-only and cannot substitute for source-resource closure,
physiological limits, native full-state replay, contact/grip or capture fit.

Pin OpenSim/Moco versions and subject-specific muscle geometry, maximum isometric force, fiber/tendon lengths, activation constants and tendon-compliance choice. Sensitivity-test plausible parameter ranges; rank-deficient parameters retain priors. Reserve and root-residual actuators have separately frozen bounds and effort reports. No muscle-only claim if hidden reserves do substantial work. EMG, when actually present and synchronized, can constrain an uncertain recruitment objective; lack of EMG limits physiology validation and does not justify invented excitation truth.

Export excitation histories **and** complete initial activations/fiber/tendon/other model states, actuator order and unit conventions, equilibrium procedure if used, final fitted parameters, contact model, external loads, solver and interpolation. Audit resulting activation, muscle force, tendon/fiber trajectories, generalized muscle torque, reserves and reaction forces. A fresh OpenSim forward simulation must reproduce the fit within ratified numerical tolerances; independently compare markers, club motion/events, contact and muscle/actuator bounds. The strongest gate uses the model's own ground contact without recorded GRF forcing. If unavailable, retain a separately labeled externally forced milestone and leave autonomous muscle-driven matching open.

## Applied-Input and Parity Contract

For torque lanes, inherit #11605: evaluate controller once per integration interval and hold its **post-allocation, post-saturation applied input** over the full interval; log that exact ZOH sequence. A single sample per mocap frame from an RK-stage-varying controller is not the same input. If a backend uses another policy, define and validate it as a distinct mode before comparison. Muscle excitations have a separately versioned interpolation/time-grid policy; preserve actuator internal dynamics.

Each bundle includes schema/version, full x0, nominal/reference distinction, ordered actuator IDs, units/frames/timebase, model/provider revisions, input history, parameters/contact/loads, integration/constraint-projection policy, random seed and evidence references. Private source provenance remains private. Reject stale hashes, incompatible mappings, missing states, finite-value failures and unsupported capability combinations.

| Parity Level                             | Requirement                                                                                                  | Limitation                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| P0: Evidence and Model Identity          | Same observation identity, clocks, frame maps, geometry/inertia/constraints where applicable                 | Different muscle/contact models must be declared                        |
| P1: Pointwise Dynamics                   | Compare FK, M, bias, constrained acceleration and mapped effort at consistent states                         | Use existing #11605 tolerances; no silent weakening                     |
| P2: Within-Engine Input Replay           | Separate input identity, same-policy reproduction, native transcription feasibility and observation accuracy | Fresh engine; no feedback, state reset or measurement access            |
| P3: Cross-Engine Torque Parity           | Same mapped model and torque bundle with declared solver/contact policy                                      | Distinguish numerical-policy differences from physics differences       |
| P4: Muscle/Biomechanical Equivalence     | Match observation/events/contact/net torque and bounded effort across differing actuator formulations        | Excitation equality is meaningful only for identical muscle definitions |
| P5: Scientific and Product Qualification | Full horizon, applicable clubs, physical/physiological gates and reproducible user workflow                  | Software tests and videos alone are insufficient                        |

The matrix has a row for every registered model/engine/drive-mode pair, with `unsupported`, `unavailable`, `implemented-unqualified` or `qualified` plus evidence. MyoSuite is not automatically the same musculoskeletal model as OpenSim. Maintain best defensible parity without calling nonidentical models identical.

## Coupling and Reduced Control Systems

Compute normalized cross-sensitivities of task residuals to control/gain blocks, off-diagonal Gauss–Newton/Hessian blocks and empirical perturbation responses. Include uncertainty, time/phase dependence, scaling and conditioning. Covariance/correlation and partial correlation are descriptive; common phase/constraints can create them without causal coupling. Use intervention-style synthetic perturbations and repeated-trial evidence to test candidate groupings. Compare reduced task controllers against the full controller on accuracy, robustness, effort and wall time; retain groups only when held-out evidence supports simplification. A control synergy is a useful model abstraction, not identified neural anatomy.

## Benchmarks and Acceptance

Freeze baseline and gate versions before data fitting. Compare independent-polynomial baseline, current computed-torque tracking, MOSAIC feedforward+TVLQR, distributed tuned feedback, selected constrained OCP, optional NMPC and OpenSim muscle tracking on identical inputs and budgets. Include analytic pendulum/manipulator fixtures, floating-base contact fixture, a small muscle model and actual private captures under approved access. Separate calibration, tuning and held-out trials; split by trial/subject when possible, not adjacent frames. If insufficient independent trials exist, report that generalization cannot be established.

Report marker RMSE by group and phase, maximum/p95 residual, club pose/speed/events, constraint/penetration/slip/GRF error, joint/actuator/physiology bounds, residual/reserve effort, open-loop drift and horizon, torque/excitation audit, uncertainty, success rate and failures. Report cold/warm compile/load/solve/replay/render times, p50/p95 over a predeclared repeat count, peak memory/VRAM and total time-to-accepted-match on named hardware. Acceptance numbers inherit current authorities; new muscle, robust-control and timing thresholds are ratified from sensor noise, integration refinement, synthetic truth and baseline measurements before evaluation. Do not select thresholds after seeing favorable fits.

## Repository Boundaries and Reuse

**UpstreamDrift:** extend `src/shared/python/motion_matching/__init__.py`, `pipeline/dynamics.py`, plant protocols, `multi_shooting_fit.py`, `cross_engine_replay.py`, receipt modules and `estimation/mosaic/`. Inspect current exports/source and provider pins before edits. Golf anatomy, tasks, OCP, muscle adaptation and acceptance stay here. Extend existing native-engine tests, receipt tests and motion-matching integration tests; preserve feature-parity and qualification registries.

**Tools:** extend `src/shared/python/sidekick/lab/mocap/` and existing schemas for general experiment identity, actuation capability, time/frame compatibility, replay/provenance and resource/preview contracts only. No swing OCP or duplicated physiology solver. Test stable APIs and pinned downstream consumption, including Gasification_Model compatibility where shared surfaces require it. Never edit `vendor/ud-tools` in UpstreamDrift.

**Private Evidence Workstream:** authoritative raw inventory, source/derived mappings, real channel availability, splits, private validation receipts and sensitive previews. Public synthetic fixtures must be independently generated. No originals, identifying filenames, private paths or restricted provenance are copied into public documentation.

## Engineering, Documentation and Turnover

TDD means commit a meaningful failing behavioral test before implementation, record the red and green commands, then refactor. Cover saturation signs, time-grid mismatch, actuator permutation, missing muscle state, hidden reserve/root drives, replay measurement access, false parity and timeout/cancellation. Use independent analytic/truth or convergence checks rather than tests that restate implementation. Native integration and physiological claims require actual engines.

DbC validates types, units, clocks, frames, finite values, positive time steps, manifold/actuator dimensions, physiological bounds and provenance. Use existing contract helpers with descriptive exceptions. LoD exposes plant/controller/allocator/receipt facades; optimizers do not reach into engine internals or UI state. DRY centralizes authoritative schemas, conversion and replay policy. Respect native source APIs and versioned capability probes.

Every implementation PR updates canonical manual QMD for changed calculations and the governed registries, calculation references as required, SPEC/change fragments and canonical handoff. This planning document is not a substitute for the qualified design manual. Turnover includes exact branch/commit/provider/PR state, data authority, commands, seeds, versions, host/license needs, receipt paths, current gates, failures, blockers and next bounded task. Never label the entire epic done merely because code/children are closed.

## Preview, Storage and Fleet Execution

Local preview root: `%USERPROFILE%\Desktop\Motion_Matching_Previews`. Use per-run directories with privacy designation, a manifest recording engine/model/drive mode, input/receipt identity, time coverage, render command and qualification status. Show synchronized observation overlays and residuals; label feedback, torque replay, excitation replay and externally forced modes. No videos have been generated for this planning deliverable.

Before jobs, measure free disk, estimate source/temporary/checkpoint/video sizes and reserve an explicit budget. Render short lower-resolution review previews before optional final exports; avoid redundant raw copies. Bound workers, GPU memory, checkpoints and cache by complete physical/provider identity. Preserve source captures and canonical receipts. Remove only this task's merged, clean worktrees after verifying merge ancestry and preserving needed ignored artifacts; use managed archive tooling for managed worktrees. Unmerged branches/worktrees stay available for review.

Tailscale fleet execution is optional and access-controlled. Inspect host capacity, storage, dependency/provider versions, required MATLAB R2025b/license and private-data access before dispatch; do not infer availability from a machine's name. Prefer sending manifests/jobs to authorized data rather than replicating raw data. Record host, paths, environment, resource budget and output destinations; verify returned hashes and cold native replay. Do not install packages, kill other users' jobs or alter remote configuration merely to obtain capacity.

## Evidence-Based Method Selection

Primary literature reviewed through 2026-10-08; applicability to golf is an engineering inference unless explicitly demonstrated. Recent robotics results inform bounded experiments, not promised golf performance.

| Source                                                                                                                                                                                                            | Result and Design Implication                                              | Limit                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [OpenSim Moco, 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493)                                                                                                              | Established direct-collocation musculoskeletal optimal-control foundation  | Solver success is not independent replay                        |
| [MocoInverse Guide](https://opensim-org.github.io/opensim-moco-site/docs/1.1.0/html_user/mocoinverse.html)                                                                                                        | Prescribed-kinematics recruitment/diagnostic warm start                    | Cannot qualify free motion prediction                           |
| [MocoTrack Guide](https://opensim-org.github.io/opensim-moco-site/docs/1.1.0/html_user/mocotrack.html) and [MocoStudy Guide](https://opensim-org.github.io/opensim-moco-site/docs/1.3.0/html_user/mocostudy.html) | Tracking with free kinematics and custom objectives/constraints            | Pin installed versions and capabilities                         |
| [Moco 1.2 User Guide](https://opensim-org.github.io/opensim-moco-site/docs/1.2.0/html_user/mocouserguide.html)                                                                                                    | Documents optimization restrictions including Controllers                  | Recheck exact installed version; forward simulator differs      |
| [Shooting/Collocation Comparison, 2023](https://arxiv.org/abs/2302.07645)                                                                                                                                         | Transcription tradeoffs and forward reintegration motivate benchmark gates | No universally fastest formulation                              |
| [Crocoddyl, 2020](https://arxiv.org/abs/1909.04947) and [Box-FDDP](https://arxiv.org/abs/2010.00411)                                                                                                              | Structured contact OCP and bounded-control feedback candidates             | Use existing adapters; contact sequence assumptions matter      |
| [acados Official Features](https://docs.acados.org/features/index.html)                                                                                                                                           | Structured SQP/RTI and warm-started NMPC options                           | Runtime/dependency spike before adoption                        |
| [MuJoCo MPC Official Overview](https://github.com/google-deepmind/mujoco_mpc/blob/main/docs/OVERVIEW.md)                                                                                                          | iLQG and predictive-sampling comparison options                            | Not automatically a golf or muscle solution                     |
| [Marker/GRF Optimal Tracking, 2023](https://peerj.com/articles/14852/)                                                                                                                                            | Sports-motion evidence for direct observation tracking                     | Running study, not a golf validation                            |
| [PRIME, RSS 2026](https://www.roboticsproceedings.org/rss22/p029.html)                                                                                                                                            | Joint state/contact/inertia estimation informs calibration experiments     | Legged-robot evidence; reuse MOSAIC authority                   |
| [IMPACT, RSS 2026](https://www.roboticsproceedings.org/rss22/p163.html) and [CRISP, RSS 2025](https://roboticsproceedings.org/rss21/p047.html)                                                                    | Recent contact-implicit alternatives for a research spike                  | Authors' benchmarks do not establish golf speed                 |
| [Diff-MSM, 2025 Preprint](https://arxiv.org/abs/2508.13303)                                                                                                                                                       | Differentiable musculoskeletal parameter-estimation direction              | Limited simulated-arm evidence                                  |
| [KINESIS, 2025 Preprint](https://arxiv.org/abs/2503.14637)                                                                                                                                                        | Learned musculoskeletal imitation is a later comparator                    | Training/recruitment claims do not replace deterministic replay |
| [Golf Ground-Reaction Study](https://pubmed.ncbi.nlm.nih.gov/31042142/)                                                                                                                                           | Supports explicit foot-force/moment and club-speed analysis                | Association does not establish controller causality             |
| [Muscle-Synergy Identifiability Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3582774/)                                                                                                                         | Mechanical/task constraints can confound inferred modules                  | Covariance is not a neural-loop identifier                      |

## Open Decisions and First Slice

First freeze the current native capability matrix, source/provider versions, existing parity tolerances and available private channels. Select the first small torque/contact and muscle fixtures; ratify missing muscle and runtime gates before human fitting. Assign accountable implementation owners only through the fleet claim/lease process. The child plan defines dependencies; this packet does not automatically dispatch expensive simulations or claim board/scientific approval.

## Review Clarifications

Adopt the existing canonical model/variant/capability registry and stable engine/variant/drive-mode keys; F01 freezes baseline/policy and F09 executes ongoing conformance. #11605/#11607 retain replay authority. Tools validates generic evidence interchange; UpstreamDrift decides physics and qualification. Replay APIs accept a frozen bundle and native plant, without controller/observation providers. F07 supplies the complete pinned muscle-model/state/contact/forward-API handoff to F08. Optional NMPC needs a report/disposition and does not block acceptance of a qualified simpler controller. Public preview manifests omit local paths; resolve configured artifacts through MOTION_MATCHING_PREVIEW_ROOT, and preserve shared/user-owned directories and videos.

## Binding Architecture Review

[Astra Architecture Review](ASTRA_REVIEW.md) refines this design and its acceptance criteria. Record dynamics/contact/integration/actuator/restart implementations; common-model parity cannot qualify native muscle/contact replay. Required capabilities remain required when unavailable. Version full state, manifold operations, input boundaries and actual executed policies rather than silently broadening Euclidean torque fixtures.

An optimized collocation trajectory must pass uninterrupted native integration and refinement checks. Score frozen nominal feedforward separately from replay of total feedback-generated input. Reserve/root assistance requires peak/RMS, integrated absolute effort and separate positive/negative work; zero signed work cannot prove muscle-only operation. Declare exact versus noisy/estimated/delayed state and validate that information pattern. Gain and recruitment estimates remain nonunique and require held-out perturbation and parameter/objective sensitivity evidence.

## Constrained Mixed OpenSim Moco Boundary (#12173)

The narrowly tested OpenSim composition is opt-in: a caller supplies an exact
declared cold start, lock targets, constraint enforcement, coordinate charts
and optional linear chart inequalities alongside the mixed muscle/mechanical
profile. The native source constraints are retained and observed at each
replay knot. Moco coordinate boxes must fit the declared chart; linear rules
are checked by conservative interval extrema. A fresh replay rejects native
Manager projection of the exact requested named-state seed or initial time.
Mixed commands remain distinct from physical outputs: muscle excitation is
dimensionless with nonlinear force in N, while CoordinateActuator commands
are scaled into N or N\*m according to native coordinate motion. The strict
muscle-only default is unchanged. The 40 ms synthetic Moco/replay regression
is software-path evidence, not a private capture, production anatomy/physics
qualification, or release. See canonical chapter48 and
`F07_CONSTRAINED_MIXED_MOCO_TURNOVER.md`.

## Exact-Source 557-Muscle Replay Boundary (#12182)

An opt-in version-three profile freezes the exact BUET–Hamner derived model
bytes and native component inventory instead of widening the reviewed v1/v2
class allowlists. Every clamped source range and the four source-coupled
rotational coordinates have explicit units and chart declarations. Attached
visual meshes are inventoried separately from dynamic resources. The
prepared named state, native options, ordered 557 excitations and constraint
policy are bound in T01; the shared scalar executor rejects initial
projection and independently observes all 17 couplers at each knot. The
original equilibrium seed fails the SC_z chart after 1 ms. A distinct
six-coordinate interior diagnostic seed has only 0.2 ms replay evidence;
neither result certifies source inertia, passive physiology, full native
restart or a captured swing. Chapter50 and the exact replay turnover bind
the inputs, equations, failures and acceptance limits.

## Native Source-Body Inertia Admission (#12191)

The OpenSim source-body gate reads exact BodySet properties and checks a fresh
native model, without equilibrating, integrating or editing parameters. It
uses the shared physical-inertia validator for positive masses and explicitly
admits native welded/internal routing carriers only when both mass and all
inertia components are exactly zero. Inertia is about the body-frame COM;
native mass, COM and six components must equal the source. The unchanged
557-muscle source fails the necessary principal-moment condition on both
clavicles and both scapulae. The gate records all failing bodies and keeps
the #12182 short replay diagnostic separate from physical admission. Chapter52
and the public receipt bind source/runtime/adapter identities; source anatomy,
passive forces, contact/grip and capture matching remain unqualified.
