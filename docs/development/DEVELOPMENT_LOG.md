# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-12 by claude
- **Current metadata review:** Last verified commit identifiers below bind path inventory only to e68baeabde8ff639f846a167ca7703ace43bd0a9; no tests or native runs were performed. Full original row text before narrative condensation is retained in [the pre-compression snapshot](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_PRE_COMPRESSION_20261011.md.snapshot). Prior verification wording and per-path Git blob identities are preserved in [the source metadata audit](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json).

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#12712 · NSX-83: Add the ClubheadSeries to RigidClubState Adapter and Constant-Twist Event Location Feeding the Shared Impact Outcome Map.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12712
- **Branch:** claude/ud-12712-1
- **PR:** #14019
- **Paths:** see #14019
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`3a0aba29`; collated from changes/12712-nsx-83-add-the-clubheadseries-to-rigidcl.md)
- **Summary:** NSX-83: add the ClubheadSeries to RigidClubState adapter and constant-twist event location feeding the shared impact outcome map.
- **Next step:** Merge the PR; then wire NSX-90/91/92 to impact_outcome_map and add Tools launch-angle/spin-rate.

### DL-#14061 · Guard Xwininfo and Xwd Dependency Checks in OpenSim Visualizer Capture

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14061
- **Branch:** fix/14061-opensim-ci-xwininfo
- **PR:** #14062
- **Paths:** see #14062
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`4c39130c`; collated from changes/14061-opensim-xwininfo-guard.md)
- **Summary:** Guard xwininfo and xwd dependency checks in OpenSim visualizer capture
- **Next step:** Merge the PR.

### DL-#13174 · HRG-D3: Render_Golfer_Realism_Showcase --V2-Spec Renders Labelled Address/Top/Impact Stills of the Full-Body-V2 Trunk Visuals; GL Stills Still Need a GL Host

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13174
- **Branch:** claude/ud-13174-1
- **PR:** #14035
- **Paths:** see #14035
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`6584e56c`; collated from changes/13174-v2-trunk-stills.md)
- **Summary:** HRG-D3: render_golfer_realism_showcase --v2-spec renders labelled address/top/impact stills of the full-body-v2 trunk visuals; GL stills still need a GL host
- **Next step:** Run the stills command on a GL host with MuJoCo and attach the three PNGs to the issue.

### DL-#14053 · Correct 6X6 Mobility Scaling by Multiplying Angular Rows by L

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14053
- **Branch:** fix/14053-mobility-scales
- **PR:** #14058
- **Paths:** see #14058
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`d5705ebb`; collated from changes/14053-correct-6x6-mobility-scaling-by-multiply.md)
- **Summary:** Correct 6x6 mobility scaling by multiplying angular rows by L
- **Next step:** Merge the PR.

### DL-#13971 · Drake Motion Matching: Catalog Finite-Difference Generalized Accelerations in Simulate_With_Coefficients (UD-36)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13971
- **Branch:** test/13971-catalog-finite-difference-qdd-drake
- **PR:** #13982
- **Paths:** see #13982
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`934b44df`; collated from changes/13971-catalog-finite-difference-qdd-drake.md)
- **Summary:** Drake motion matching: catalog finite-difference generalized accelerations in simulate_with_coefficients (UD-36)
- **Next step:** Merge PR

### DL-#13970 · Catalog Constant Static Friction Contract in Contact_Law_Kernel for Torsional_Moment_Kernel and Link Epic #13757 (UD-33)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13970
- **Branch:** test/13970-catalog-torsional-moment-static-friction
- **PR:** #13980
- **Paths:** see #13980
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`eac54cbe`; collated from changes/13970-catalog-torsional-moment-static-friction.md)
- **Summary:** Catalog constant static friction contract in contact_law_kernel for torsional_moment_kernel and link Epic #13757 (UD-33)
- **Next step:** Develop coupled slip-spin friction limit surface under Epic #13757 (HRG-HX)

### DL-#13972 · OpenSim Spine: Catalog Coordinate Unit-Speed Probing in Inverse_Dynamics _U_Indices (UD-39)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13972
- **Branch:** test/13972-catalog-coordinate-unit-speed-probing
- **PR:** #13983
- **Paths:** see #13983
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`03f7e9b2`; collated from changes/13972-catalog-coordinate-unit-speed-probing.md)
- **Summary:** OpenSim spine: catalog coordinate unit-speed probing in inverse_dynamics _u_indices (UD-39)
- **Next step:** Merge PR

### DL-#13969 · Clarify VelocityRepresentation Origin Convention in Docstrings, Documenting Evaluation at Base Reference Point / Center of Mass vs Featherstone World Origin O and Linking Epic #13789 (UD-13)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13969
- **Branch:** doc/13969-clarify-inertial-spatial-velocity-origin
- **PR:** #13976
- **Paths:** see #13976
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`2ba8eb82`; collated from changes/13969-clarify-inertial-spatial-velocity-origin.md)
- **Summary:** Clarify VelocityRepresentation origin convention in docstrings, documenting evaluation at base reference point / center of mass vs Featherstone world origin O and linking Epic #13789 (UD-13)
- **Next step:** Address kinematic coordinate transformations for bodies displaced from world origin under Epic #13789

### DL-#13967 · Enforce Engine Availability Assertion and Cite Simscape Baseline Prerequisite (UD-26)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13967
- **Branch:** test/13967-cross-engine-parity-guards
- **PR:** #13973
- **Paths:** see #13973
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`7ec71954`; collated from changes/13967-cross-engine-parity-guards.md)
- **Summary:** Enforce engine availability assertion and cite Simscape baseline prerequisite (UD-26)
- **Next step:** Merge PR

### DL-#13968 · Clarify and Assert Zero-Linear-Velocity Contract in Single_Floating_Body_H_G, Documenting Zero Linear Coriolis Bias Under Center-of-Mass Frame Conventions and Linking Epic #13789 (UD-14)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13968
- **Branch:** doc/13968-single-floating-body-linear-velocity-contract
- **PR:** #13975
- **Paths:** see #13975
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`7296e223`; collated from changes/13968-single-floating-body-linear-velocity-contract.md)
- **Summary:** Clarify and assert zero-linear-velocity contract in single_floating_body_h_g, documenting zero linear Coriolis bias under center-of-mass frame conventions and linking Epic #13789 (UD-14)
- **Next step:** Address non-zero linear velocity sensitivity contracts in multi-body floating base models under Epic #13789

### DL-#13962 · Require _Advance_to to Reach Target Time Within Substep Budget (UD-37)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13962
- **Branch:** fix/13962-mujoco-advance-to-substep-budget
- **PR:** #13963
- **Paths:** see #13963
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`5e855a22`; collated from changes/13962-mujoco-advance-to-substep-budget.md)
- **Summary:** Require _advance_to to reach target time within substep budget (UD-37)
- **Next step:** Merge PR

### DL-#13959 · Enforce Both Upper Bound and Zero Transverse Force in Check_Static_Equilibrium (UD-30)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13959
- **Branch:** fix/13959-grip-static-equilibrium
- **PR:** #13960
- **Paths:** see #13960
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`15e55478`; collated from changes/13959-grip-static-equilibrium.md)
- **Summary:** Enforce both upper bound and zero transverse force in check_static_equilibrium (UD-30)
- **Next step:** Merge the PR into main.

### DL-#13956 · Allow MacDonaldHanzelyModel and Constant-Coefficient Models to Use Declared Cl Without 0.155 Constant-Spin Clamp

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13956
- **Branch:** fix/13956-flight-models-lift-cap
- **PR:** #13958
- **Paths:** see #13958
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`eb8b7140`; collated from changes/13956-flight-models-lift-cap.md)
- **Summary:** Allow MacDonaldHanzelyModel and constant-coefficient models to use declared cl without 0.155 constant-spin clamp
- **Next step:** Merge the PR into main.

### DL-#13953 · OpenSim: Correct SimTK Quaternion Ordering Documentation in Coord_Map (UD-12)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13953
- **Branch:** fix/13953-opensim-simtk-quat-docs
- **PR:** #13954
- **Paths:** see #13954
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`f838b4e2`; collated from changes/13953-opensim-simtk-quat-docs.md)
- **Summary:** OpenSim: correct SimTK quaternion ordering documentation in coord_map (UD-12)
- **Next step:** Merge PR

### DL-#13949 · Pinocchio Viz: Adjust 1-Based Impact_Idx to 0-Based Indexing in Error_Timecourse (UD-23)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13949
- **Branch:** fix/13949-pinocchio-viz-impact-idx
- **PR:** #13950
- **Paths:** see #13950
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`2f369c5a`; collated from changes/13949-pinocchio-viz-impact-idx.md)
- **Summary:** Pinocchio viz: adjust 1-based impact_idx to 0-based indexing in error_timecourse (UD-23)
- **Next step:** Merge PR

### DL-#13947 · Restore Docstrings in Motion_Matching Cost Term Helpers (UD-20)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13947
- **Branch:** fix/13947-cost-term-docstrings
- **PR:** #13948
- **Paths:** see #13948
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`b5182571`; collated from changes/13947-cost-term-docstrings.md)
- **Summary:** Restore docstrings in motion_matching cost term helpers (UD-20)
- **Next step:** Merge PR

### DL-#13869 · Architecture Budget: Full-Scan Mode With Checked-In Baseline Ratchet (UD-25)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13869
- **Branch:** fix/13869-architecture-budget-baseline
- **PR:** #13934
- **Paths:** see #13934
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`d652ae11`; collated from changes/13869-architecture-budget-baseline.md)
- **Summary:** Architecture budget: full-scan mode with checked-in baseline ratchet (UD-25)
- **Next step:** Merge PR

### DL-#13868 · Motion Matching: Architecture Guard and Canonical _Geodesic Helpers (UD-24)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13868
- **Branch:** fix/13868-quaternion-helper-guard
- **PR:** #13933
- **Paths:** see #13933
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`d86cf2e5`; collated from changes/13868-quaternion-helper-guard.md)
- **Summary:** Motion matching: architecture guard and canonical _geodesic helpers (UD-24)
- **Next step:** Merge PR

### DL-#13761 · HRG-HX4 Part 2: Bushing_Wrench Force_Limit_N Caps the Bushing Force Magnitude, Direction Kept, Moment Unchanged, Default None Unlimited and Bit-Identical (Option C, Python Bushing Law Only)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13761
- **Branch:** claude/13761-bushing-force-cap
- **PR:** #13941, #13939, #14033
- **Paths:** see #13941
- **Started:** 2026-10-27
- **Last verified:** 2026-10-29 (`6c3b0bea`; collated from changes/13761-hrg-hx4-part-3-bushing-force-limit-wiring.md)
- **Summary:** HRG-HX4 part 2: bushing_wrench force_limit_n caps the bushing force magnitude, direction kept, moment unchanged, default None unlimited and bit-identical (option c, Python bushing law only)
- **Next step:** Merged and closes #13761

### DL-#13935 · Bump Dnspython 2.8.0 to 2.9.0 in Both Lock Files to Clear CVE-2026-114605 From Pip-Audit

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13935
- **Branch:** claude/13935-dnspython-cve
- **PR:** #13936
- **Paths:** see #13936
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`725b14e0`; collated from changes/13935-bump-dnspython-2-8-0-to-2-9-0-in-both-lo.md)
- **Summary:** Bump dnspython 2.8.0 to 2.9.0 in both lock files to clear CVE-2026-114605 from pip-audit
- **Next step:** Merge the PR.

### DL-#13872 · Pinocchio Residual: Append Regulariser Row and Impose Torque Bounds (UD-02)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13872
- **Branch:** fix/13872-pinocchio-residual-regularizer
- **PR:** #13927
- **Paths:** see #13927
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`e9825c55`; collated from changes/13872-pinocchio-residual-regularizer.md)
- **Summary:** Pinocchio residual: append regulariser row and impose torque bounds (UD-02)
- **Next step:** Merge PR

### DL-#13903 · EnhancedBallFlightSimulator: Route Deterministic Aero Through Rust Facade (UD-03)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13903
- **Branch:** fix/13903-enhanced-simulator-rust-facade
- **PR:** #13929
- **Paths:** see #13929
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`19f532dd`; collated from changes/13903-enhanced-simulator-rust-facade.md)
- **Summary:** EnhancedBallFlightSimulator: route deterministic aero through Rust facade (UD-03)
- **Next step:** Merge PR

### DL-#13874 · Unify Solver_Status Contract and Provenance Across All Four Physics Engines

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13874
- **Branch:** fix/13874-solver-status-contract
- **PR:** #13920
- **Paths:** see #13920
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`28be4a7a`; collated from changes/13874-solver-status-contract.md)
- **Summary:** Unify solver_status contract and provenance across all four physics engines
- **Next step:** Merge PR

### DL-#13871 · Project Pinocchio Analytical Orientation Jacobian Onto Relative Rotation Axis (UD-01)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13871
- **Branch:** fix/13871-pinocchio-orientation-jacobian
- **PR:** #13924
- **Paths:** see #13924
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`29f340df`; collated from changes/13871-pinocchio-orientation-jacobian.md)
- **Summary:** Project Pinocchio analytical orientation Jacobian onto relative rotation axis (UD-01)
- **Next step:** Merge PR

### DL-#13876 · Tighten MuJoCo Mj_Inverse Test Tolerances and Replay Refinement (UD-15, UD-16)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13876
- **Branch:** fix/13876-mujoco-mj-inverse-tolerances
- **PR:** #13925
- **Paths:** see #13925
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`70be2a3e`; collated from changes/13876-mujoco-mj-inverse-tolerances.md)
- **Summary:** Tighten MuJoCo mj_inverse test tolerances and replay refinement (UD-15, UD-16)
- **Next step:** Merge PR

### DL-#13867 · Branch Orientation Cost on Q_Orientation_Repr and Enforce Body_Marker Precondition

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13867
- **Branch:** fix/13867-q-orientation-repr-and-body-marker-term
- **PR:** #13917
- **Paths:** see #13917
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`dd68977a`; collated from changes/13867-q-orientation-repr-and-body-marker-term.md)
- **Summary:** Branch orientation cost on q_orientation_repr and enforce body_marker precondition
- **Next step:** Merge PR

### DL-#13866 · Enforce 1E-6 Relative Tolerance in Ball Flight Rust/Python Parity Test

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13866
- **Branch:** fix/13866-ball-flight-parity-rel-1e6
- **PR:** #13921
- **Paths:** see #13921
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`517f1074`; collated from changes/13866-ball-flight-parity-rel-1e6.md)
- **Summary:** Enforce 1e-6 relative tolerance in ball flight Rust/Python parity test
- **Next step:** Merge PR

### DL-#13902 · Remove Unused Signature Bind and Memoize Inspect.Signature in Contract Decorators

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13902
- **Branch:** fix/13902-precondition-signature-overhead
- **PR:** #13919
- **Paths:** see #13919
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`8a1791c8`; collated from changes/13902-dbc-decorator-overhead.md)
- **Summary:** Remove unused signature bind and memoize inspect.signature in contract decorators
- **Next step:** Merge PR

### DL-#13873 · Soft-Fail Divergent Rollout Evaluations in MuJoCo Fit_Swing With Finite Penalty

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13873
- **Branch:** fix/13873-mujoco-divergent-rollout-soft-fail
- **PR:** #13916
- **Paths:** see #13916
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`4cf42e83`; collated from changes/13873-mujoco-divergent-rollout-soft-fail.md)
- **Summary:** Soft-fail divergent rollout evaluations in MuJoCo fit_swing with finite penalty
- **Next step:** Merge PR

### DL-#13879 · \ Route C3D Analog and Reader CoP Through Canonical Center of Pressure\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13879
- **Branch:** fix/13879-cop-c3d-analog-canonical
- **PR:** #13907
- **Paths:** see #13907
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`67d3db29`; collated from changes/13879-route-c3d-analog-and-reader-cop-through.md)
- **Summary:** \ Route C3D analog and reader CoP through canonical center of pressure\
- **Next step:** \Merge PR 13907\

### DL-#13877 · \ Fail Cross-Engine Validation When Deviation Exceeds Tolerance\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13877
- **Branch:** fix/13877-cross-engine-validator-tolerance
- **PR:** #13906
- **Paths:** see #13906
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`2bbbb020`; collated from changes/13877-fail-cross-engine-validation-when-deviat.md)
- **Summary:** \ Fail cross-engine validation when deviation exceeds tolerance\
- **Next step:** \Merge PR 13906\

### DL-#13837 · Option A MuJoCo Swing-Recovery Trace Fixture; Reuses Triple_Pendulum_Spec Segments in Humanoid_Robotics.Option_a.Plant.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13837
- **Branch:** cursor/humanoid-option-a-swing-recovery-f3e4
- **PR:** #13838
- **Paths:** see #13838
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`7c90712e`; collated from changes/13837-add-option-a-mujoco-swing-and-recovery-t.md)
- **Summary:** Option A MuJoCo swing-recovery trace fixture; reuses triple_pendulum_spec segments in humanoid_robotics.option_a.plant.
- **Next step:** Merge PR #13838 after CI is green.

### DL-#13880 · \ Zero Friction Cone Capacity in Tension\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13880
- **Branch:** fix/13880-friction-cone-tension
- **PR:** #13889
- **Paths:** see #13889
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`43d5476c`; collated from changes/13880-zero-friction-cone-capacity-in-tension.md)
- **Summary:** \ Zero friction cone capacity in tension\
- **Next step:** \Merge PR 13889\

### DL-#13865 · \ Rotate Backspin Axis With Azimuth and Align Sidespin With Vertical\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13865
- **Branch:** fix/13865-ball-flight-spin-vector
- **PR:** #13908
- **Paths:** see #13908
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`51776474`; collated from changes/13865-rotate-backspin-axis-with-azimuth-and-al.md)
- **Summary:** \ Rotate backspin axis with azimuth and align sidespin with vertical\
- **Next step:** \Open PR for UD 13865\

### DL-#13842 · Fast-Tier Test Selection Now Runs the Monolith Register Test for Src/ Changes

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13842
- **Branch:** fix/13842-fast-tier-monolith-register
- **PR:** #13843
- **Paths:** see #13843
- **Started:** 2026-10-26
- **Last verified:** 2026-10-26 (`b0dd190b`; collated from changes/13842-fast-tier-test-selection-now-runs-the-mo.md)
- **Summary:** Fast-tier test selection now runs the monolith register test for src/ changes
- **Next step:** Merge the PR.

### DL-#13780 · ML-WS-3: Capture-A Post-Impact Replay Audit Receipt and Tests

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13780
- **Branch:** cursor/ml-ws-3-replay-post-impact-1bfa
- **PR:** #13841
- **Paths:** see #13841
- **Started:** 2026-10-26
- **Last verified:** 2026-10-26 (`f85d5895`; collated from changes/13780-ml-ws-3-capture-a-post-impact-replay-aud.md)
- **Summary:** ML-WS-3: capture-A post-impact replay audit receipt and tests
- **Next step:** Merge PR #13841; open ML-WS-3b for follow-through reference branching.

### DL-#13762 · Specify HRG-HX5 Hand Proxy Decision and Implementation Acceptance Gates

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13762
- **Branch:** staff/issue-remediator-13762-b36738
- **PR:** #13817
- **Paths:** docs/adr/0057-hrg-hx5-articulated-hand-option.md, docs/adr/README.md, docs/development/full_body_models/HANDS_AUDIT.md
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`83720aa2`; collated from changes/13762-specify-hrg-hx5-hand-proxy-decision-and.md)
- **Summary:** Specify HRG-HX5 hand proxy decision and implementation acceptance gates
- **Next step:** Open a follow-up runtime issue after HX6 licensing clearance.

### DL-#13758 · Add SO(3) Observability Evidence for Grip Fitting

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13758
- **Branch:** staff/issue-remediator-13758-4adf95
- **PR:** #13816
- **Paths:** see #13816
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`652ea2e1`; collated from changes/13758-add-so-3-observability-evidence-for-grip.md)
- **Summary:** Add SO(3) observability evidence for grip fitting
- **Next step:** Use Jacobian evidence to select the neutral-prior fit change

### DL-#13656 · Pin Tools 160C9c, Sync Child APIs, Reconcile Impact Evidence, and Adapt Voice Routes

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13656
- **Branch:** codex/13656-knowledge-pin
- **PR:** #13657
- **Paths:** vendor/ud-tools,tests/unit/ai/test_vendor_knowledge_search_api.py
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`c920b8b8`; collated from changes/13656-knowledge-pack-sanitizer-pinned-tools.md)
- **Summary:** Pin Tools 160c9c, sync child APIs, reconcile impact evidence, and adapt voice routes
- **Next step:** Verify the pinned child copies, served impact bundle, sanitizer import, and voice routes in CI.

### DL-#13434 · OS-9 Re-Registers the Tour-Marker Model on the Address Hold With Anatomical Priors and a Plausibility Gate; OS-8 Trunk Angles Were From a Mirrored Registration

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13434
- **Branch:** fix/13434-static-address-registration
- **PR:** #13484
- **Paths:** see #13484
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`7ebc584c`; collated from changes/13434-os-9-re-registers-the-tour-marker-model.md)
- **Summary:** OS-9 re-registers the tour-marker model on the address hold with anatomical priors and a plausibility gate; OS-8 trunk angles were from a mirrored registration
- **Next step:** Owner: decide the address trunk screening bound (30.2 vs 30 deg, lumbar flexion); then re-run spine evidence #13427/#12769 on the OS-9 IK

### DL-#13791 · Route UD-Owned Python Gravity Values and MJCF Gravity Through the Canonical Constant, Guarded by a Tested Literal Scanner.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13791
- **Branch:** staff/issue-remediator-13791-0c6d70
- **PR:** #13799
- **Paths:** src Python gravity users, owned MuJoCo MJCF, tests/unit/core gravity guard
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`3f7dcf04`; collated from changes/13791-route-ud-owned-python-gravity-values-and.md)
- **Summary:** Route UD-owned Python gravity values and MJCF gravity through the canonical constant, guarded by a tested literal scanner.
- **Next step:** Review the ready PR; follow-up #13798 covers the deferred MATLAB/Simscape literals.

### DL-#13759 · Add Audited Per-Engine Hand Capability Matrix and Frame Contract

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13759
- **Branch:** staff/issue-remediator-13759-d7f0a0
- **PR:** #13803
- **Paths:** docs/development/full_body_models/HANDS_AUDIT.md,tests/unit/full_body_models/test_hands_audit_inventory.py
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`36a0289e`; collated from changes/13759-add-audited-per-engine-hand-capability-m.md)
- **Summary:** Add audited per-engine hand capability matrix and frame contract
- **Next step:** Review and merge the hand audit PR.

### DL-#13778 · Reconcile Neural Motion Matching and Simscape Documentation With Diagnostic Receipts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13778
- **Branch:** staff/issue-remediator-13778-ebd00a
- **PR:** #13788
- **Paths:** see #13788
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`4da07ebf`; collated from changes/13778-reconcile-neural-motion-matching-and-sim.md)
- **Summary:** Reconcile neural motion matching and Simscape documentation with diagnostic receipts
- **Next step:** Review the receipt-backed prose and merge PR

### DL-#13720 · Route Legacy Chat UI Builder Imports to the Pinned Tools Module With Shared Module Identity.

- **State:** in_review
- **Owner:** codex
- **Issue:** #13720
- **Branch:** codex/13720-chat-ui-retirement
- **PR:** #13724, #13747, #13810
- **Paths:** src/shared/python/ud_import_alias_policy.py,tests/unit/repo_hygiene/test_pinned_chat_ui_builder_alias.py
- **Started:** 2026-10-24
- **Last verified:** 2026-10-25 (`c68b4273`; collated from changes/13720-retire-the-remaining-local-chat-ui-build.md)
- **Summary:** Route legacy chat UI builder imports to the pinned Tools module with shared module identity.
- **Next step:** Review the focused provider-origin, export, parity, inventory, and canonical validation results before publishing this physical-retirement slice.

### DL-#13552 · Simscape Grip Kernel Follows the Shared Capsule Line-Contact Law; Ten_Finger Verifier Back to 1E-11; Structural Freshness Check for Generated JSON; Goldens Rebuilt From the Rounded Export Params; Committed R2025b Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13552
- **Branch:** claude/grip-13556-rounded-goldens
- **PR:** #13677
- **Paths:** see #13677
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`645410ff`; collated from changes/13552-simscape-line-contact-kernel.md)
- **Summary:** Simscape grip kernel follows the shared capsule line-contact law; ten_finger verifier back to 1e-11; structural freshness check for generated JSON; goldens rebuilt from the rounded export params; committed R2025b evidence
- **Next step:** Merge via queue. Any later change to the shared grip law or scenarios requires regenerating the kernel (scripts/export_simscape_grip_kernel.py) and rerunning R2025b (gs3dx_grip_contact_run); test_r2025b_receipt_hashes_match_committed_kernel_and_inputs fails until the receipts are refreshed.

### DL-#13346 · Fix(Motion_Matching): Reclassify Grip_Style as Physical Pick and Clean Sidecar

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13346
- **Branch:** fix/13346-grip-style-physical
- **PR:** #13370
- **Paths:** see #13370
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`60523ca6`; collated from changes/13346-fix-motion-matching-reclassify-grip-styl.md)
- **Summary:** fix(motion_matching): reclassify grip_style as physical pick and clean sidecar
- **Next step:** Merge the PR.

### DL-#13208 · Lock Per-Runner BuildKit Cache Regression Guards

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13208
- **Branch:** cursor/close-13208-docker-cache-doc-66dc
- **PR:** #13681
- **Paths:** see #13681
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`3b9cd5a6`; collated from changes/13208-lock-per-runner-buildkit-cache-regressio.md)
- **Summary:** Lock per-runner BuildKit cache regression guards
- **Next step:** Merge PR #13681

### DL-#13578 · Adopt Fast-Tier Pytest Changed-Module Selection for Unit-Test-Gate

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13578
- **Branch:** cursor/ci-fast-tier-devlog-archive-7ec5
- **PR:** #13669
- **Paths:** see #13669
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`670ce9da`; collated from changes/13578-adopt-fast-tier-pytest-changed-module-se.md)
- **Summary:** Adopt fast-tier pytest changed-module selection for unit-test-gate
- **Next step:** Merge PR

### DL-#13648 · Characterize the Existing KnowledgePack Search Contract Before Provider Synchronization.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13648
- **Branch:** codex/13648-knowledgepack-search
- **PR:** #13649, #13667
- **Paths:** tests/unit/ai/test_knowledge_pack_search.py
- **Started:** 2026-10-23
- **Last verified:** 2026-10-24 (`75a80849`; collated from changes/13648-extend-knowledgepack-search-characteriza.md)
- **Summary:** Characterize the existing KnowledgePack search contract before provider synchronization.
- **Next step:** Use these regression tests when synchronizing the provider-owned search implementation.

### DL-#12577 · Add Loop Null-Space Interpretation Help and Salvage Tracking

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12577
- **Branch:** cursor/nsx-45-help-doc-3c3f
- **PR:** #13607, #13692, #13690
- **Paths:** see #13607
- **Started:** 2026-10-23
- **Last verified:** 2026-10-24 (`86a4e007`; collated from changes/12577-nsx-45-user-help-and-interpretation-guid.md)
- **Summary:** Add loop null-space interpretation help and salvage tracking
- **Next step:** Merge PR #13607 after checks pass.

### DL-#13650 · Clear Deleted ThemeManager Main-Window Wrappers During Test Cleanup

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13650
- **Branch:** codex/13650-theme-manager-fixture
- **PR:** #13651, #13667
- **Paths:** see #13651
- **Started:** 2026-10-23
- **Last verified:** 2026-10-24 (`75a80849`; collated from changes/13650-isolate-capture-rig-theme-compliance-the.md)
- **Summary:** Clear deleted ThemeManager main-window wrappers during test cleanup
- **Next step:** Merge the Qt fixture cleanup fix.

### DL-#13641 · Defer Generated Collation Branch Updates When the Canonical Queue Preflight Cannot Verify Editability

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13641
- **Branch:** codex/13641-collator-update-preflight
- **PR:** #13643
- **Paths:** see #13643
- **Started:** 2026-10-23
- **Last verified:** 2026-10-23 (`2dc1348b`; collated from changes/13641-defer-generated-collation-branch-updates.md)
- **Summary:** Defer generated collation branch updates when the canonical queue preflight cannot verify editability
- **Next step:** Publish after canonical RM2335 merges; qualify natural consumer execution

### DL-#13636 · Route OpenSim Classification and Skip Reporting Through the Existing Reversible Public Hosted Policy; Fail Closed on Diff Errors While Preserving Real Engine Docker Validation.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13636
- **Branch:** codex/13636-opensim-lightweight-routing
- **PR:** #13637
- **Paths:** see #13637
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`a0fe9edc`; collated from changes/13636-route-opensim-classification-and-skip-re.md)
- **Summary:** Route OpenSim classification and skip reporting through the existing reversible public hosted policy; fail closed on diff errors while preserving real engine Docker validation.
- **Next step:** Qualify source and admit after the current nightly cohort; verify natural routing before runtime savings claims.

### DL-#11161 — Capture Registry, Capture Export and Swing Comparison (Owner Swing vs Tour Average)

- **Next step:** Coordinate the separately tracked engine deliveries in #11165–#11169 and cross-engine comparison #11170; keep this parent active until its scoped acceptance is met.
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** claude
- **Issue:** #11161
- **Branch:** `feat/capture-registry-11161`
- **Paths:** `data/capture_registry.json`, `src/motion_capture/capture_registry.py`, `src/motion_capture/capture_export.py`, `src/shared/python/swing_comparison/`, `src/shared/python/motion_matching/leaderboard.py`, `src/shared/python/motion_matching/pipeline/`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Capture-O parent coordinates a neutral capture registry/export and engine-independent swing comparison. Engine slices #11165 through #11169 and cross-engine comparison #11170 remain separately tracked.

### DL-#11421 · Dynamics-Informed Mocap Matching With ZTCF Prediction and Continuous Forward Replay

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #11421 (children #11422–#11430, #11434, #11435, #11437 closed; #11436 active)
- **PR:** #11436
- **Branch:** `feat/dime-11436-learned-initializers`
- **Paths:** `src/shared/python/estimation/dime_learned_initializers.py`, `src/shared/python/estimation/__init__.py`, `tests/unit/estimation/test_dime_learned_initializers.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Epic #11421: ZTCF drift prediction, kinematics-informed mocap matching, calibrated uncertainty weighting, uninterrupted forward simulation replay without frame-to-frame resetting.
- **Next step:** Create PR, verify CI Standard, squash-merge into origin/main.

### DL-#11313 · Necromatcher Video Export Force/Torque Layer

- **State:** in_review
- **Owner:** claude (Sonnet 5 CLI delegate)
- **Issue:** #11313; parent #11285
- **PR:** draft from `claude/issue-11313` (number on the PR; `Fixes #11313`)
- **Branch:** `claude/issue-11313`
- **Paths:** `src/shared/python/workspace/necromatcher_video_forces.py`, `src/shared/python/workspace/necromatcher_video.py`, `src/shared/python/workspace/necromatcher_video_worker.py`, `src/shared/python/workspace/necromatcher_video_jobs.py`, `src/api/routes/necromatcher.py`, `src/tools/necromatcher/video_dialog.py`, `ui/src/components/necromatcher/VideoExportControls.tsx`, `tests/unit/workspace/test_necromatcher_video_forces.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Off-by-default `force_layer {enabled, kinds, scale, segment_shading}` drawn through the shared FTO-8 renderer with the fit's camera; analytic v/a from the preserved spline (kinematic-only fits report the layer unavailable); legend states research-fit, not measured forces; manifest and per-frame receipts are deterministic.
- **Next step:** Frontier review of the draft PR, then rebase against paused draft #11359 (also edits `necromatcher_video.py`) before merge.

### DL-#11278 · COV-10 Model-Candidate Video Comparison

- **State:** in_progress
- **Owner:** claude
- **Issue:** #11278; parent #11268; blocked (real data) on #11165
- **PR:** not created (Refs #11278)
- **Branch:** `claude/issue-11278`
- **Paths:** `src/motion_capture/reference/model_candidate_source.py`, `tests/unit/motion_capture/test_cov10_model_candidate_source.py`, `docs/development/capture-o-video/procedure.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Python-only engine-generic path projecting any model candidate into video views and comparing it through the COV-7 L2 receipt; real Simscape evidence needs a stored `capture-O` candidate from #11165 and an R2025b host.
- **Next step:** After #11165 stores a candidate, run `compare_candidate_2d` on an R2025b host for every paired swing and record aggregate results on #11278.

### DL-#11268 · Capture-O Video Companion

- **State:** in_progress
- **Owner:** local
- **Issue:** #11268; children #11269–#11279; parent program #11161
- **PR:** #11502 (Closes #11271, Refs #11268)
- **Branch:** `feat/cov-3-protocol-decision-11271`
- **Paths:** `docs/development/capture-o-video/comparison-protocol.md`, `src/config/cov_comparison_profile.v1.json`, `src/config/cov_landmark_correspondence.json`, `tests/unit/motion_capture/test_cov3_protocol_and_correspondence.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Ratify comparison protocol L0-L3, frozen comparison profile, and landmark correspondence table.
- **Next step:** Merge PR for COV-3, complete Epic #11268.

- **PR:** #11499 (Closes #11272, #11274, #11275, #11276, #11277, #11279, Refs #11268)
- **Branch:** `feat/cov-benchmark-suite-11268`
- **Paths:** `src/motion_capture/reference/**`, `docs/development/capture-o-video/**`, `tests/unit/motion_capture/test_cov*`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Complete implementation and verification of Capture-O Video benchmark suite spanning levels L0–L3, virtual camera projection, pairing confidence, marker-anchored subject scaling, 2D/3D residuals, and error budget reporting.
- **Next step:** Merge consolidated PR, close superseded individual draft PRs, and close Epic #11268.

- **PR:** #11416 (Closes #11270, Refs #11268)
- **Branch:** `feat/cov-2-register-sources-11270`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-3 comparison protocol specification and level definitions (#11271).

- **PR:** #11413 (Closes #11269, Refs #11268)
- **Branch:** `feat/cov-1-acquisition-11269`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-2 (#11270) video source registration and swing window grading.

- **PR:** #11417 (Closes #11273, Refs #11268)
- **Branch:** `feat/cov-5-runner-matrix-11273`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-6 validation harness and error budget report (#11274).

### DL-#11095 — Qualify OpenSim Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11095
- **Branch:** `feat/mmr-10o-opensim-dual-club-11095`
- **Paths:** `src/engines/physics_engines/opensim/python/native_qualification.py`, `tests/unit/engines/opensim/test_opensim_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/opensim/`
- **Last verified:** 2026-09-29 at d5dd606ffa6dafc11408614676a2576764a5e092 — 17 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native OpenSim qualification NOT achieved: opensim bindings unavailable on all reachable hosts.
- **Summary:** OpenSim receipts now fail closed on missing runtime/evidence; fabricated marker metrics, unenforced tolerances and placeholder hashes were removed. Native qualification remains blocked without the pinned bindings.

### DL-#11096 — Qualify MyoSuite Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11096
- **Branch:** `feat/mmr-10m-myosuite-dual-club-11096`
- **Paths:** `src/engines/physics_engines/myosuite/python/native_qualification.py`, `tests/unit/engines/myosuite/test_myosuite_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/myosuite/`
- **Last verified:** 2026-09-29 at 613b0e28ed88b5af4719ecccfe876d5d9db90419 — 19 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native MyoSuite qualification NOT achieved: myosuite/MuJoCo unavailable on all reachable hosts.
- **Summary:** MyoSuite receipts fail closed on unavailable runtime and missing/invalid rollout, marker, derivative or quaternion evidence; fabricated metrics and unenforced tolerances were removed. Native qualification remains unavailable.

### DL-#11094 — Qualify Drake Native Dual-Club Dynamics and Replay

- **Started:** not recorded in source
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** UDFixTrio10x
- **Issue:** #11094
- **Branch:** `feat/mmr-10d-drake-dual-club-11094`
- **Paths:** `src/engines/physics_engines/drake/python/native_qualification.py`, `tests/unit/engines/drake/test_drake_dual_club_qualification.py`, `scripts/ci/run_native_engine_lane.py`, `scripts/ci/run_native_engine_lane.sh`, `tests/scripts/test_run_native_engine_lane.py`, `docs/development/matched_swing_program/evidence/nightly/drake_receipt.json`, `docs/development/matched_swing_program/evidence/drake/`
- **Last verified:** 2026-09-29 at 3b23bd41502a7b7d028f291564997b6460f14cbc — 36 focused tests passed (test_drake_dual_club_qualification.py 16, test_run_native_engine_lane.py + test_native_lane_freshness.py 20); red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native Drake qualification NOT achieved: pydrake unavailable on all reachable hosts.
- **Summary:** Drake receipts now fail closed on missing runtime/evidence; fabricated marker metrics and placeholder hashes were removed, and Drake entered the nightly lane. Native qualification remains blocked without pydrake.
- **Next step:** Merge drivers follow; native qualification remains blocked on engine availability (recorded in receipts, not silently).

### DL-#10286 · Native Swing ZTCF/ZVCF Fail-Closed Dynamics and DTACK Semantics

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #10286
- **PR:** draft PR from `agy/ud-10286-fail-closed`
- **Branch:** `agy/ud-10286-fail-closed`
- **Paths:** `src/engines/physics_engines/myosuite/python/_drift_control.py`, `src/engines/physics_engines/pendulum/python/golf_swing_physics_engine.py`, `src/engines/physics_engines/pinocchio/python/dtack/gui/main_window.py`, `src/engines/physics_engines/pinocchio/python/dtack/sim/dynamics.py`, `tests/engines/physics_engines/test_golf_swing_pendulum.py`, `tests/engines/physics_engines/test_myosuite_engine.py`, `tests/unit/engines/pinocchio/dtack/sim/test_dynamics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Remove success-shaped empty/zero array returns from uninitialized ZTCF/ZVCF counterfactual methods (raising StateError and ValueError for invalid dimensions), update dtack compute_zvcf to canonical (v=0, tau=0) semantics, add compute_zero_velocity_controlled for control-preserved dynamics, and update GUI caller and button label to "Zero-velocity (control kept)".
- **Next step:** CI green, review, merge.

### DL-#9619 · Amend ADR-0041 to Record Consumer-Side Fitter Decision

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local (session `claude-deskcomputer-20260925-ud`)
- **Issue:** #9619 (companion #9630)
- **Branch:** `agy/ud-9619-adr0041`
- **Paths:** `docs/adr/0041-markerless-mocap-consumer-authority.md`, `tests/architecture/test_markerless_mocap_authority.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Amended ADR-0041 to record that consumer-side self-calibration fitters live in UpstreamDrift (`src/motion_capture/reconstruct/`) while Tools maintains vendor-neutral reference geometry (#9630, #9619).
- **Next step:** CI green, review, merge.

### DL-#9688 · Bunker P1: Decide and Benchmark a Genuine 3D Sand-Motion Tier

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9688
- **PR:** draft PR from `agy/ud-9688-capability-register`
- **Branch:** `agy/ud-9688-capability-register`
- **Paths:** `src/bunkershot3d/solvers/capability.py`, `src/bunkershot3d/solvers/exceptions.py`, `src/bunkershot3d/solvers/__init__.py`, `docs/adr/0044-out-of-plane-fidelity-for-bunkershot3d.md`, `tests/bunkershot3d/solvers/test_capability_9688.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added fail-closed sand-motion capability register (SandMotionKind, SandMotionCapability, register mapping, and require\_\* guards) in bunkershot3d.solvers to prevent presenting F0/F1/proxy/tracer outputs as genuine 3-D individual-grain trajectories or spherical ball spin.
- **Next step:** CI green, review, merge.

### DL-#9613 · Rig Soak: Multi-Camera Repeat Record Mode

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9613
- **PR:** draft PR from `agy/ud-9613-record-repeat`
- **Branch:** `agy/ud-9613-record-repeat`
- **Paths:** `src/motion_capture/rig/__main__.py`, `src/motion_capture/rig/soak.py`, `tests/motion_capture/rig/test_record_repeat.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added `record --repeat N [--pause S]` soak mode to camera rig CLI, supporting N back-to-back takes in take subdirectories, machine-readable soak summary, and worst-take exit code.
- **Next step:** CI green, review, merge.

### DL-#9411 · Keep MyPy Exclusion Budget and Unit-Gate Quarantine Ratchets Green

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #9411
- **PR:** draft PR from `claude/ud-9411-unquarantine-passing`
- **Branch:** `claude/ud-9411-unquarantine-passing`
- **Paths:** `scripts/config/unit_gate_quarantine.json`, quarantined unit test files, `src/shared/python/validation_pkg/data_fitting.py`, `src/tools/model_explorer/mujoco_viewer.py`, `src/shared/python/ai/adapters/`, `src/api/routes/__init__.py`, `src/shared/python/physics/impact_model/utils.py`, `src/shared/python/pendulum_simulator/physics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 — #10997 Linux run: kernel built and imported, none of the 12 retired BunkerShot IDs failed; the tests the kernel un-skipped exposed #11000 (fixed in #11004) and #11003 (fixed in #11010), and an untracked root `Cargo.lock` (fixed); replayed onto main d716c698f, ledger contract passes at 55 IDs.
- **Summary:** Retire unit-gate quarantine IDs whose tests pass on the Linux CI lane after fixing real defects (shoulder FK origin), deduplicating the `data_fitting` and `mujoco_viewer` coordinators onto their existing helper modules, and retargeting stale tests. Tools-owned `ai` adapter changes and the gear-effect sign question are out of scope here (Tools#5362 / a physics-convention decision). Earlier slice (#10973) landed the mypy budget and 43 retirements. Rust slice (#10997): unit-test-gate maturin-builds upstream-physics into its venv (ball_simulator enforces strict Rust parity), retiring the 12 BunkerShot workbench/GUI IDs. Consolidated delivered subscopes from the preserved source rows: Claude #10995 (claude/ud-9411-vendor-bump-safe-eval) advanced the Tools pin and retired 10 quarantined tests; Antigravity #10999 (feat/ud-9411-quarantine-burndown-batch1) retired 6; #11005 (feat/ud-9411-ai-adapters-convergence) converged Gemini/BitNet adapters and retired 5; #11013 (feat/ud-9411-quarantine-batch2) retired 1 and narrowed optional-dependency mock scope. The source snapshot retains original owners, branches, paths, verification text and next steps. Broader parent mypy/quarantine readiness remains in review; no full ratchet completion is claimed.
- **Next step:** Complete parent-level mypy-exclusion and quarantine acceptance against a source-bound Linux unit-gate run, then reconcile the remaining eligible IDs. The delivered child tranches do not establish full parent readiness; the original child next steps are preserved in the source snapshot.

### DL-#9703 · Versioned Pre-Impact Bundle for Tools Impact Kernels

- **State:** in_review
- **Owner:** claude
- **Issue:** #9703 (IA-U2; parent #9700)
- **PR:** draft (see branch)
- **Branch:** `feat/9703-pre-impact-bundle`
- **Paths:** `src/shared/python/physics/pre_impact_bundle.py`, `src/shared/python/physics/_pre_impact_contracts.py`, `src/shared/python/physics/_pre_impact_frames.py`, `tests/shared_contracts/test_pre_impact_bundle.py`, `docs/development/impact_acoustics_program.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — 48/48 tests pass against vendor pin a9ed0e7c5 (incl. python -O); pre-push mypy and unit hooks pass (Rust wheel hidden, #10946)
- **Summary:** Consumer-side PreImpactBundle v1 composing Tools conventions: provenance, bounded timebase, world/head/grip poses, head COM inertia and contact, ball, reduced shaft modal state and prestress field, per-hand wrench/impedance, per-field origin; absent fields raise; power-invariant transforms; energy-reporting modal projection. Engine adapters and installed-wheel fixtures remain.
- **Next step:** Review the draft PR, then implement the first engine adapter against the bundle under #9703.

### DL-#10342 · OpenSim/MyoSuite Native Nightly Lane Receipts

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local
- **Issue:** #10342 (MS-43, epic #10363)
- **Branch:** fix/issue-10342-ms-43-native-lane-local
- **Paths:** scripts/ci/run_native_engine_lane.py; scripts/ci/run_native_engine_lane.sh; docs/development/matched_swing_program/evidence/nightly/; tests/docs/test_native_lane_freshness.py; tests/scripts/test_run_native_engine_lane.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added idempotent native-engine lane runner emitting hashed nightly receipts for OpenSim and MyoSuite (`requires_opensim` / `requires_myosuite` pytest markers), freshness gate warning at seven days and failing at thirty days, and ControlTower verification docs without editing `.github/workflows`.
- **Next step:** Refresh receipts on ControlTower opensim-10003 venv; merge PR closing #10342.
- **Evidence:** docs/development/matched_swing_program/evidence/nightly/opensim_receipt.json; docs/development/matched_swing_program/evidence/nightly/myosuite_receipt.json; tests/docs/test_native_lane_freshness.py.

### DL-#9162 · Local Branch Triage

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9162 (rollout Repository_Management#1460; no-seed bootstrap #9161)
- **Branch:** conductor/issue-9162
- **Paths:** docs/development/branch_triage_9162.md
- **Started:** 2026-09-15
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Disposition ledger for every local branch in the primary checkout: 3 protected, 27 deferred to the worktree pass, 34 live (merged sweep then DL entry), 178 stale and 144 abandoned scratch branches (bundle snapshot then delete). The runbook in the ledger executes the deletions and the coordination notice for agent-owned branches.
- **Next step:** Execute the runbook in docs/development/branch_triage_9162.md from the primary checkout and fill in its Outcome section.

### DL-#9410 · Adversarial Product Review Remediation Epic

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9410 (program Repository_Management#1505; children #8820–#8943, #8360, #8641, #8843, #8846, #8853, #8861–#8870, #8874–#8876, #8894)
- **Branch:** conductor/issue-9410
- **Paths:** docs/development/adversarial_review_remediation_9410.md
- **Started:** 2026-09-16
- **Last verified:** 2026-09-16 (SELF; every child re-checked against `db4fe88c4` by first-parent history search plus source reads of each residual)
- **Summary:** Reconciliation ledger for the 2026-08-21 adversarial review: 34 of 61 children landed on `main` (SHAs recorded), 27 residual. Three residual fixes already exist on unmerged branches (`readiness/p0-9412-one-tile-registry`, `conductor/issue-8865`, `conductor/issue-8866`). Epic acceptance (one registry, one API factory, one C3D reader, one pose type, no GUI-thread simulation) is not met on `main`.
- **Next step:** Rebase and merge `readiness/p0-9412-one-tile-registry`, then re-verify the Cluster B rows in the ledger.

### DL-#10602 · Club-Only Motion Matching Plan

- **State:** in_review
- **Owner:** codex
- **Issue:** #10602
- **Branch:** `agy/ud-10602-real-matrix`
- **PR:** draft PR from `agy/ud-10602-real-matrix`
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/; src/shared/python/motion_matching/club_only/
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Published bounded implementation issues with TDD/DbC/LoD/DRY prompts, dependency ordering, native validation gates and shared technical review. Planning artifacts do not qualify physical results or speedup.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/plans/club_neural_review/REVIEW.md; docs/plans/club_neural_review/excel_audit.json.

### DL-#10363 · Matched Swing Continuation Review & Drake G1 Retraction

- **State:** in_review
- **Owner:** claude-deskcomputer-20260925-ud
- **Issue:** #10363
- **Branch:** agy/ud-10363-drake-retraction
- **PR:** draft PR from `agy/ud-10363-drake-retraction`
- **Paths:** docs/development/matched_swing_program; src/shared/python/motion_matching/acceptance.py; src/engines/physics_engines/drake/python/full_body_fit.py; evidence/matched/driver_g1_drake/reevaluation.json; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; reports/matched_swing_ledger.json
- **Started:** 2026-09-18
- **Last verified:** 2026-09-25 at ff9fbee62 (retracted Drake G1 fabricated PASS via reevaluation.json; closed synthesis paths in full_body_fit.py; added fail-closed integrity gates in acceptance.py; 0 PASSED rows in ledger; motion_matching/tour_baselines/opensim ladder/browser suites green except 13 pre-existing local failures)
- **Summary:** 2026-09-18 continuation review (codex) preserved checkpoint/source recovery evidence and bounded the cheaper-agent continuation, with no physical acceptance claimed. Then: retracted the fabricated Drake G1 receipt (PR #10506) under MS-100; added fail-closed evidence integrity gates (\_evaluate_evidence_integrity) rejecting placeholder hashes and zero residuals without replay evidence; removed physical_audit numeric literals, np.zeros target fallback, and silent warm-start fallback in Drake full_body_fit.py; regenerated ledger and status matrices confirming 0 PASSED rows.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/development/matched_swing_program/evidence/continuation_20260918/matching-handoff-snapshot.json; evidence/matched/driver_g1_drake/reevaluation.json; reports/matched_swing_ledger.json; tests/unit/motion_matching/test_acceptance.py; tests/unit/motion_matching/test_drake_full_body_fit.py.

### DL-#10362 · Tools Dependency Gate for Matched Swing Program

- **State:** in_progress
- **Owner:** local
- **Issue:** #10362 (epic #10363)
- **Branch:** feat/ms95-tools-dependency-gate-10362
- **PR:** open
- **Paths:** vendor/ud-tools; Cargo.toml; requirements-tools.txt; docs/shared_tools/divergence_inventory.md; docs/shared_tools/divergence_inventory.v1.json; docs/agent_context/README.md; docs/agent_context/index.html
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 at 62e8cdbf9 (SELF; Tools main green, Tools #4494 and #4262 closed, Tools #5227 landed; four-way pin bumped to 62e8cdbf9; divergence inventory regenerated; agent context verified; test_no_shadow_of_tools_shared and run_checks pass)
- **Summary:** UpstreamDrift ownership of humanoid_character_builder and model_generation ruled per Tools #4494; Tools #4262 and #4494 closed; vendor/ud-tools, requirements-tools.txt, Cargo.toml, and divergence inventory repinned to Tools main 62e8cdbf9.
- **Next step:** Open PR referencing Closes #10362, enable auto-merge, release lease.
- **Evidence:** docs/shared_tools/divergence_inventory.v1.json; tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py; tests/fixtures/reference_calibration/run_checks.py.

### DL-#9967 · Native Simscape Tour Matching

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** codex (turnover review; execution ownership by next lease)
- **Issue:** #9967 (parent #9921)
- **Branch:** feat/9967-native-simscape-pinocchio
- **Paths:** src/shared/python/motion_matching; docs/development/simscape_tour_matching
- **Started:** 2026-09-09
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Run101 improves yaw and has measured R2025b–Pinocchio prefix agreement of 0.0605 mm maximum. Terminal RMS 40.31 mm fails the 35 mm gate; full 1.814 s capture is incomplete. No optimizer launched by this review.
- **Next step:** Follow COMPLETION_HANDOFF_20260916.md: restore clean-runtime native providers, coordinate #10260 finite-weld derivative qualification, produce articulated feasibility evidence, fit0.90 s and deliver repeatable reports. Run102 remains rejected; no new compute launched by this review.

## Shipped (Last 90 Days)

### DL-#14049 · CAP-9 (#14038): Pinocchio Capability Adapter at the Hand-Club Interfaces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14049
- **Branch:** feat/14049-pinocchio-capability-adapter
- **PR:** #14076
- **Paths:** see #14076
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`e0ede838`; collated from changes/14049-pinocchio-capability-adapter.md)
- **Summary:** Pinocchio capability adapter emitting acceleration and force capacity ellipsoids per grip point
- **Next step:** Shipped in PR #14076.

### DL-#13938 · Evaluate Glass Conductivity From Temperature Only; Drop Power-Density Heating in Default Model.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13938
- **Branch:** merged via #14065
- **PR:** #14065
- **Paths:** see #14065
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`0bbbe4e3`; collated from changes/13938-evaluate-glass-conductivity-from-tempera.md)
- **Summary:** Evaluate glass conductivity from temperature only; drop power-density heating in default model.
- **Next step:** Shipped in PR #14065.

### DL-#14043 · CAP-4 (#14038): MuJoCo Capability Adapter at the Hand-Club Interfaces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14043
- **Branch:** feat/14043-mujoco-capability-adapter
- **PR:** #14071
- **Paths:** see #14071
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`a6bad706`; collated from changes/14043-mujoco-capability-adapter.md)
- **Summary:** MuJoCo capability adapter emitting acceleration and force capacity ellipsoids per grip point
- **Next step:** Shipped in PR #14071.

### DL-#14047 · CAP-7 (#14038): Dynamically Consistent Force-Capacity Core

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14047
- **Branch:** feat/14047-force-capacity-core
- **PR:** #14067
- **Paths:** see #14067
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`89f99a0a`; collated from changes/14047-cap-7-dynamically-consistent-force-capac.md)
- **Summary:** CAP-7: Dynamically consistent force-capacity core
- **Next step:** Shipped in PR #14067.

### DL-#14046 · CAP-6 (#14038): Exporter Option --Interface-Ellipsoid With a --Hand-Mobility Alias

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14046
- **Branch:** feat/14046-interface-ellipsoid-options
- **PR:** #14064
- **Paths:** see #14064
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`05491337`; collated from changes/14046-interface-ellipsoid-options.md)
- **Summary:** CAP-6: Exporter option --interface-ellipsoid with a --hand-mobility alias
- **Next step:** Shipped in PR #14064.

### DL-#14039 · Add Interface-Ellipsoid Kind/Provenance Contracts and Engine Parity Ledger (CAP-0)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14039
- **Branch:** merged via #14055
- **PR:** #14055
- **Paths:** see #14055
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`494cbba4`; collated from changes/14039-add-interface-ellipsoid-kind-provenance.md)
- **Summary:** Add interface-ellipsoid kind/provenance contracts and engine parity ledger (CAP-0)
- **Next step:** Shipped in PR #14055.

### DL-#14027 · Drake GUI No Longer Crashes After MeshCat Starts and Opens the V2 Meshed Golfer With a Drake-vs-MuJoCo Parity Check

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14027
- **Branch:** merged via #14028
- **PR:** #14028
- **Paths:** see #14028
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`1a7cee00`; collated from changes/14027-drake-gui-no-longer-crashes-after-meshca.md)
- **Summary:** Drake GUI no longer crashes after MeshCat starts and opens the v2 meshed golfer with a Drake-vs-MuJoCo parity check
- **Next step:** Shipped in PR #14028.

### DL-#14003 · EllipsoidGlyph + MuJoCo Ellipsoid Drawing for Native_Viewer_Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14003
- **Branch:** merged via #14022
- **PR:** #14022
- **Paths:** see #14022
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`30684dd5`; collated from changes/14003-ellipsoidglyph-mujoco-ellipsoid-drawing.md)
- **Summary:** EllipsoidGlyph + MuJoCo ellipsoid drawing for native_viewer_export
- **Next step:** Shipped in PR #14022.

### DL-#13779 · Fix(Motion-Matching): L-BFGS-B Stop Reason Is Fail-Closed (ABNORMAL Exits Are Error, Not Max_Iterations) and a Non-Finite Trial No Longer Ends the Run or Freezes Best-Iterate Tracking

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13779
- **Branch:** merged via #14013
- **PR:** #14013
- **Paths:** see #14013
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`939ec1d6`; collated from changes/13779-fix-motion-matching-l-bfgs-b-stop-reason.md)
- **Summary:** fix(motion-matching): L-BFGS-B stop reason is fail-closed (ABNORMAL exits are error, not max_iterations) and a non-finite trial no longer ends the run or freezes best-iterate tracking
- **Next step:** Shipped in PR #14013.

### DL-#13795 · Correct Cross_Engine_V1 Dependency Status: The CC-7/9/10 Draft PRs Were Closed Unmerged, No Comparison Has Been Run

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13795
- **Branch:** merged via #14017
- **PR:** #14017, #14018
- **Paths:** see #14017
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`28f0ee65`; collated from changes/13795-add-the-cross-engine-perturbation-receip.md)
- **Summary:** Correct cross_engine_v1 dependency status: the CC-7/9/10 draft PRs were closed unmerged, no comparison has been run
- **Next step:** Shipped in PR #14017.

### DL-#12684 · R4 Cross-Engine Address Knee Flexion Check Over the Committed Receipts (Scripts/address_knee_engine_check.py): Per-Engine Disagreement Over 0.1 Deg, Unavailable Engines With Reasons, Fail-Closed on Malformed Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12684
- **Branch:** merged via #14014
- **PR:** #14014
- **Paths:** see #14014
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`8aee5888`; collated from changes/12684-r4-cross-engine-address-knee-flexion-che.md)
- **Summary:** R4 cross-engine address knee flexion check over the committed receipts (scripts/address_knee_engine_check.py): per-engine disagreement over 0.1 deg, unavailable engines with reasons, fail-closed on malformed receipts
- **Next step:** Shipped in PR #14014.

### DL-#14040 · CAP-1: Acceleration_Capacity Ellipsoid and Drift_Acceleration in Operational_Space

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14040
- **Branch:** merged via #14056
- **PR:** #14056
- **Paths:** see #14056
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`bc8dad19`; collated from changes/14040-cap-1-acceleration-capacity-ellipsoid-an.md)
- **Summary:** CAP-1: acceleration_capacity ellipsoid and drift_acceleration in operational_space
- **Next step:** Shipped in PR #14056.

### DL-#14041 · CAP-2: Torque-Limit Provider Adapting CapacityVector to (B, Tau_Lo, Tau_Hi) With Provenance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14041
- **Branch:** merged via #14054
- **PR:** #14054
- **Paths:** see #14054
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`cfef12fb`; collated from changes/14041-cap-2-torque-limit-provider-adapting-cap.md)
- **Summary:** CAP-2: torque-limit provider adapting CapacityVector to (B, tau_lo, tau_hi) with provenance
- **Next step:** Shipped in PR #14054.

### DL-#13785 · ML-WS-8: ProposalInitializer Protocol and --Neural-Mode Preview Warm-Start Hook Into MJX Knot Optimiser

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13785
- **Branch:** merged via #14034
- **PR:** #14034
- **Paths:** see #14034
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`a0ee91aa`; collated from changes/13785-ml-ws-8-neural-mode-preview-warm-start.md)
- **Summary:** ML-WS-8: ProposalInitializer protocol and --neural-mode preview warm-start hook into MJX knot optimiser
- **Next step:** Shipped in PR #14034.

### DL-#14031 · Bump Tornado 6.5.10 to 6.6 for CVE-2026-116577, CVE-2026-115927 and CVE-2026-116578, and Msgpack 1.2.1 to 1.2.3 in the Docker Images for CVE-2026-114613 (No Audit Waiver)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14031
- **Branch:** merged via #14032
- **PR:** #14032
- **Paths:** see #14032
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`292f8195`; collated from changes/14031-bump-tornado-6-5-10-to-6-6-for-cve-2026.md)
- **Summary:** Bump tornado 6.5.10 to 6.6 for CVE-2026-116577, CVE-2026-115927 and CVE-2026-116578, and msgpack 1.2.1 to 1.2.3 in the Docker images for CVE-2026-114613 (no audit waiver)
- **Next step:** Shipped in PR #14032.

### DL-#14010 · Catch TypeError and Probe Availability in Cross-Engine Builder and Degrade Uninitialized Engines to Stub

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14010
- **Branch:** merged via #14029
- **PR:** #14029
- **Paths:** see #14029
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`891d03fb`; collated from changes/14010-cross-engine-probe-unusable-engines.md)
- **Summary:** Catch TypeError and probe availability in cross-engine builder and degrade uninitialized engines to stub
- **Next step:** Shipped in PR #14029.

### DL-#14004 · MuJoCo Adapter for Per-Hand Constrained Mobility W = J A J^T and Effective Masses at Each Grip Point (HCM-3A)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14004
- **Branch:** merged via #14026
- **PR:** #14026
- **Paths:** see #14026
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`5dc17c7c`; collated from changes/14004-mujoco-adapter-for-per-hand-constrained.md)
- **Summary:** MuJoCo adapter for per-hand constrained mobility W = J A J^T and effective masses at each grip point (HCM-3a)
- **Next step:** Shipped in PR #14026.

### DL-#14002 · Add Operational-Space Mobility Matrix, Velocity/Force Ellipsoids and Spatial Transport

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14002
- **Branch:** merged via #14020
- **PR:** #14020
- **Paths:** see #14020
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`42ea10f3`; collated from changes/14002-add-operational-space-mobility-matrix-ve.md)
- **Summary:** Add operational-space mobility matrix, velocity/force ellipsoids and spatial transport
- **Next step:** Shipped in PR #14020.

### DL-#13995 · Thread One ResolvedHandSpec (Grip Style, Strength, Handedness, Hand Length) Through Every Engine Emitter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13995
- **Branch:** merged via #14021
- **PR:** #14021
- **Paths:** see #14021
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`78189f73`; collated from changes/13995-thread-one-resolvedhandspec-grip-style-s.md)
- **Summary:** Thread one ResolvedHandSpec (grip style, strength, handedness, hand length) through every engine emitter
- **Next step:** Shipped in PR #14021.

### DL-#13994 · Hands Close-Up Camera Tracks the Kinematic Grip Midpoint From the Club Pose, With the Overlay on or Off

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13994
- **Branch:** merged via #14001
- **PR:** #14001
- **Paths:** see #14001
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`77a17e4b`; collated from changes/13994-hands-close-up-camera-tracks-the-kinemat.md)
- **Summary:** Hands close-up camera tracks the kinematic grip midpoint from the club pose, with the overlay on or off
- **Next step:** Shipped in PR #14001.

### DL-#13993 · Taper the Shared Visual Forearm 20 Percent Toward the Wrist (Appearance Only; Inertias Unchanged)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13993
- **Branch:** merged via #14008
- **PR:** #14008
- **Paths:** see #14008
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`831b01a9`; collated from changes/13993-taper-the-shared-visual-forearm-20-perce.md)
- **Summary:** Taper the shared visual forearm 20 percent toward the wrist (appearance only; inertias unchanged)
- **Next step:** Shipped in PR #14008.

### DL-#13992 · Graft Pelvis_Frame_From_Spec Accepts HIP_MIRROR-Conjugated Hips

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13992
- **Branch:** merged via #13999
- **PR:** #13999
- **Paths:** see #13999
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`05ba1f99`; collated from changes/13992-graft-pelvis-frame-from-spec-accepts-hip.md)
- **Summary:** Graft pelvis_frame_from_spec accepts HIP_MIRROR-conjugated hips
- **Next step:** Shipped in PR #13999.

### DL-#13793 · Add Sensitivity Fields (FTLE, Sigma-Max, Agreement Horizon) to Pendulum Parity Receipts (SAP-4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13793
- **Branch:** merged via #13988
- **PR:** #13988
- **Paths:** see #13988
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`fd233d9c`; collated from changes/13793-sensitivity-receipts.md)
- **Summary:** Add sensitivity fields (FTLE, sigma-max, agreement horizon) to pendulum parity receipts (SAP-4)
- **Next step:** Shipped in PR #13988.

### DL-#13945 · Pin Pytest <9.2.0 Across Repository Dependencies and CI Workflows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13945
- **Branch:** merged via #13945
- **PR:** #13945
- **Paths:** see #13945
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`9aba9431`; collated from changes/13945-pin-pytest-below-9-2-0.md)
- **Summary:** Pin pytest <9.2.0 across repository dependencies and CI workflows
- **Next step:** Shipped in PR #13945.

### DL-#13854 · Renew Hash-Frozen Research Budget Exceptions to 2027-01-31 Under #13853.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13854
- **Branch:** merged via #13862
- **PR:** #13862
- **Paths:** see #13862
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`441d4390`; collated from changes/13854-renew-hash-frozen-research-budget-except.md)
- **Summary:** Renew hash-frozen research budget exceptions to 2027-01-31 under #13853.
- **Next step:** Shipped in PR #13862.

### DL-#13891 · Add GripGeometry Source of Truth With Presets and USGA Rules Validator

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13891
- **Branch:** merged via #13913
- **PR:** #13913
- **Paths:** see #13913
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`51ab07b9`; collated from changes/13891-add-gripgeometry-source-of-truth-with-pr.md)
- **Summary:** Add GripGeometry source of truth with presets and USGA rules validator
- **Next step:** Shipped in PR #13913.

### DL-#13881 · Align Tendon Compliance and Active Force-Length Width With Thelen 2003

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13881
- **Branch:** merged via #13910
- **PR:** #13910
- **Paths:** see #13910
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`1a917a95`; collated from changes/13881-align-tendon-compliance-and-active-force.md)
- **Summary:** align tendon compliance and active force-length width with Thelen 2003
- **Next step:** Shipped in PR #13910.

### DL-#13890 · Implement Cited GripFrictionBand and Combined Squeeze Vector Bound

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13890
- **Branch:** merged via #13911
- **PR:** #13911
- **Paths:** see #13911
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`61d679a6`; collated from changes/13890-implement-cited-gripfrictionband-and-com.md)
- **Summary:** implement cited GripFrictionBand and combined squeeze vector bound
- **Next step:** Shipped in PR #13911.

### DL-#13775 · Regenerate the Proximal-Distal Release Claim Review and Manifests After the de Leva Provenance Change.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13775
- **Branch:** merged via #13839
- **PR:** #13839
- **Paths:** see #13839
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`0963aa38`; collated from changes/13775-regenerate-the-proximal-distal-release-c.md)
- **Summary:** Regenerate the proximal-distal release claim review and manifests after the de Leva provenance change.
- **Next step:** Shipped in PR #13839.

### DL-#13792 · Per-Engine Step-Size Convergence Study and Exact Torque Switch-Time Handling

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13792
- **Branch:** merged via #13855
- **PR:** #13855
- **Paths:** see #13855
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`1a76ba08`; collated from changes/13792-step-convergence-study-and-switch-time.md)
- **Summary:** Per-engine step-size convergence study and exact torque switch-time handling
- **Next step:** Shipped in PR #13855.

### DL-#13899 · Render Workers Put the Interpreter's Base_Prefix Lib Dir First on LD_LIBRARY_PATH, Because Opensim's Linux __Init__ Overwrites It With PATH; This Fixes the Requires_Opensim _Ctypes ABI Failure

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13899
- **Branch:** merged via #13900
- **PR:** #13900
- **Paths:** see #13900
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`ce1dd79e`; collated from changes/13899-render-worker-ld-library-path.md)
- **Summary:** Render workers put the interpreter's base_prefix lib dir first on LD_LIBRARY_PATH, because opensim's Linux __init__ overwrites it with PATH; this fixes the requires_opensim _ctypes ABI failure
- **Next step:** Shipped in PR #13900.

### DL-#13896 · Lift Render Worker PYTHONPATH Uses Shared Worker_Env() (Repo Roots Only), Fixing Requires_Opensim _Ctypes ABI Failure

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13896
- **Branch:** merged via #13897
- **PR:** #13897
- **Paths:** see #13897
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`ed08a3f2`; collated from changes/13896-lift-render-worker-pythonpath-uses-share.md)
- **Summary:** Lift render worker PYTHONPATH uses shared worker_env() (repo roots only), fixing requires_opensim _ctypes ABI failure
- **Next step:** Shipped in PR #13897.

### DL-#13852 · Add WSCG Masked-JSON Parity Test Scaffold for Budget-Split PRs.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13852
- **Branch:** merged via #13860
- **PR:** #13860, #13937, #13942, #13989
- **Paths:** see #13860
- **Started:** 2026-10-26
- **Last verified:** 2026-10-28 (`fed67279`; collated from changes/13852-wscg-case-split-3.md)
- **Summary:** Add WSCG masked-JSON parity test scaffold for budget-split PRs.
- **Next step:** Shipped in PR #13860.

### DL-#13851 · Remove Stale Urdf_Exporter _Add_Revolute_Joint Parameters Budget Exception.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13851
- **Branch:** merged via #13856
- **PR:** #13856
- **Paths:** see #13856
- **Started:** 2026-10-26
- **Last verified:** 2026-10-26 (`a6699390`; collated from changes/13851-remove-stale-urdf-exporter-add-revolute.md)
- **Summary:** Remove stale urdf_exporter _add_revolute_joint parameters budget exception.
- **Next step:** Shipped in PR #13856.

### DL-#13760 · ANSUR II Stature-Tied Hand Length and Breadth (HRG-HX3).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13760
- **Branch:** merged via #13840
- **PR:** #13840
- **Paths:** see #13840
- **Started:** 2026-10-26
- **Last verified:** 2026-10-26 (`eed5077d`; collated from changes/13760-ansur-ii-stature-tied-hand-length-and-br.md)
- **Summary:** ANSUR II stature-tied hand length and breadth (HRG-HX3).
- **Next step:** Shipped in PR #13840.

### DL-#13045 · Feat(Lifting): Barbell Visual Geometry and Lift Video Camera Framing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13045
- **Branch:** merged via #13823
- **PR:** #13823, #13845, #13835, #13833, #13832, #13944
- **Paths:** see #13823
- **Started:** 2026-10-26
- **Last verified:** 2026-10-28 (`01da1de0`; collated from changes/13045-osl4-contact-sheet-test.md)
- **Summary:** feat(lifting): barbell visual geometry and lift video camera framing
- **Next step:** Shipped in PR #13823.

### DL-#13464 · Docs: After-One-Solution Workflow Mapping, Runnable Example and Smoke Test for Loop_Nullspace

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13464
- **Branch:** merged via #13493
- **PR:** #13493
- **Paths:** see #13493
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`ed446ef4`; collated from changes/13464-docs-after-one-solution-workflow-mapping.md)
- **Summary:** Docs: after-one-solution workflow mapping, runnable example and smoke test for loop_nullspace
- **Next step:** Shipped in PR #13493.

### DL-#13463 · Muscle_Fiber: Affine-Hull-Aware Fibre Dimension (Implicit Equalities) and Hit-and-Run Sampling in Hull Coordinates; Corrected Dim Claims

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13463
- **Branch:** claude/nse-ud3-13463
- **PR:** #13467
- **Paths:** see #13467
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`43911da1`; collated from changes/13463-muscle-fiber-affine-hull-aware-fibre-dim.md)
- **Summary:** muscle_fiber: affine-hull-aware fibre dimension (implicit equalities) and hit-and-run sampling in hull coordinates; corrected dim claims
- **Next step:** Shipped in PR #13467.

### DL-#12688 · R9a Evidence: MeshCat Appearance Render Script, MuJoCo Stills, and Receipt (#12688)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12688
- **Branch:** merged via #13448
- **PR:** #13448
- **Paths:** see #13448
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`8861c3ad`; collated from changes/12688-r9a-evidence-meshcat-appearance-render-s.md)
- **Summary:** R9a evidence: MeshCat appearance render script, MuJoCo stills, and receipt (#12688)
- **Next step:** Shipped in PR #13448.

### DL-#13176 · Qualify HRG Full-Body-V2 Mass and Gravity in Drake

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13176
- **Branch:** merged via #13820
- **PR:** #13820, #14015
- **Paths:** see #13820
- **Started:** 2026-10-25
- **Last verified:** 2026-10-29 (`e9545fa7`; collated from changes/13176-hrg-q1-opensim-mass-matrix-and-gravity-q.md)
- **Summary:** Qualify HRG full-body-v2 mass and gravity in Drake
- **Next step:** Shipped in PR #13820.

### DL-#13796 · Add Benettin Largest Lyapunov Exponent Check and Receipt for Canonical Triple Pendulum

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13796
- **Branch:** merged via #13813
- **PR:** #13813
- **Paths:** see #13813
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`ff962751`; collated from changes/13796-benettin-largest-lyapunov-exponent.md)
- **Summary:** Add Benettin largest Lyapunov exponent check and receipt for canonical triple pendulum
- **Next step:** Shipped in PR #13813.

### DL-#13804 · Explicitly Load Pytest-Timeout in Motion-Matching Smoke

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13804
- **Branch:** merged via #13807
- **PR:** #13807
- **Paths:** see #13807
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`e41edfb7`; collated from changes/13804-explicitly-load-pytest-timeout-in-motion.md)
- **Summary:** Explicitly load pytest-timeout in motion-matching smoke
- **Next step:** Shipped in PR #13807.

### DL-#12851 · Split Proximal-Distal Research Docs Under Doc Size Budget (#12851)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12851
- **Branch:** merged via #13693
- **PR:** #13693
- **Paths:** see #13693
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`691bdfdb`; collated from changes/12851-split-proximal-distal-research-docs-unde.md)
- **Summary:** Split proximal-distal research docs under doc size budget (#12851)
- **Next step:** Shipped in PR #13693.

### DL-#13798 · Route Remaining MATLAB and Simscape Gravity Literals Through Canonical Standard Gravity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13798
- **Branch:** merged via #13808
- **PR:** #13808
- **Paths:** see #13808
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`0d43d5ca`; collated from changes/13798-route-matlab-gravity-canonical.md)
- **Summary:** Route remaining MATLAB and Simscape gravity literals through canonical standard gravity
- **Next step:** Shipped in PR #13808.

### DL-#13043 · Joint Reaction Loads for Deadlift, Snatch, and Clean and Jerk

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13043
- **Branch:** merged via #13521
- **PR:** #13521
- **Paths:** see #13521
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`5c867150`; collated from changes/13043-joint-reaction-loads-for-deadlift-snatch.md)
- **Summary:** joint reaction loads for deadlift, snatch, and clean and jerk
- **Next step:** Shipped in PR #13521.

### DL-#13304 · Consolidate Pinocchio Topology Contact Simulation Onto Shared G6 Model to Fix Swing Replay Divergence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13304
- **Branch:** merged via #13474
- **PR:** #13474
- **Paths:** see #13474
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`c1d832e8`; collated from changes/13304-consolidate-pinocchio-topology-contact-s.md)
- **Summary:** Consolidate Pinocchio topology contact simulation onto shared G6 model to fix swing replay divergence
- **Next step:** Shipped in PR #13474.

### DL-#13175 · HRG-E4 Slice: Desktop and Web Anatomical Event Table View With a Shared Golden Fixture and a Feature_Parity Gap Entry (Web Fetch Awaits #13538)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13175
- **Branch:** merged via #13543
- **PR:** #13543, #14012
- **Paths:** see #13543
- **Started:** 2026-10-25
- **Last verified:** 2026-10-29 (`fecb06f9`; collated from changes/13175-hrg-e4-web-anatomical-event-table-fetche.md)
- **Summary:** HRG-E4 slice: desktop and web anatomical event table view with a shared golden fixture and a feature_parity gap entry (web fetch awaits #13538)
- **Next step:** Shipped in PR #13543.

### DL-#13790 · Unify Triple-Pendulum Parameters and Correct MuJoCo Gravity Plane

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13790
- **Branch:** merged via #13797
- **PR:** #13797
- **Paths:** see #13797
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`dd3ac27a`; collated from changes/13790-unify-triple-pendulum-parameters-and-cor.md)
- **Summary:** Unify triple-pendulum parameters and correct MuJoCo gravity plane
- **Next step:** Shipped in PR #13797.

### DL-#13577 · Document Fast/Full Tier Contract Linkage in ci-standard.yml and Clarify Post-Merge P90 Observation Follow-Up (#13577).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13577
- **Branch:** merged via #13697
- **PR:** #13697, #13694
- **Paths:** see #13697
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`3bd0336a`; collated from changes/13577-public-repo-workflow-security.md)
- **Summary:** Document fast/full tier contract linkage in ci-standard.yml and clarify post-merge P90 observation follow-up (#13577).
- **Next step:** Shipped in PR #13697.

### DL-#13150 · Prove Partial UNBOUNDED Envelope Frames Reach the NSX Plot Model (#13150).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13150
- **Branch:** merged via #13731
- **PR:** #13731
- **Paths:** see #13731
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`2c6b9b8f`; collated from changes/13150-prove-partial-unbounded-envelope-frames.md)
- **Summary:** Prove partial UNBOUNDED envelope frames reach the NSX plot model (#13150).
- **Next step:** Shipped in PR #13731.

### DL-#12567 · Add a Zero-Side Consistency Shim for HARD_ZERO_TRAIL Allocation and Reject Algebraically Inconsistent Zero-Side Systems.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12567
- **Branch:** merged via #13606
- **PR:** #13606, #14016
- **Paths:** see #13606
- **Started:** 2026-10-25
- **Last verified:** 2026-10-29 (`2639f4cd`; collated from changes/12567-nsx24-frame-seam.md)
- **Summary:** Add a zero-side consistency shim for HARD_ZERO_TRAIL allocation and reject algebraically inconsistent zero-side systems.
- **Next step:** Shipped in PR #13606.

### DL-#13771 · Fix Tests/Api Assertions Stale After the APIException Error-Envelope Change; Shared Error_Message Fixture

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13771
- **Branch:** merged via #13772
- **PR:** #13772
- **Paths:** see #13772
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`982d54a1`; collated from changes/13771-fix-tests-api-assertions-stale-after-the.md)
- **Summary:** Fix tests/api assertions stale after the APIException error-envelope change; shared error_message fixture
- **Next step:** Shipped in PR #13772.

### DL-#13699 · Private Path Leak Ratchet, Denylist Patterns, and Mechanical Owner-Home Redaction Consolidation.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13699
- **Branch:** merged via #13756
- **PR:** #13756
- **Paths:** see #13756
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`ed5ad50d`; collated from changes/13699-private-path-leak-ratchet.md)
- **Summary:** Private path leak ratchet, denylist patterns, and mechanical owner-home redaction consolidation.
- **Next step:** Shipped in PR #13756.

### DL-#13770 · Add Cursor Cloud Environment.Json Fleet Bootstrap via Repository_Management Scripts/Cursor/Bootstrap.Sh.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13770
- **Branch:** merged via #13769
- **PR:** #13769
- **Paths:** see #13769
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`08dffcd6`; collated from changes/13770-add-cursor-cloud-environment-json-fleet.md)
- **Summary:** Add Cursor Cloud environment.json fleet bootstrap via Repository_Management scripts/cursor/bootstrap.sh.
- **Next step:** Shipped in PR #13769.

### DL-#13462 · Loop_Nullspace: Local Solution-Manifold Continuation (Predictor-Corrector With Rank Gates) Over ForceNullSpace (NSE-UD-2)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13462
- **Branch:** merged via #13470
- **PR:** #13470
- **Paths:** see #13470
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`dbd72379`; collated from changes/13462-loop-nullspace-local-solution-manifold-c.md)
- **Summary:** loop_nullspace: local solution-manifold continuation (predictor-corrector with rank gates) over ForceNullSpace (NSE-UD-2)
- **Next step:** Shipped in PR #13470.

### DL-#13461 · Loop_Nullspace: Classify_Functional (Numerical Fibre Identifiability With Optional Enclosing-Radius Bound) and Attribution_Quantities (Per-Source Induced-Acceleration Envelopes); Decomposition Now Stores Variable_Scale

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13461
- **Branch:** merged via #13475
- **PR:** #13475
- **Paths:** see #13475
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`eb00e096`; collated from changes/13461-loop-nullspace-classify-functional-numer.md)
- **Summary:** loop_nullspace: classify_functional (numerical fibre identifiability with optional enclosing-radius bound) and attribution_quantities (per-source induced-acceleration envelopes); Decomposition now stores variable_scale
- **Next step:** Shipped in PR #13475.

### DL-#12784 · Route Motion-Matching Nightly Smoke to GitHub-Hosted Runners on Public Repos.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12784
- **Branch:** merged via #13754
- **PR:** #13754, #13755
- **Paths:** see #13754
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`0f9ff771`; collated from changes/12784-motion-matching-smoke-hosted-runner.md)
- **Summary:** Route motion-matching nightly smoke to GitHub-hosted runners on public repos.
- **Next step:** Shipped in PR #13754.

### DL-#13725 · Make the Grip MJCF Independent of the BLAS Kernel and Libm: Deterministic Damping Design and Pad Ring Trig

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13725
- **Branch:** merged via #13727
- **PR:** #13727
- **Paths:** see #13727
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`8f67942d`; collated from changes/13725-make-the-grip-mjcf-independent-of-the-bl.md)
- **Summary:** Make the grip MJCF independent of the BLAS kernel and libm: deterministic damping design and pad ring trig
- **Next step:** Shipped in PR #13727.

### DL-#13548 · Correct the OpenSim Ten_Finger Swing Xfail Reason: Net_Force Now Passes; the Remaining 0.04 s Gap Is Net_Moment_Com (#13560)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13548
- **Branch:** merged via #13743
- **PR:** #13743
- **Paths:** see #13743
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`7c17a81d`; collated from changes/13548-correct-the-opensim-ten-finger-swing-xfa.md)
- **Summary:** Correct the OpenSim ten_finger swing xfail reason: net_force now passes; the remaining 0.04 s gap is net_moment_com (#13560)
- **Next step:** Shipped in PR #13743.

### DL-#13739 · Capture Rig Theme-Compliance Fixture Discards a ThemeManager Singleton Orphaned by QApplication Teardown (Fixes Flaky QSettings-Deleted Teardown Errors)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13739
- **Branch:** merged via #13741
- **PR:** #13741
- **Paths:** see #13741
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`88bf5384`; collated from changes/13739-capture-rig-theme-compliance-fixture-dis.md)
- **Summary:** Capture Rig theme-compliance fixture discards a ThemeManager singleton orphaned by QApplication teardown (fixes flaky QSettings-deleted teardown errors)
- **Next step:** Shipped in PR #13741.

### DL-#13534 · Disable Pytest-Xvfb Autoload for Motion-Matching Smoke; Document -P No:Xvfb in Run_Motion_Matching_Smoke.Sh

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13534
- **Branch:** merged via #13566
- **PR:** #13566
- **Paths:** see #13566
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`c474414f`; collated from changes/13534-motion-matching-smoke-xvfb-autoload.md)
- **Summary:** Disable pytest-xvfb autoload for motion-matching smoke; document -p no:xvfb in run_motion_matching_smoke.sh
- **Next step:** Shipped in PR #13566.

### DL-#12691 · R11: The R6 Chain Shaft Runs in Drake (RevoluteSpring Plus Joint Damping) and Passes the Same Static, Modal, Topology and Same-Input Parity Gate as OpenSim and MyoSuite; Drake Joins CHAIN_ENGINES

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12691
- **Branch:** merged via #13738
- **PR:** #13738, #13746, #13751
- **Paths:** see #13738
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`b443e958`; collated from changes/12691-r11-the-r6-chain-shaft-runs-in-pinocchio.md)
- **Summary:** R11: the R6 chain shaft runs in Drake (RevoluteSpring plus joint damping) and passes the same static, modal, topology and same-input parity gate as OpenSim and MyoSuite; Drake joins CHAIN_ENGINES
- **Next step:** Shipped in PR #13738.

### DL-#13618 · Install the Governed Canonical Admission Publisher Copy and Protected CI Caller With Independently Pinned Private Provider Runtime and Trusted Copy Drift Checking.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13618
- **Branch:** merged via #13645
- **PR:** #13645
- **Paths:** see #13645
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`89280c80`; collated from changes/13618-protected-admission-publication.md)
- **Summary:** Install the governed canonical admission publisher copy and protected CI caller with independently pinned private provider runtime and trusted copy drift checking.
- **Next step:** Shipped in PR #13645.

### DL-#13270 · Port Frame_Rate, Trunk_Spin, Hold_Limit_Jump, Rebase_Turns and Reference_Rate With Tests From Commit 22419Ee13 and Record the Failed Stance Probes in FORWARD_DYNAMICS.md

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13270
- **Branch:** merged via #13735
- **PR:** #13735, #13733, #13732, #13734, #13744, #13749
- **Paths:** see #13735
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`90f8c1de`; collated from changes/13270-port-gs3dx-com-shift-and-the-capture-con.md)
- **Summary:** Port frame_rate, trunk_spin, hold_limit_jump, rebase_turns and reference_rate with tests from commit 22419ee13 and record the failed stance probes in FORWARD_DYNAMICS.md
- **Next step:** Shipped in PR #13735.

### DL-#13255 · GS3DX GRF Exporter: Opt-In Lock_Joints (Gs3dx_Lock_Joints, in Memory) and DRY Shared Contact-Geometry / Sole-Point / CSV / SHA-256 Helpers; the Locked-Joint Stance Costs ~140 s Wall per Simulated ms

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13255
- **Branch:** merged via #13726
- **PR:** #13726, #13740, #13748
- **Paths:** see #13726
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`13cf33f7`; collated from changes/13255-add-the-simscape-rigid-body-stance-expor.md)
- **Summary:** GS3DX GRF exporter: opt-in lock_joints (gs3dx_lock_joints, in memory) and DRY shared contact-geometry / sole-point / CSV / SHA-256 helpers; the locked-joint stance costs ~140 s wall per simulated ms
- **Next step:** Shipped in PR #13726.

### DL-#13256 · Add Opt-In Allocate_Muscle_Excitations_Until_Feasible (Stop_When_Feasible) for Bounded Muscle Allocation: Stops During a Counted Evaluation on an Exact Feasible Witness and Reports Declared_Feasibility_Satisfied

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13256
- **Branch:** merged via #13691
- **PR:** #13691
- **Paths:** see #13691
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`669f4613`; collated from changes/13256-add-opt-in-stop-when-feasible-to-bounded.md)
- **Summary:** Add opt-in allocate_muscle_excitations_until_feasible (stop_when_feasible) for bounded muscle allocation: stops during a counted evaluation on an exact feasible witness and reports declared_feasibility_satisfied
- **Next step:** Shipped in PR #13691.

### DL-#12686 · Chain Shaft Runs Through the Ground-Support Pipeline: IK Holds Chain Coordinates at Rest, Face Targets Accept the Tip Link, Replay Uses the 0.125 ms Substep

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12686
- **Branch:** merged via #13719
- **PR:** #13719
- **Paths:** see #13719
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`c206545a`; collated from changes/12686-chain-shaft-runs-through-the-ground-supp.md)
- **Summary:** Chain shaft runs through the ground-support pipeline: IK holds chain coordinates at rest, face targets accept the tip link, replay uses the 0.125 ms substep
- **Next step:** Shipped in PR #13719.

### DL-#13017 · Fix DrakeKinematicsService.Set_Pose/Get_Link_Transforms Against a Real MultibodyPlant (Floating-Base Quaternion Layout, Body Enumeration)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13017
- **Branch:** merged via #13715
- **PR:** #13715, #13696
- **Paths:** see #13715
- **Started:** 2026-10-24
- **Last verified:** 2026-10-25 (`e5b10108`; collated from changes/13017-heavy-drake-lane-fixes.md)
- **Summary:** Fix DrakeKinematicsService.set_pose/get_link_transforms against a real MultibodyPlant (floating-base quaternion layout, body enumeration)
- **Next step:** Shipped in PR #13715.

### DL-#12723 · NSX-94: Run the Co-Contraction Trade-Off Sweep on the Real Pinned MyoSuite Elbow (Myoelbow_1Dof6muscles); Replace the Never-Running Importorskip(Myo_Sim) Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12723
- **Branch:** merged via #13717
- **PR:** #13717, #13723
- **Paths:** see #13717
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`5b486938`; collated from changes/12723-ci-the-tests-job-checks-out-only-the-pin.md)
- **Summary:** NSX-94: run the co-contraction trade-off sweep on the real pinned MyoSuite elbow (myoelbow_1dof6muscles); replace the never-running importorskip(myo_sim) test
- **Next step:** Shipped in PR #13717.

### DL-#12722 · NSX-93: Cross-Engine Conformance of the Declared Impedance and Delayed-Feedback Layer Through the Torque-Engine Derivative Seam (Pinocchio Reference, MuJoCo, Drake; Simscape UNAVAILABLE)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12722
- **Branch:** merged via #13716
- **PR:** #13716
- **Paths:** see #13716
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`6d4d5583`; collated from changes/12722-nsx-93-cross-engine-conformance-of-the-d.md)
- **Summary:** NSX-93: cross-engine conformance of the declared impedance and delayed-feedback layer through the torque-engine derivative seam (Pinocchio reference, MuJoCo, Drake; Simscape UNAVAILABLE)
- **Next step:** Shipped in PR #13716.

### DL-#12565 · Static_Envelopes Retries a NUMERICAL Warm-Started HiGHS Solve Cold (Anthro Plate Benchmark, Highspy 1.15.1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12565
- **Branch:** merged via #13714
- **PR:** #13714
- **Paths:** see #13714
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`66f09f23`; collated from changes/12565-static-envelopes-retries-a-numerical-war.md)
- **Summary:** static_envelopes retries a NUMERICAL warm-started HiGHS solve cold (anthro plate benchmark, highspy 1.15.1)
- **Next step:** Shipped in PR #13714.

### DL-#12720 · Guard Test Enforcing the Single Paired Monte Carlo Runner Rule for Strategy Comparisons (NSX-91)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12720
- **Branch:** merged via #13688
- **PR:** #13688
- **Paths:** see #13688
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`bbaa4153`; collated from changes/12720-guard-test-enforcing-the-single-paired-m.md)
- **Summary:** Guard test enforcing the single paired Monte Carlo runner rule for strategy comparisons (NSX-91)
- **Next step:** Shipped in PR #13688.

### DL-#2195 · Add Conflict-Disarm Caller Workflow (Dry-Run) and CI Allowlists for RM Reusable @main.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2195
- **Branch:** merged via #13570
- **PR:** #13570
- **Paths:** see #13570
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`d2ff79a4`; collated from changes/2195-add-conflict-disarm-caller-workflow-dry.md)
- **Summary:** Add conflict-disarm caller workflow (dry-run) and CI allowlists for RM reusable @main.
- **Next step:** Shipped in PR #13570.

### DL-#13579 · Record #13579 Closure Evidence (Docs/Operations/Evidence/shipped-even-issues-20261024.md)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13579
- **Branch:** cursor/issue-closure-receipts-66dc
- **PR:** #13685, #13986
- **Paths:** see #13685
- **Started:** 2026-10-24
- **Last verified:** 2026-10-28 (`dd75bc61`; collated from changes/13579-engine-docker-cache-contract.md)
- **Summary:** Record #13579 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md)
- **Next step:** Shipped in PR #13685.

### DL-#13171 · HRG-E3: Lower-Body, Shoulder, Elbow, and Line Turn Terms in ISB and Golf Language

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13171
- **Branch:** merged via #13623
- **PR:** #13623
- **Paths:** see #13623
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`addd8534`; collated from changes/13171-hrg-e3-lower-body-shoulder-elbow-and-lin.md)
- **Summary:** HRG-E3: lower-body, shoulder, elbow, and line turn terms in ISB and golf language
- **Next step:** Shipped in PR #13623.

### DL-#13104 · Record #13104 Closure Evidence (Docs/Operations/Evidence/shipped-even-issues-20261024.md)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13104
- **Branch:** cursor/issue-closure-receipts-66dc
- **PR:** #13685
- **Paths:** see #13685
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`501afd4c`; collated from changes/13104-close-shipped-issue-with-receipt-regress.md)
- **Summary:** Record #13104 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md)
- **Next step:** Shipped in PR #13685.

### DL-#13072 · Record #13072 Closure Evidence (Docs/Operations/Evidence/shipped-even-issues-20261024.md)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13072
- **Branch:** cursor/issue-closure-receipts-66dc
- **PR:** #13685
- **Paths:** see #13685
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`501afd4c`; collated from changes/13072-close-shipped-issue-with-receipt-regress.md)
- **Summary:** Record #13072 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md)
- **Next step:** Shipped in PR #13685.

### DL-#13683 · Route Native F02 and BoxFDDP Mass Reads Through Full_Mass_Matrix on MuJoCo 3.13+.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13683
- **Branch:** merged via #13684
- **PR:** #13684
- **Paths:** see #13684
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`5a385701`; collated from changes/13683-route-native-f02-and-boxfddp-mass-reads.md)
- **Summary:** Route native F02 and BoxFDDP mass reads through full_mass_matrix on MuJoCo 3.13+.
- **Next step:** Shipped in PR #13684.

### DL-#13660 · Wire Subscription CLI Adapters (Claude, Codex, Antigravity), Simulation HostActionPort Controls, and Local Whisper VTT Into Sidekick

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13660
- **Branch:** merged via #13664
- **PR:** #13664
- **Paths:** see #13664
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`434ae1bd`; collated from changes/13660-sidekick-cli-voice.md)
- **Summary:** Wire subscription CLI adapters (Claude, Codex, Antigravity), simulation HostActionPort controls, and local Whisper VTT into Sidekick
- **Next step:** Shipped in PR #13664.

### DL-#13479 · Bump Vendor/Ud-Tools to Tools 37A7477 (Owned Restartable UI Timers, Tools #5670; Chat Dock Stops Scroll Timers on Close, Tools #5674) and Re-Sync Src/Shared/Python/Chat

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13479
- **Branch:** merged via #13551
- **PR:** #13551
- **Paths:** see #13551
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`60f260f9`; collated from changes/13479-bump-vendor-ud-tools-to-tools-37a7477-chat-timers.md)
- **Summary:** Bump vendor/ud-tools to Tools 37a7477 (owned restartable UI timers, Tools #5670; chat dock stops scroll timers on close, Tools #5674) and re-sync src/shared/python/chat
- **Next step:** Shipped in PR #13551.

### DL-#13179 · Use Full_Mass_Matrix in Grip-Weld Axial Closure KKT Tests for MuJoCo 3.13+.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13179
- **Branch:** merged via #13682
- **PR:** #13682, #13689
- **Paths:** see #13682
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`2941e14f`; collated from changes/13179-forward-rollout-root-residual-test-close.md)
- **Summary:** Use full_mass_matrix in grip-weld axial closure KKT tests for MuJoCo 3.13+.
- **Next step:** Shipped in PR #13682.

### DL-#13044 · OSL-7 Predictive Deadlift: Solved Statically Supported Poses With Free Shoulders, a Planted-Feet Goal, Hicks Moment Gate and a 17-Run History; Stage 1 Still Does Not Converge With Foot Contact

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13044
- **Branch:** merged via #13512
- **PR:** #13512, #13729, #13721, #13737, #13736
- **Paths:** see #13512
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`f72e2b0b`; collated from changes/13044-wire-the-osl-7-predictive-versus-tracked.md)
- **Summary:** OSL-7 predictive deadlift: solved statically supported poses with free shoulders, a planted-feet goal, Hicks moment gate and a 17-run history; stage 1 still does not converge with foot contact
- **Next step:** Shipped in PR #13512.

### DL-#13555 · Grip Contact: Add Antagonist Pads Opposite the Heel, Trigger and Ten-Finger Little-Finger Pads so the Neutral-Pose Closure Preload Exerts Zero Net Force and Moment on the Club

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13555
- **Branch:** merged via #13666
- **PR:** #13666
- **Paths:** see #13666
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`955dd563`; collated from changes/13555-grip-contact-add-antagonist-pads-opposit.md)
- **Summary:** Grip contact: add antagonist pads opposite the heel, trigger and ten-finger little-finger pads so the neutral-pose closure preload exerts zero net force and moment on the club
- **Next step:** Shipped in PR #13666.

### DL-#13547 · Hand Off the HRG-GS Grip Parity Residuals: AGENT_HANDOFF Entry for Epic #13297 and Xfail Reasons Pointing at the Open Tracking Issues #13547 (MuJoCo Deflection) and #13548 (OpenSim Ten_Finger)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13547
- **Branch:** merged via #13666
- **PR:** #13666
- **Paths:** see #13666
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`955dd563`; collated from changes/13547-hand-off-the-hrg-gs-grip-parity-residual.md)
- **Summary:** Hand off the HRG-GS grip parity residuals: AGENT_HANDOFF entry for epic #13297 and xfail reasons pointing at the open tracking issues #13547 (MuJoCo deflection) and #13548 (OpenSim ten_finger)
- **Next step:** Shipped in PR #13666.

### DL-#13531 · Align Drake Autodiff Fit Theta With Canonical URDF Actuator Count

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13531
- **Branch:** merged via #13666
- **PR:** #13666, #13565
- **Paths:** see #13666
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`5343a4af`; collated from changes/13531-align-drake-autodiff-fit-theta-with-cano.md)
- **Summary:** Align Drake autodiff fit theta with canonical URDF actuator count
- **Next step:** Shipped in PR #13666.

### DL-#13450 · Drake Samples the Canonical Mid_Hands/Club_Head Frames Like Pinocchio; Legacy Grip Names Are Aliases Only and a Missing Frame Fails Closed Instead of Falling Back to Right_Hand or the Shaft

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13450
- **Branch:** merged via #13666
- **PR:** #13666, #13565
- **Paths:** see #13666
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`5343a4af`; collated from changes/13450-drake-samples-the-canonical-mid-hands-cl.md)
- **Summary:** Drake samples the canonical mid_hands/club_head frames like Pinocchio; legacy grip names are aliases only and a missing frame fails closed instead of falling back to right_hand or the shaft
- **Next step:** Shipped in PR #13666.

### DL-#13518 · CI: Set Rust-Cache Cache-Bin False so Post Steps Stop Deleting the Shared Host Rustup Binary (Refs Repository_Management#2138)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13518
- **Branch:** merged via #13518
- **PR:** #13518
- **Paths:** see #13518
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`00d8a71b`; collated from changes/13518-ci-set-rust-cache-cache-bin-false-so-pos.md)
- **Summary:** CI: set rust-cache cache-bin false so post steps stop deleting the shared host rustup binary (Refs Repository_Management#2138)
- **Next step:** Shipped in PR #13518.

### DL-#13109 · Lock Canonical Driver Seeds Closure and Weight-Fraction Gates in CI (#13109)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13109
- **Branch:** merged via #13665
- **PR:** #13665, #13446
- **Paths:** see #13665
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`b15fdbd9`; collated from changes/13109-lock-canonical-driver-seeds-closure-and.md)
- **Summary:** Lock canonical driver seeds closure and weight-fraction gates in CI (#13109)
- **Next step:** Shipped in PR #13665.

### DL-#12969 · Pull Club Sole to Ground at Address IK; Record Sole Height in Calibrated Receipt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12969
- **Branch:** merged via #13665
- **PR:** #13665
- **Paths:** see #13665
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`2aeda180`; collated from changes/12969-pull-club-sole-to-ground-at-address-ik-r.md)
- **Summary:** Pull club sole to ground at address IK; record sole height in calibrated receipt
- **Next step:** Shipped in PR #13665.

### DL-#12743 · Add Load_Coupling_Law Require_Verified Gate for SPN-3 Cited Coupling Data

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12743
- **Branch:** merged via #13665
- **PR:** #13665
- **Paths:** see #13665
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`2aeda180`; collated from changes/12743-add-load-coupling-law-require-verified-g.md)
- **Summary:** Add load_coupling_law require_verified gate for SPN-3 cited coupling data
- **Next step:** Shipped in PR #13665.

### DL-#12711 · Audit Near-Zero-Mean CV Callers: Return NaN With Reason in Cross-Engine Runner and Pendulum Variability Summary

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12711
- **Branch:** merged via #13665
- **PR:** #13665
- **Paths:** see #13665
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`2aeda180`; collated from changes/12711-audit-near-zero-mean-cv-callers-return-n.md)
- **Summary:** Audit near-zero-mean CV callers: return NaN with reason in cross-engine runner and pendulum variability summary
- **Next step:** Consume DisturbanceSpec in NSX-90 variability experiments

### DL-#13673 · Bump Transitive Undici to 8.12.3 in Ui/Package-Lock.Json to Clear HIGH Npm Audit Advisories That Ejected Every Merge-Queue Entry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13673
- **Branch:** merged via #13674
- **PR:** #13674
- **Paths:** see #13674
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`d9279f35`; collated from changes/13673-bump-transitive-undici-to-8-12-3-in-ui-p.md)
- **Summary:** Bump transitive undici to 8.12.3 in ui/package-lock.json to clear HIGH npm audit advisories that ejected every merge-queue entry
- **Next step:** Shipped in PR #13674.

### DL-#13658 · Add .oss-scanner Security-Scan Contract (project.yaml, Offline Audit Dockerfile, Threat Model) for the Fleet Scanner Pilot

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13658
- **Branch:** merged via #13659
- **PR:** #13659
- **Paths:** see #13659
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`2ca23f54`; collated from changes/13658-add-oss-scanner-security-scan-contract-p.md)
- **Summary:** Add .oss-scanner security-scan contract (project.yaml, offline audit Dockerfile, threat model) for the fleet scanner pilot
- **Next step:** Shipped in PR #13659.

### DL-#13638 · Skip Redundant OpenSim CI Runs When Unrelated PR Labels Are Applied

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13638
- **Branch:** merged via #13668
- **PR:** #13668
- **Paths:** see #13668
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`23b3f4df`; collated from changes/13638-skip-opensim-unrelated-labels.md)
- **Summary:** Skip redundant OpenSim CI runs when unrelated PR labels are applied
- **Next step:** Shipped in PR #13668.

### DL-#13670 · Bump Ui Lockfile (Zod 4.6.7) to Clear GHSA-Mrhx-C7xc-M3w5 so the UI Security Audit and Merge Queue Pass

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13670
- **Branch:** merged via #13671
- **PR:** #13671
- **Paths:** see #13671
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`fb5706fd`; collated from changes/13670-bump-ui-lockfile-zod-4-6-7-to-clear-ghsa.md)
- **Summary:** Bump ui lockfile (zod 4.6.7) to clear GHSA-mrhx-c7xc-m3w5 so the UI Security Audit and merge queue pass
- **Next step:** Shipped in PR #13671.

### DL-#12942 · Model Explorer Headless Export: Support NPZ Trajectories With 'Q', Document and Verify Results Replay for Anthro Driver Mjx_Package With Stiff Weld Constraint

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12942
- **Branch:** merged via #13425
- **PR:** #13425
- **Paths:** see #13425
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`9cc0160e`; collated from changes/12942-model-explorer-headless-export-support-n.md)
- **Summary:** Model explorer headless export: support NPZ trajectories with 'q', document and verify results replay for anthro driver mjx_package with stiff weld constraint
- **Next step:** Shipped in PR #13425.

### DL-#13334 · Critique Physical Grip-Contact Calibration and Validation Evidence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13334
- **Branch:** merged via #13661
- **PR:** #13661
- **Paths:** see #13661
- **Started:** 2026-10-23
- **Last verified:** 2026-10-23 (`46a71b8a`; collated from changes/13334-critic-grip-contact-2026-10-23.md)
- **Summary:** Critique physical grip-contact calibration and validation evidence
- **Next step:** Shipped in PR #13661.

### DL-#13608 · Exclude Generated Capability-Atlas Output From Tauri Workflow Path Triggers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13608
- **Branch:** staff/issue-remediator-13608-119eb1
- **PR:** #13652
- **Paths:** .github/workflows/tauri-build.yml,tests/ci/test_ci_infrastructure.py
- **Started:** 2026-10-23
- **Last verified:** 2026-10-23 (`b22f0717`; collated from changes/13608-exclude-generated-capability-atlas-outpu.md)
- **Summary:** Exclude generated capability-atlas output from Tauri workflow path triggers
- **Next step:** Shipped in PR #13652.

### DL-#13626 · Prevent Duplicate Xvfb Startup in the Motion-Matching Smoke Command.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13626
- **Branch:** merged via #13635
- **PR:** #13635
- **Paths:** see #13635
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`8117cbc1`; collated from changes/13626-motion-smoke-display.md)
- **Summary:** Prevent duplicate Xvfb startup in the motion-matching smoke command.
- **Next step:** Shipped in PR #13635.

### DL-#13619 · Use Python 3.12 for the Reviewed Nightly Cross-Engine Dependencies.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13619
- **Branch:** merged via #13635
- **PR:** #13635, #13755
- **Paths:** see #13635
- **Started:** 2026-10-22
- **Last verified:** 2026-10-24 (`0f9ff771`; collated from changes/13619-nightly-hosted-runner-routing.md)
- **Summary:** Use Python 3.12 for the reviewed nightly cross-engine dependencies.
- **Next step:** Shipped in PR #13635.

### DL-#13568 · Quarantine CI Requires OpenSim Lane (RM #2252) Until #13564 Fix Lands.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13568
- **Branch:** merged via #13614
- **PR:** #13614
- **Paths:** see #13614
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`efff779f`; collated from changes/13568-quarantine-ci-requires-opensim-lane-rm-2.md)
- **Summary:** Quarantine CI Requires OpenSim lane (RM #2252) until #13564 fix lands.
- **Next step:** Shipped in PR #13614.

### DL-#13564 · Restore CI Requires OpenSim: Squat JR Kinematics Revision 6 and Native Muscle-Effort Allocation.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13564
- **Branch:** merged via #13615
- **PR:** #13615, #13647
- **Paths:** see #13615
- **Started:** 2026-10-22
- **Last verified:** 2026-10-23 (`3368efe0`; collated from changes/13564-remove-stale-ci-requires-opensim-lane-qu.md)
- **Summary:** Restore CI Requires OpenSim: squat JR kinematics revision 6 and native muscle-effort allocation.
- **Next step:** Close #13564; the real fix landed in #13615 and has been verified green on main across multiple CI runs.

### DL-#13572 · Gate Workflow Edits With Actionlint; Fix Cross-Engine Leaderboard Publish Concurrency Expression

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13572
- **Branch:** merged via #13616
- **PR:** #13616, #13631, #13680
- **Paths:** see #13616
- **Started:** 2026-10-22
- **Last verified:** 2026-10-24 (`9497fdb6`; collated from changes/13572-pin-live-required-check-export-for-ud-ci.md)
- **Summary:** Gate workflow edits with actionlint; fix cross-engine leaderboard publish concurrency expression
- **Next step:** Shipped in PR #13616.

### DL-#13629 · Schedule Nightly Development Log Archiver Workflow and Noop Contract Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13629
- **Branch:** merged via #13628
- **PR:** #13628
- **Paths:** see #13628
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`29cb48eb`; collated from changes/13629-schedule-nightly-development-log-archive.md)
- **Summary:** Schedule nightly development log archiver workflow and noop contract test
- **Next step:** Shipped in PR #13628.

### DL-#13621 · Reconcile Development Log Against GitHub PR Truth and Archive Terminal Rows for Fleet Policy Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13621
- **Branch:** merged via #13621
- **PR:** #13621
- **Paths:** see #13621
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`c28cdb20`; collated from changes/13621-reconcile-development-log-against-github.md)
- **Summary:** Reconcile development log against GitHub PR truth and archive terminal rows for fleet policy gate
- **Next step:** Shipped in PR #13621.

### DL-#5915 · Remove @dieterolson From CODEOWNERS; Keep Team-Based Ownership

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5915
- **Branch:** merged via #13617
- **PR:** #13617
- **Paths:** see #13617
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`3dfbb4dc`; collated from changes/5915-remove-dieter-codeowners.md)
- **Summary:** Remove @dieterolson from CODEOWNERS; keep team-based ownership
- **Next step:** Shipped in PR #13617.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.
No material development-log change — Bolt `np.linalg.norm` → `einsum` consolidation (#11073, #11074, #11076) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — Bolt `np.linalg.norm` → `sqrt(einsum)`/`math.sqrt(np.vdot)` consolidation (#11112, #11128, #11129) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — PyJWT floor/lock bump to 2.14.0 for OSV GHSA-w6j9-cwv2-h6wq (#11153) is a dependency-only change with no feature entry.
No material development-log change — urllib3 (2.8.0, CVE-2026-97687) and PyJWT (2.15.0, CVE-2026-101918) floor/lock bumps (#11191) are dependency-only changes with no feature entry.
No material development-log change — Bolt np.linalg.norm → sqrt(einsum) in rigidity.py (#11189) is a behaviour-preserving micro-optimisation with no feature entry.

## DL-#11357 — Pure Raw Shaft Fragment Derivative Diagnostics

- **Change:** Canonical immutable options/check/result contracts and public
  raw fragment analytic/central FD diagnostics; canonical camera/marker reuse.
- **Validation:** Repository fail-first scaffold regression followed by focused
  native-free synthetic tests, curated exports and fingerprint inclusion.
- **Limits:** No historical native/Library/render/fit operation, no objective
  or evidence-role change, and no clinical/scientific qualification.
- **Next Step:** Review typed core, then separately design authenticated
  workspace consumers using existing job/store/renderer boundaries.

## Authored Replay Impact Application Checkpoint — #11464

The normal React journey now previews impact from a verified authored replay,
with exact geometry/sample/flight declarations, bounded native execution,
cancellation, persisted run-ID recall and authenticated four-file download.
Saved results display their own assumptions and metrics. Canonical receipt and
trajectory owners remain unchanged; source/runtime/parent and artifact checks
fail closed. Research acceptance remains rejected, with authored seconds and
false physical/scientific qualification.

Root final validation: 125 focused Python cases pass with one Windows symlink
capability skip; one additional opt-in native case passes with actual MuJoCo,
clean process transport, impact solver and Rust flight (no substitutions).
The React cohort passes 104 cases; configured six-owner mypy, scoped lint and
format checks pass. Earlier failures and immutable agent freezes are retained
in the application review. Four-page LaTeX Application Supplement 1 is compiled
without final layout warnings and all pages visually reviewed; editable source,
PDF and the explicitly synthetic native bundle are saved on the local Desktop.
Earlier Methods V8 and the five-page extraction report remain unchanged.

See [Application Procedure](necromatcher-replay-impact.md) and its linked review/report. Tiger stays
within original frames 0–190 ([0,191)); Hogan stays [0,750). No historical
Library mutation, qualified match or new ControlTower delivery is established.
Remaining goal work includes improved historical matching, authored dynamics,
simulation golf integration and desktop parity. The full goal remains active.

## Native Desktop Replay Impact Checkpoint — #11487

The native Necromatcher tile now imports registered authored replays, runs the
shared bounded replay-impact session off the GUI thread, cancels owned jobs,
recalls retained run IDs after restart and exports an authenticated four-file
ZIP exclusively. Saved assumptions remain separate from editable declarations.
The public Shot Tracer handoff retains exact detached positions and provenance.

Root validation passes 146 focused Qt/viewer/parity cases without skips and
configured mypy on three production owners. One additional opt-in actual
MuJoCo/HDF5/clean-worker/impact/Rust native Qt case passes with Python exit 0.
The three final synthetic screenshots are readable and visually reviewed;
retained positions are verified, while OpenGL pixel accuracy is unverified.
Test-only font bootstrap uses existing Windows Segoe UI when the offscreen
font database is empty. The 120-second whole-test cap does not change the
60-second native budget; earlier timeout evidence is preserved.

The three-page Native Desktop Supplement 1 and editable LaTeX, screenshots and
intact synthetic bundle are saved on the local Desktop. See [Procedure](necromatcher-replay-impact.md)
and [Frozen Review](historical_capture/native-desktop-replay-impact-review.json). Tiger remains [0,191), Hogan [0,750).
Historical fits, scientific qualification, simulation golf and final overall
desktop parity acceptance remain open. No new ControlTower delivery or remote
CI-green claim is made. The full goal remains active.

## Local Research Admission Race Follow-Up — #11493

A concurrent request regression reproduced reused shot-ID admission after another
request prepared and cancelled during authentication. The API now applies the
same identity guard before and immediately after authentication. All 10 API
cases pass; a fresh additional actual MuJoCo/clean-worker/Rust API and Qt
acceptance passes in 24.56 s, with seven exact retained samples and no
re-simulation on recall. Configured mypy passes for the changed API owner.
The earlier 209-case cohort and 106 React cases remain separate checkpoint
evidence. The follow-up preserves unverified qualification and all historical
limits; the full goal remains active.

See [Follow-Up Review](historical_capture/local-research-golf-admission-race-review.json). Source inventory covers 6,697 src/test
files, with equal before/after maps; this is not a whole-repository baseline.

## User-Requested Handoff and Stop Checkpoint

The user requested a committed checkpoint and instructed this agent to stop
pursuing the goal after turnover. See [Next-Agent Turnover](necromatcher-next-agent-turnover.md) for the
exact implementation state at f0e14944, scoped work packets, public owners,
validation commands, historical evidence boundaries and remaining acceptance.
No proposed optimized video export or new fit was started. This documentation
checkpoint follows the published local research integration; the full goal is
unfinished and is to be paused after normal commit/push and handoff publication.

## Capture Registry Parent Integration - #11162 / #11172

Accepted-main synchronization preserves capture-registry, export and swing-comparison source changes. Record conflicts retain distinct entries from both branches. All 133 ordered matched-swing ledger rows and non-timestamp metadata are identical; retain accepted-main timestamp. Regenerate the monolith register from merged source. Scientific matching/dynamics acceptance remains separate.

Record validation: all five monolith-register tests pass; no duplicate primary headings or conflict markers; all 133 ledger rows and non-timestamp metadata preserved exactly; registry/export/comparison source unchanged. Broader capture tests are pending pinned Tools initialization; first collection attempt stopped on the missing submodule. Protected parent CI and full scoped rerun remain required before merge.

## Capture Comparison DRY Gate Remediation - #11162 / #11172

CI identified duplicated club endpoint selection in angular-speed and wrist metrics. A shared private helper preserves explicit endpoint precedence, measured-marker aliases, array identity and missing-endpoint/shaft-axis behavior. AGY Gemini Flash supplied the bounded extraction; root reviewed the exact diff. Existing focused regression suite passes 190 tests with 15 unavailable-private-workbook skips, zero errors/failures, natural exit 0. Ruff format/check and full DRY duplication gate pass without changing baselines. Regenerate the monolith register for the revised file length. Current-head protected CI remains required.

## Capture Leaderboard Suite Classification - #11162 / #11172

Protected CI verified the DRY extraction, then identified 14 net-new unmarked tests in TestPerCaptureLeaderboard. Mark this pure-Python class as unit; all 14 execute and pass under unit selection with no skips/errors. Full suite-marker ratchet passes without baseline changes. Broader focused capture/comparison qualification remains 190 passed and 15 unavailable-private-workbook skips. Protected current-head CI remains required before merge.

## Capture Comparison Inventory Integration - #11162 / #11172

CI at bd1ee9a8e passes repository structure and 20,546 unit cases; its only unit failure identifies five new swing_comparison paths missing from the shared-tools divergence inventory. Regenerate JSON and Markdown with the authoritative generator against the pinned Tools tree. All five paths classify as UD-only; existing overlap totals remain unchanged. All ten inventory tests pass with no errors/skips, including the committed-tree freshness test, and generator --check succeeds. Protected current-head CI remains required before merging.

### Selected Club Continuity Update (2026-10-03)

PR #11256 merged at `7db86ff173653a235ea63bb4af347af0a8a5ab83`. The eight verified Human ellipsoid MP4s remain on both Desktops in `Simscape_Matches_20261003_Club_Refined_1080p`; these are IK reconstructions. The source reference now presents this current selection before historical experiments.

The selected club fits were tested at all anchors and midpoints using independent native position, velocity and fresh 91-state checks. Sequential seeding accepted 1,306/1,307 tour states and 732/733 owner states. A targeted diagnostic recovered the owner's rejected original anchor from its saved closed-arm seed; the previous accepted pose reproduced an alternate branch. A complete anchor-seeded rerun exited naturally at 20:31:23 UTC and accepted all 733 owner states (367 anchors and 366 midpoints), while the tour still rejected midpoint 998 at 1.3847222 s. All original anchors matched within 1.73e-15 physical residual. The peak accepted dependent angular rates remain approximately 1,639 and 4,118 deg/s; sampled-state acceptance does not qualify between-sample branch continuity, dependent acceleration, anatomy, contact or forward dynamics.

The private production integration candidate passed 17 native pre-model contract cases (parent RED, candidate GREEN, 51 recorded checks) at 20:20:58 UTC. Numerical whole-output parity and positive-fit validation remain separate pending gates. Original failed runner receipts are preserved. A custom vector measurement block built successfully, but the eight-axis instrumented simulation hit the Home license's 1,000-nonvirtual-block limit. No all-35 measured torque or moving feedback-off replay claim follows from these tests. Original physical model files were not saved.

See the separate editable research reference and the extended aggregate `tangent_c2_checkpoint_20261003.json` for scope and provenance. The built-in LaTeX compiler still fails with `Unable to find standard directories for platform`; PDF compilation/page review are unverified. Canonical calculation inventory and governed manual release remain blocked; this update grants neither a release exemption nor scientific approval.

## F05d Native Manifold BoxFDDP Candidate (#11932; 2026-10-09)

A test-first MuJoCo 3.8/Crocoddyl 3.2.1 child now solves the same
contact-free floating-root/two-hinge physical state with a 16-dimensional
configuration/velocity tangent, exact native Euler action derivative,
bounded direct motor input and nonlinear native fallback admission. The
paired source-hashed receipt measures BoxFDDP against SciPy native shooting
under identical reference, truth, input box, horizon, iteration and
cooperative wall budget, alternates solver order and charges a common
Tools T01 bootstrap to both. Every applied input independently replays from
complete native integration state. This is synthetic controller evidence,
not a hard deadline, measured capture fit or full-body qualification;
the F05 parent and canonical manual publication remain blocked. Read
chapter 26 and `F05D_NATIVE_MANIFOLD_BOX_FDDP_TURNOVER.md`.

## Bounded Nonuniform Moco Refinement Plan (#12244; 2026-10-10)

The 20-cell synthetic Moco parent inputs are retained byte-for-byte and the
offline activation report reproduces from those committed source, request,
solution, control and replay-audit artifacts. A pure source-bound planner marks
the observed full Hermite-Simpson cells, caps one child pass at four marked
cells and 24 intervals, and rejects evidence/hash or grid mismatches. The worker
uses the normalized custom mesh and parent state seed, then linearly samples
the persisted controls at every child endpoint and midpoint without control
trajectory resampling. Pure planner/interpolation regressions pass. No native
child solve or replay was run; solver API/time-grid acceptance and all physical
qualification remain open. The context catalog could not be refreshed here:
its `vendor/ud-tools` junction escapes this checkout to the frozen
`feedback-drake-native-11838` donor, and the context CLI correctly rejects that
path. See the #12244 change fragment, chapter 13 and F07 ladder turnover.

The follow-up recomputes and compares the full plan before solver setup and
builds the post-solve mesh audit from the seeded grid in the child request.
Pure JSON-reload and tampering regressions pass; there is still no native child
solve or replay evidence.

The next correction uses the OpenSim 4.6-compatible states-table insertion
signature and restores exact declared initial values after interpolation; a
strict fake-SWIG test covers the behavior without running the native solver.

An installed OpenSim 4.6 API smoke also passed for the committed parent STOs:
the states table inserted into a native 43-row `MocoTrajectory`, controls were
PWL-seeded, and cold initial values matched exactly. It ran under the 30-second
owned-tree supervisor with 140 MB sampled peak, and called no model init, solve,
or replay. A first malformed empty-trajectory smoke crashed in a SWIG setter;
the corrected data-bearing constructor passed. Actual solver acceptance remains
unverified.
