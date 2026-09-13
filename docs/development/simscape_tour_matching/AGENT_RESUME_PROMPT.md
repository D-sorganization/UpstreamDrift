# Copy-Ready Agent Resume Prompt

Resume the original tour-average C3D forward-dynamics matching goal. This work
was paused at the user's request for handoff, not completed. Pursue the full
objective after reading the current evidence; do not substitute a short prefix,
feedback tracking, pose fitting or representation roundtrip for the final result.

## Workspace, Coordination and First Reads

Worktree: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native.
Branch: feat/9967-native-simscape-pinocchio.
Repository: D-sorganization/UpstreamDrift.
Parent epic9921; native implementation9967; representations10043; OpenSim10003.

Read AGENTS.md, CLAUDE.md, docs/development/HANDOFF.md, and the following files:

1. docs/development/simscape_tour_matching/NEXT_AGENT_CONVERGENCE_EXECUTION.md.
2. native_evidence/regularized_fit_9967_73/HANDOFF.md under that directory.
3. native_evidence/two_window_preflight_9967_75/HANDOFF.md and
   native_evidence/two_window_derivatives_9967_76/HANDOFF.md.
4. REPRESENTATION_HANDOFF.md, PINOCCHIO_MANIFOLD_HANDOFF.md, and
   docs/development/mujoco_native_matching/MUJOCO_MANIFOLD_HANDOFF.md.
5. Existing Drake and OpenSim turnover documents linked from the main handoff.

Verify git status, HEAD and origin before edits. Renew the appropriate issue lease
through Repository_Management/scripts/check_agent_claim and post_agent_lease,
using the actual agent identity. Respect another agent's lease and preserve user
changes. Use topic-branch commits and normal checks; never push to main.

## Non-Negotiable Final Outcome

Produce an uninterrupted forward-dynamics replay from the original initial state
through all654 capture frames/1.813888889 seconds. Optimize native actuator inputs
as continuous global degree-six polynomials; retain traceable geometry/attachments,
joint conventions and constraints. Preserve valid-marker masks and existing
marker, closure, effort and early-motion acceptance gates. Keep the original
polynomial coefficients and absolute clock when resuming. If changing basis or
coverage, document the transformation and verify same-input replay first.

MATLAB R2025b is the required final reference release. R2026a is not required and
cannot substitute for R2025b acceptance. Pinocchio, MuJoCo and Drake native and
alternate representations must preserve physical properties and actuator routing.
URDF alone does not encode the closed loop; retain the canonical model specification,
sidecar and explicit rigid-closure adapters. Do not silently substitute stock
MuJoCo compliant equality dynamics, add inertias or release constrained joints.

## Actual Starting Point

Run73 is the latest cleanly returned candidate, still rejected:0.85 seconds only,
whole RMS28.105 mm, terminal65.398 mm, early10.860 mm. All ten Jacobian-primal
checks passed, but optimization did not converge. Its canonical candidate SHA256 is
786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a.
Use returned-candidate.json and sampled-markers-state.npz in that evidence folder.
The NPZ contains actual307-by-54 q/qd states, markers and clock; it does not contain
qdd. marker-comparison.png is the measured rejected-prefix visual.

The original remote model hash begins b817fea; the formatted repository copy begins
0202c8b2. Preserve both raw and canonical identities rather than confusing formatting
differences with physics changes. Read the recorded full hashes in the receipts.

Run75 passes the two-window baseline at the saved0.6 s interior node: marker
discrepancy5.369e-12 m against uninterrupted73, closure and full42-dimensional
q/v retraction qualified. Use its saved basis, scales and retraction derivatives.
Do not repeat fixture75 or rebuild static pose seeds unnecessarily.

Run76 completed all16 signed trials with process exit0 but scientific status
failed_derivative_gates. LS B6 at h1e-4 and mixed direction at h1e-6 passed all
blocks; smaller steps failed some resolved checks. Node directions passed resolved
blocks but failed relative checks on near-zero cross-continuity derivatives:
analytic norms about5e-14/2e-17 versus finite-difference norms about1e-10/1e-11.
Establish a measured, physical absolute-error floor or independently verify the
structural-zero components; do not demand relative accuracy against zero and do
not indiscriminately waive failed checks. No optimizer was launched.

Run76 is the bounded native directional-derivative audit. Its terminal receipt
and dedicated HANDOFF are authoritative. Inspect every direction, step size,
marker block, q/qd endpoint block and continuity block. Weak or step-dependent
results are not an unconditional pass. Do not start an optimizer merely because
the augmented primal checks passed. Diagnose any failed or unresolved derivative
at the exact saved fixture with a bounded experiment and an integration-error floor.

## Immediate Execution Path

1. Verify all prior processes are terminal from receipts/handles before any launch.
   Do not restart a job because an observation timed out.
2. Resolve run76's remaining derivative questions without another blind fit. Reuse
   native_sensitivity, node_retraction and the recorded perturbation directions.
3. Only after qualification, use the existing multi_shooting_fit SLSQP backend
   for a bounded two-window trial. Set
   window_jacobian_state_coordinates="node" to consume direct chart sensitivities;
   do not multiply them by a pseudoinverse just to reconstruct ambient columns.
   Retain physical endpoint rows and the negative next-node retraction derivative
   in continuity. The new option defaults to legacy physical-state behavior.
4. Preserve the exact run73 base coefficients and start with zero increments.
   Derive equivalent bounds from saved optimizer parameters, recording roundoff;
   do not silently reconstruct a different initial candidate. Use the same
   control subset/objective initially to isolate the formulation change. Check
   shared-boundary observation counting: current MS includes both window boundary
   samples; document or explicitly test a once-only policy for objective parity.
5. Initialize nodes from integrated states. Chart bounds protect local validity;
   they must not recreate permanent tight boxes around static target poses.
   Report projected rank, active bounds and predicted versus actual reduction.
   Recenter charts only after an independently accepted step.
6. Require useful uninterrupted improvement before expanding horizons, adding nodes
   or unlocking controls. Final acceptance always uses original-state replay with
   zero intermediate resets, then independent R2025b validation and visual review.

Do not repeat run19's126 active bounds or run38's apparently good segmented fit
with89.22 continuity defect and201.2 mm uninterrupted terminal error. Audit74 finds
marker-Jacobian condition7.45e7, still3.77e7 after column normalization. Sampled
rotation-chart conditions alone are much smaller; quaternions are not a proven cure
for the complete trajectory-control conditioning. The fixed-attachment rigidity
floor is17.393 mm RMS; three head markers share Hub. Keep that model limitation
visible, and identify any physical variant separately rather than dropping markers.

## Compute and Reproducibility

Use SSH alias controltower and WSL distribution ControlTower-Runner. Native Python:
/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python.
Frozen numeric runtime73: /home/dieterolson/native-regularized-fit-9967-73.
Set PYTHONPATH to the selected frozen runtime, OPENBLAS_NUM_THREADS=1 and
OMP_NUM_THREADS=1. Remote inputs are under /mnt/c/Users/diete:
native_geometry_spec_9967.json and driver_marker_payload_9967.json.

Clone a fresh runtime before incorporating new provider code; do not edit frozen
runtime73. The direct-node fitter option is newer than runtime73. Qualify exact
source bytes and relevant real-engine tests before execution. Qualified sensitivity
settings are grouped control, max_step0.0000625, rtol1e-10/atol1e-12 and200000
augmented calls per window; independent primal uses1e-11/1e-13. Inspect receipts
for actual overrides. Budgets are not proof of accuracy or convergence.

Preserve launch commands, environment, source/input hashes, failure/terminal receipts,
raw arrays, exact candidate coefficients, masks and original clocks. Save actual
state trajectories when needed; state_jacobian is not a state trajectory. Keep
failed experiments immutable. Do not rerun solely to recreate missing prose.

## Representation and Other-Engine Work

Reuse pose_interchange: NativeJointStateAdapter, NativeMotionSequence,
export/restore_native_motion, load/save_native_motion, SerialRotationChart and
FixedFrameTransport. File I/O is versioned and atomic. Preserve units, wxyz versus
xyzw conventions, source/target frames, winding/branch references, convective
acceleration and dual effort/power maps. Unsupported singular inverses must fail.

Pinocchio spherical dynamics exists, but robust trajectory equivalence is incomplete.
MuJoCo spherical MJCF export has verified compiled geometry/inertia/closure and FK;
its tangent/effort mapping and rigid-closure dynamics still need implementation.
Drake alternate dynamics and full moving-frame acceleration transport remain open.
OpenSim has a staged epic; do not claim runtime acceptance from its plan. Coordinate
bounded parallel ownership without diverting the primary matching lane.

## Validation and Turnover

Use TDD, explicit shape/finite/unit/frame contracts, DRY shared providers and narrow
dependencies. At minimum rerun relevant shooting/node tests, native sensitivity
tests and real-engine checks for changed providers, plus Ruff/mypy and normal hooks.
The direct-node option passed44 combined shooting/node tests; optional engine skips
are not qualification. Keep HANDOFF.md, DEVELOPMENT_LOG.md and per-run turnover
current in every implementation commit. Push incremental checkpoints.

Stop only for a genuine external blocker or user-requested pause. Do not declare
the goal complete until full-swing matching and required cross-engine/R2025b gates
are verified. Report limitations candidly and leave evidence the next agent can run.
