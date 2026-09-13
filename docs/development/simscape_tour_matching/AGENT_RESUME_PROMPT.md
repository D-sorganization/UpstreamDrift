# Copy-Ready Agent Resume Prompt

Resume the original tour-average C3D forward-dynamics matching goal. The work
is in progress and not completed. Pursue the full
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
3. native_evidence/two_window_floor_9967_77/HANDOFF.md, then the
   two_window_fit_9967_78, \_79 and \_80 HANDOFF/receipts (latest trials).
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

## Actual Starting Point (Updated 2026-09-13)

Run73 remains the last cleanly returned single-shooting candidate (0.85 s only,
whole RMS28.105 mm, terminal65.398 mm, SHA256 786522cd…). Audit77 qualified every
run76 derivative block against measured replay/retraction floors with unchanged
gates (native_evidence/two_window_floor_9967_77): the integration is max-step
limited, per-replay noise is step-sequence roundoff (q about1e-9, qd1.6e-8), and
the two near-zero node blocks are structural zeros verified from chart
orthonormality. Do not re-audit derivatives unless the fixture or providers change.

Two-window direct-node SLSQP trials78/79 reproduce run73's objective at zero
displacement to1.9e-12 (once-only shared boundary, run73 effort penalty).
Trial78 exhausted ten evaluations and returned the start; trial79 (primal-only
residuals,15 iterations) returned5313c283… whose uninterrupted original-state
replay gives whole27.563 mm, terminal55.470 mm, club23.27 mm but early11.359 mm
and pelvis yaw11.3 %, with63 of123 variables at bounds and scaled continuity
defect3.99e-4. Run80 (node recentered on returned79's integrated state) returned
96c786ec… with whole26.797/terminal55.208 mm; run81 (box widened to ±4 N/Nm)
returned dfafdff1… with whole26.366/terminal46.305 mm but early11.427 mm and
pelvis yaw13.9 %. All are still REJECTED; all jobs are terminal. The next factor
(weighting, node box or horizon) needs its own receipt; start from returned81.
The terminal weight (100x) only reshapes the last0.1 s; error growth from0.4 s
is unchanged (two_window_fit_9967_79/marker-comparison.png).

Preserve exact coefficients and absolute clock when restarting: apply Bernstein
increments to the exact returned candidate (never reconstruct from the parent),
recompute the physical box (±2 N/Nm over parent19) relative to the restart, and
record the roundoff. Any change of box, node bound, weighting or horizon needs
its own receipt. Use runtime78 (or a new qualified clone) and the run79/80
driver pattern: primal replays for residuals, sensitivities only for Jacobians,
systemd-run --user launches through run_job.py, receipts with driver hashes.

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
