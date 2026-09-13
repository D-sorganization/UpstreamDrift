# Next Agent: Qualified Two-Window Convergence Trial

## Objective and Current Evidence

Continue epic9921/native9967 and representation10043. Match the original tour
capture with uninterrupted forward dynamics and global degree-six native actuator
inputs. MATLAB R2025b remains the final reference. Read ../HANDOFF.md first for
the latest terminal fit and any live process; do not restart an active job.

Fit73 is the currently authorized bounded single-shooting experiment. It uses
grouped sensitivity control, max_step0.0000625 and200000 sensitivity calls per
evaluation. Diagnostic72 passes its exact restart candidate at2.5069e-8 m marker
agreement. Numerical reliability improved, but marker matching improvements
remain small. Audit74's marker/terminal Jacobian condition is7.452e7 and remains
3.767e7 after column normalization; penalty rows and bounds were excluded.
This evidence is not proof of a particular nonlinear failure mechanism.

## Preserve the Existing Providers

Use these existing modules; do not implement another solver or convention stack:

- motion_matching/multi_shooting_fit.py: window and node Jacobians, node transforms,
  scaled defects, SLSQP equality backend and uninterrupted final rollout.
- motion_matching/node_retraction.py: closure-preserving q/v retraction, tangent
  scaling and implicit derivatives.
- motion_matching/shooting_schedule.py: exact capture-sample window boundaries.
- motion_matching/shooting_state_seed.py: identity-checked sample selection only;
  it does not establish feasibility.
- pinocchio/python/native_sensitivity.py: original-state/window sensitivities;
  supplied initial-state tangent columns follow control columns.
- motion_matching/native_effort_profile.py and native_candidate.py: native effort
  routing, absolute time and global polynomial basis. Preserve source coefficients
  directly; record any coefficient roundtrip difference instead of claiming identity.

All motion_matching paths above are under src/shared/python; Pinocchio paths are
under src/engines/physics_engines. Read provider contracts and existing tests first.

## Stage 1: Two-Window Fixture

After fit73 is terminal, select its cleanly returned candidate if available;
otherwise retain the last clean candidate recorded by the main handoff. Freeze its
exact bytes, original model/capture hashes, runtime and source. Replay once from
the original initial state. Save actual q/qd samples, markers and exact clock.
Run72's state_jacobian is a sensitivity array, not a saved state trajectory.

Use the actual0.6 s capture sample as the sole interior boundary if present,
ending at the candidate's current0.85 s horizon. Use sampled_shooting_windows;
never interpolate or retime the capture. Initialize the interior node from the
uninterrupted state and verify closure and velocity constraints independently.
Keep the initial node fixed and preserve absolute-time polynomial evaluation in
both windows. The zero-displacement segmented baseline must reproduce the
uninterrupted baseline and existing physical continuity gates before optimization.

## Stage 2: Derivative Preflight

Write failing tests for window endpoint and marker derivatives, node retraction
chain rules and the negative next-node term in continuity. Then implement only
the missing orchestration using existing providers. Center the existing full q/v
closure retraction on the integrated interior state, with zero node displacement.
Record physical coordinate/velocity scales and chart-radius guards explicitly.

Compare analytic and central-replay directions for LSInputX B6, one node-position
tangent, one node-velocity tangent and one mixed control/node direction. Use
multiple perturbation sizes, preserving exact model/time/basis. Include marker
residuals and endpoint/continuity defects; a passed marker-only derivative is
insufficient. Reuse existing derivative tolerances and report relative plus
absolute discrepancies. Record integration-error floors when a weak direction
cannot be resolved; do not widen thresholds to manufacture a pass.

Archive exact drivers, inputs, raw outputs, settings, budgets, failure/terminal
receipts and hashes. Do not optimize if this preflight fails. Diagnose the failed
chain at the same fixture rather than adding more fitting iterations.

## Stage 3: Bounded Existing SLSQP Trial

After preflight passes, keep fit73's control subset, objective and physical model
for one bounded trial to isolate the formulation change. Use independent scaled
continuity equalities and retain full physical-defect checks. Chart bounds protect
local retraction validity; they must not become permanent tight boxes around
static target poses. Recenter only after an independently accepted step, retaining
an audit of physical node displacement, model/branch identity and objective change.

Report projected rank, active control/node bounds, predicted versus actual
reduction and original-state replay metrics. Earlier run19 had126 active bounds;
run38's52.5 mm segmented terminal error became201.2 mm uninterrupted error with
89.22 continuity defect. Do not repeat those acceptance mistakes.

Require useful uninterrupted improvement before adding nodes, changing control
scales or unlocking coefficients. Changes to one factor need their own receipt.
Final acceptance still requires the entire1.813888889 s/654-frame capture,
original initial state, existing marker/closure/effort gates, and R2025b replay.

## Parallel Representation Lane and Handoff Discipline

Versioned native-motion file I/O and shared state/frame maps are implemented;
do not recreate them. Pinocchio's spherical builder has incomplete trajectory
qualification. MuJoCo's spherical exporter has real compiled kinematic/inertia
parity only; continue its tangent/effort/rigid-closure stages in
../mujoco_native_matching/MUJOCO_MANIFOLD_HANDOFF.md. Drake alternate dynamics and
moving-frame acceleration transport remain open. Coordinate ownership before edits.

Use TDD, explicit contracts and shared providers. After each bounded milestone,
update HANDOFF.md and DEVELOPMENT_LOG.md, commit owned files with normal checks,
and push the topic branch. Keep failed evidence and incremental checkpoints.
Never mark the parent goal complete from a short prefix or representation test.
