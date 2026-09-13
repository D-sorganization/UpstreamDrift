# Copy-Ready Prompt: OpenSim Tour-Average Matching, Next Bounded Tasks

Work epic #10003 (OpenSim full-body match to the tour-average C3D) from
worktree C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003,
branch docs/10003-opensim-matching-epic. Read AGENTS.md, CLAUDE.md,
docs/development/opensim_tour_matching/HANDOFF.md, EPIC_10003.md, the OS-1 to
OS-3 receipts under evidence/, and the code in
src/engines/physics_engines/opensim/python/tour_matching/ with its tests in
tests/opensim/. Lease #10003 (Repository_Management scripts.check_agent_claim
and post_agent_lease with your real agent identity), register presence, and
update HANDOFF.md plus the DL-#10003 entry in every implementation commit.

Rules: write the failing test first and commit it with the implementation;
validate every public input and document postconditions; take spec objects
and arrays, do not reach into other modules' engine internals; reuse
tour_capture_contract, marker_map, trc, marker_set and marker_calibration
instead of re-reading the C3D, re-mapping markers or re-implementing the
alternation. Every ControlTower run archives receipt, log, driver hash, inputs
and outputs under evidence/<run>/ and is described in HANDOFF.md; failed runs
stay archived. Never claim a match from IK; acceptance is an uninterrupted
forward-dynamics replay with the shared gates and R2025b cross-checks.

Runtime: ControlTower venv /home/dieterolson/opensim-10003 (OpenSim 4.6,
Moco), bundle /home/dieterolson/opensim-10003-runtime, launch with
systemd-run --user (see HANDOFF.md). Keep the Pinocchio venv untouched.

Task OS-2b (golf model variant, small):

1. RED test in tests/unit/scripts (or tests/opensim) that the built model
   has no locked arm/lumbar coordinates and that clamp ranges cover a golf
   swing (arm flexion at least -120 to 180 deg, lumbar rotation +-120 deg,
   wrist deviation +-45 deg; record your source for each range).
2. Extend scripts/build_humanoid_osim.py with a golf-variant step that
   reuses tour_matching.marker_set.unlock_coordinates and sets ranges; set
   the club length from the capture's club marker cluster (Marker_2/Marker_3
   centroid distance in frame 0) and record it; regenerate golf_humanoid.osim
   and update its README provenance and SHA256 in the handoff.
3. Rerun os3_calibrate_ik.py on the regenerated model (stride 20, 4
   iterations) and archive as evidence/os3_variant_stride20/.

Task OS-3b (scaling and full IK, medium):

1. RED tests for a pure-Python segment-length estimator: pairwise marker
   distances on frame 0 for femur (knee-hip proxy via waist), tibia
   (knee-ankle), foot (ankle-toe), humerus (shoulder-elbow), forearm
   (elbow-wrist), torso (waist-shoulder); output scale factors per body with
   provenance and the residual of the rigid assumption over the first 20
   frames.
2. Apply scales through OpenSim ScaleTool (or the model API) in a driver on
   ControlTower; archive the scaled model with hash.
3. Modify marker_calibration.calibrate_marker_offsets to return the best
   iteration (test first) and add per-marker RMS to CalibrationResult.
4. Run IK on all 654 frames (stride 1); archive q trajectory (.mot and npz),
   per-frame RMS, per-marker RMS, and an overlay animation GIF named with the
   model hash; report the five shared metrics computed from marker positions
   (whole, early <=0.6 s, terminal, club, pelvis yaw).

Task OS-4 (Moco tracking pilot, medium, after OS-3b):

1. RED test for the tracking configuration builder (pure Python: problem
   bounds, marker weights, mesh interval, actuator limits, from a config
   object with validation).
2. MocoTrack over 0 to 0.85 s with coordinate actuators only, initial guess
   from the OS-3b IK, on ControlTower with a bounded time budget; archive the
   solution, solver log and a receipt; report marker RMS and effort.
3. Replay the Moco controls forward without state resets and compare the
   uninterrupted replay to the Moco states; this is the honesty check.

Task OS-5 (global degree-six efforts, hard, coordinate with the Claude lane):
fit one degree-six polynomial per coordinate actuator over the full capture
by forward shooting from the original state, using the shared
multi_shooting_fit providers where possible; acceptance by the shared gates.

Report every result on the five shared metrics, never by optimiser cost, and
state the head-marker and club-attachment limitations in every receipt.
