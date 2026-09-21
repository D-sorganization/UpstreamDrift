# Copy-Ready Prompt: Full-Body Models With Lower Limbs and Ground Contact

Work epic #10062 from a fresh topic branch off the branch that carries
docs/development/full_body_models/EPIC_FULL_BODY_CONTACT.md (currently
docs/10003-opensim-matching-epic; main after its PR). Read AGENTS.md,
CLAUDE.md, the epic design document, the native lane's
docs/development/HANDOFF.md (branch feat/9967-native-simscape-pinocchio) and
src/shared/python/motion_matching/tour_capture_contract.py with its tests.
FB-1 (#10063) and FB-2 (#10064) are DONE on this branch (see HANDOFF.md: spec
full_body_spec_v1.json, contact_law.py, 19 tests). Lease the child issue you
take (#10065 to #10067 per engine first; #10068 calibration; #10069 fitting;
#10070 parity) and
register presence before editing. Update docs/development/HANDOFF.md and the
DL-#10062 entry in every implementation commit.

Rules that every child follows (the epic body repeats them):

- TDD: the failing test is committed with the implementation.
- DbC: validate every public input; document postconditions.
- LoD: adapters consume spec objects and arrays only; no reaching into
  another module's engine internals.
- DRY: capture loading (tour_capture_contract), marker inventory
  (MARKER_SEGMENTS), contact parameters (FB-2 module) and the calibration
  algorithm (opensim/python/tour_matching/marker_calibration.py) exist once;
  import them. Move a module to src/shared only with a re-export and tests.
- Never edit the qualified upper-body specification
  (native_geometry_spec_9967.json), its sidecar, MJCF or URDF. Full-body
  artefacts are new documents with their own hashes and names.
- Every numerical run archives inputs, hashes, environment, logs, outputs and
  a HANDOFF under an evidence folder; failed runs stay archived.
- Report kinematic results as milestones; acceptance is the uninterrupted
  forward replay over all 654 frames with the shared gates and the five
  shared metrics (whole, early, terminal, club, pelvis yaw).

Order of work and done gates:

1. FB-1 and FB-2: done; consume full_body_spec.load_full_body_spec and
   contact_law.sphere_ground_contact; do not re-derive either.
2. (reserved)
3. FB-3 (#10065/#10066/#10067): per-engine builders and adapters; upper-body
   slice parity to 1e-12; contact parity on the harness; same-input replay
   against Pinocchio with convergence. Run real-engine tests on ControlTower
   (Pinocchio venv /home/dieterolson/simscape-pinocchio-9967, Drake runtime
   per the Drake handoff) or locally for MuJoCo 3.3.4; archive receipts.
4. FB-4 (#10068): per-engine marker calibration and IK using the shared
   alternating algorithm; offsets bound to the spec hash; per-frame RMS;
   overlay animation.
5. FB-5 (#10069, Claude): full-body two-window shooting fit with contact.
6. FB-6 (#10070): cross-engine parity and visual review.

Compute conventions: ssh alias controltower, WSL distro ControlTower-Runner,
launch long jobs with systemd-run --user through a launcher that records pid
and exit code (see the native lane's run_job.py pattern); ssh-spawned
background processes die at logout. One BLAS/OpenMP thread for comparable
timings. Never overwrite a run directory.
