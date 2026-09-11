# Incremental Checkpoints and Replay

This is the continuation guide for epic #9921. The current implementation is
SELF: resolve it with `git rev-parse HEAD` in the checkout containing this guide.
The canonical state remains [AGENT_HANDOFF.md](../../../AGENT_HANDOFF.md).

## Current Milestone and Resume Entry Point

The verified 0.4 s quadratic fit has 3.119775 mm RMS with exact independent cold replay and an explicit successful 22-channel legacy audit. Its 148-file archive and both native exit-zero receipts are recorded in native_evidence/prefix_400ms_bundle_receipt.json. A 0.5 s quadratic fit is now running on the promoted 27-channel model at 0715f95f3. Follow AGENT_HANDOFF.md for live handles and source identity; the sections below retain earlier experiments. Do not treat their next-step prose as current instructions. New saved-audit calls require 27 channels by default; pass expected_channels=22 only for historical raw logs.

## Verified 0.2-Second Linear Candidate

The completed 73-sample, 25-marker fit has 0.859647 mm RMS, 1.873274 mm p95 and
3.575221 mm maximum error. The same t0 state and fixed geometry/attachments are
used throughout. Its first 0.1 s remains at 0.367566 mm RMS. Independent cold
replay has zero marker difference; all sixteen logged torque profiles match the
requested linear polynomials exactly. Eleven actuator coordinates remain unlogged.
Optimizer convergence is false because the evaluation budget was reached.

The verified 104-file prefix-200ms-linear-01-bundle.zip is available in both
machine checkpoint locations. SHA256 is
32a491e2a14631c950ae4633edbad860fe917c90954cfd6839fa1e7f046bdfba.
See native_evidence/prefix_200ms_linear_bundle_receipt.json for exact paths and
per-file hashes. Use qualified_candidate_replay(repo,run_dir) for a cold replay,
then audit_saved_golf_candidate(repo,run_dir) to check the available actuator logs.

Next transfer this completed report into a fresh 0.3 s linear run. The extrapolated
candidate passes the current endpoint bounds. Full-swing and anatomical qualification
remain open; retain the earlier short-prefix archives as distinct experiments.

## Transferring a Candidate to a Longer Prefix

Use --transfer-report <completed-report.json> with --basis linear --duration 0.2
in a fresh run directory. This explicitly transfers only the final torque candidate,
with a source-report hash and a new evaluation history. --checkpoint remains for
exact-identity warm starts and cannot be combined with --transfer-report.

Constant controls are duplicated into equal linear endpoints. Later linear transfers
preserve the physical slope when the prefix grows; if the new endpoint exceeds the
existing effort bounds, the transfer fails rather than clipping. Native q/qd,
geometry, body offsets, marker/joint order and joint effort scales must be unchanged.
Linear endpoint bounds apply over [0, prefix end]; the A–G coefficients retain t0=0.
Every forward evaluation replays from the same initial state. Forty-seven tests
cover the state, candidate transfer, basis conversion and shared prefix optimizer.

## Qualified-State 0.1-Second Candidate

The new state/geometry fit covers 37 samples and 25 markers over 0–0.1 s.
RMS is 0.356491 mm, p95 0.827594 mm and maximum 1.591758 mm. A fresh R2025b
cold replay exactly reproduces every marker coordinate and the initial q/qd.
The optimizer hit its evaluation limit; accepted_numerically remains false.
This is a short-prefix experiment with provisional geometry and fixed t0 offsets.

The complete archive, including 77 hashed files, is prefix-100ms-01-bundle.zip
under the local simscape-tour-checkpoints and remote SimscapeTour9921 directories.
Its SHA256 is e673a2a9db8772c3718ae8cefaa049b224a5830f3599cae43ad7b5a910c1fa1a.
See native_evidence/prefix_100ms_bundle_receipt.json for exact paths and provenance.
The raw final_native_replay.mat preserves the exact polynomial coefficients.

For an independent replay, extract the bundle without modifying the original,
add native_evidence/reproduction to R2025b's path and call
qualified_candidate_replay(repo, run_dir). It reads the saved report, raw candidate
and capture payload, disables Fast Restart, and asserts marker/state parity.
To regenerate the plot, run python native_evidence/reproduction/plot_prefix_100ms.py
with the extracted run directory as its positional argument.

Next grow to 0.2 s with an explicitly transferred candidate and continuous
nonconstant polynomial efforts. Exact checkpoint resume rejects a changed horizon;
a new objective requires a fresh report, not inherited evaluation history.

## Resume Identity

Reports may declare a fit_identity manifest containing ordered native coordinates,
initial q/qd, geometry, fixed body offsets, basis and horizon. The checkpoint
reader requires exact equality whenever either side declares this manifest;
it rejects omission as well as a changed value. Legacy snapshots can still
resume against legacy reports. A changed-state or longer-horizon experiment
must begin a fresh fit rather than silently reusing old objective history.
The runner populates this manifest from a qualified state and verifies the native
initial q/qd and markers before optimization. Ten state-contract tests and four
checkpoint tests pass, including the red-to-green identity mismatch regression.

## Saved First-Prefix Candidate

The 25-marker R2025b fit covers only 0–0.05 s. Its final Euclidean marker RMS is
0.176671 mm, p95 0.280906 mm, and maximum 0.594141 mm. It uses fixed offsets
calibrated from the initial frame, default lengths, zero initial joint velocities,
and constant polynomial efforts. The optimizer reached its evaluation limit;
`accepted_numerically` is false. This is a useful warm start, not a qualified tour
swing. Preserve that distinction when quoting the result.

- Result: `native_evidence/first_prefix_fit_r2025b.json`.
- Plot: `native_evidence/first_prefix_fit_r2025b.png`.
- Initial pose: `native_evidence/native_pose_seed_waist_r2025b.json`.
- Local checkpoint directory:
  `C:/Users/diete/Repositories/simscape-tour-checkpoints/first-prefix-20260910`.
- DeskComputer checkpoint directory:
  `C:/Users/diete/SimscapeTour9921/checkpoints/first-prefix-25`.
- Saved snapshot:
  `evaluation-00283-56e0ff8fb6fdfba07912fa7dbf3fb70197da89899d4c85955865726f8303dcab.json`.

Snapshots contain the complete report, its SHA256, the best evaluation and the
corresponding optimizer parameters. A repeated report is a no-op; a new report
creates a new file. Partial reports are rejected and prior snapshots are never
overwritten. This resumes a candidate, not SciPy's internal trust-region state.

## Independent Cold Replay

A fresh R2025b process reproduced every saved marker coordinate exactly (maximum
absolute difference 0 m), with initial q error 1.410e-9 and Fast Restart off.
MATLAB exited zero. Evidence: `native_evidence/cold_candidate_replay_r2025b.json`.
This qualifies repeatability of the 50 ms candidate only.

To repeat, copy `native_evidence/reproduction/cold_candidate_input.json` into a
fresh output directory, add the reproduction directory to MATLAB's path, and run:

```matlab
cold_candidate_replay(repo, run_dir)
```

Pass absolute repository and output-directory paths. Use the explicit R2025b
executable with `-batch`; the function asserts the release, loads the model,
uses the authoritative forward wrapper via the native marker adapter, and saves
JSON plus a raw MAT replay before asserting parity and initial-state tolerances.
Its fixed input preserves the original coefficients and body offsets; it does
not recalibrate against the replay. The original raw MAT is preserved on
DeskComputer at `C:/Users/diete/SimscapeTour9921/cold_candidate_replay.mat`.

## Running or Resuming on DeskComputer

Use the isolated interpreter at
`C:/Users/diete/SimscapeTour9921/python-r2025b/Scripts/python.exe`.
It uses Python 3.12 and the engine installed from the local R2025b installation;
the global R2024b engine was not changed. MATLAB must report `2025b` and root
`C:/Program Files/MATLAB/R2025b`. Do not use the default MATLAB on PATH.

Start from a checkout containing this implementation, with its helper paths,
and a fresh run directory containing `driver_marker_payload.json` and
the versioned `initial_velocity_seed_qualified_r2025b.json` state. Keep the original result
directory intact. The replay script is
`native_evidence/reproduction/first_prefix_fit.py` relative to this guide.

```powershell
$fitRepo = 'C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime'
$fitPython = 'C:/Users/diete/SimscapeTour9921/python-r2025b/Scripts/python.exe'
$fitRun = 'C:/Users/diete/SimscapeTour9921/resumed-first-prefix'
$fitState = "$fitRepo/docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json"
$fitScript = "$fitRepo/docs/development/simscape_tour_matching/native_evidence/reproduction/first_prefix_fit.py"
& $fitPython $fitScript --repo $fitRepo --run-dir $fitRun --initial-state $fitState --duration 0.1 --max-nfev 10
```

The script requires a qualified initial state and an exact capture-sample horizon.
To resume, add --checkpoint with a snapshot from the same state and horizon. It
checks capture hash, marker order, effort scales and fit identity before accepting
a checkpoint. The original default-state 50 ms candidate is incompatible with the
new state and must remain a separate archived experiment. It saves the baseline, every five evaluations, each completed
stage and the final result. Immutable snapshots go to the new run directory's
`checkpoints/first-prefix` folder. Omit `--checkpoint` for the zero-effort start.
The report's effort scales map parameters by `effort = scale * (parameter - 1)`;
translation efforts are N and angular efforts are Nm.

To snapshot an already running report without changing its process:

```powershell
& $fitPython "$fitRepo/src/engines/Simscape_Multibody_Models/python/tour_checkpoints.py" "$fitRun/first_prefix_fit.json" "$fitRun/checkpoints/first-prefix"
```

Before starting anything, inspect the existing Python/MATLAB processes and their
logs. An observation timeout is not a stopped process. Do not launch duplicate
fits or change a running process's source checkout. For remote launches, retain
the SSH session with `Start-Process -WindowStyle Hidden -Wait`; a detached child
may be terminated when its SSH session closes.

## Source and Data Breadcrumbs

The initial experiment ran on native source base
`0d8b3400421c8c71807356eba4d9b7cdb273c33f` with the marker adapter and finite-
difference changes recorded in this implementation. The checkpoint bundle has a
SHA256 manifest, exact original run script, enhanced replay script, source patches,
pose seed, registered marker payload and result. It excludes licenses, environments
and model binaries; restore the model from the recorded Git revision.

The original C3D hash is
`545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba`.
The payload retains all 654 timestamps, source labels, validity flags and the
explicit source-to-world rotation. Invalid values are represented by zero plus a
false validity flag; the fit restores them to NaN before constructing MarkerTarget.
For another C3D, rerun the canonical C3DAdapter/capture audit and establish its
registration, pose and attachments. Do not relabel this target or reuse its offsets
as if they were a new calibration.

## Geometry Sweep Checkpoint

The completed nine-candidate sweep is saved in `native_evidence/native_geometry_seed_sweep_r2025b.json`.
Its verified two-machine archive and source/data hashes are recorded in
`native_evidence/geometry_sweep_bundle_receipt.json`. Extract the archive, retain
the original directory, and place its two pose/target input JSON files into a
fresh run directory. Add the reproduction directory to the R2025b MATLAB path
and run `native_geometry_seed_sweep(repo, run_dir)` with a checkout containing
`fit_golf_pose_seed.m`. Every candidate writes a numbered checkpoint. Existing
checkpoint names cause an error rather than replacement.

Best tested starting-pose proxy RMS is 49.343 mm at (14.5,12) inches versus
54.471 mm at default (12,14). The boundary result is a seed only; anatomical
identification, multiple-frame checks and changed-geometry dynamics remain open.

## Changed-Geometry Replay

The best tested geometry seed passed a 50 ms zero-effort native forward replay
in R2025b. Evidence is `native_evidence/geometry_candidate_replay_r2025b.json`;
run `geometry_candidate_replay(repo, run_dir)` after adding the reproduction
directory to MATLAB's path. The probe uses the versioned geometry-sweep result,
applies identical geometry to KinematicsSolver and the forward wrapper, and
checks initial q, 15 directly logged frames, and actual arm lengths. It restores
workspace values without saving the model. This is geometry/state qualification,
not a fitted swing. Raw MAT evidence remains in DeskComputer scratch.

## Unqualified Initial-Velocity Seed

The native velocity map passes its pose-difference test, but the nonzero-velocity
forward initialization currently fails because torso and wrist velocity targets
are disabled. Preserve `initial_velocity_seed_unqualified_r2025b.json` and
`initial_velocity_mismatch_audit.json` as the failing regression. The raw MAT
has a verified local copy; its paths and SHA256 are recorded in the audit.
Run `initial_velocity_seed(repo, run_dir)` after adding the reproduction directory
to R2025b's path, with a fresh directory containing `driver_marker_payload.json`.
The probe saves its candidate before replay and its actual state before asserting
acceptance. A report marked `replayed` is not proof that those assertions passed.

Next implement per-simulation independent velocity priority controls in the sole
forward wrapper, verify restoration of model defaults, and pass the saved qd
regression before using this seed for fitting. See the current handoff for exact
missing rates and mask selectors. Preserve the earlier verified torque candidate.

## Qualified Initial State and Session Lifetime

The latest state is `native_evidence/initial_velocity_seed_qualified_r2025b.json`.
It has `initial_state_verified=true`; retain earlier unqualified files as failure
evidence. `qualified_velocity_bundle_receipt.json` records the verified two-machine
archive, inputs, raw replay and source hashes.

After loading the native model, create
`priority_cleanup = configure_capture_velocity_targets()` and keep it alive
across optimizer replays. Clear it on completion; it releases Fast Restart and
restores original target settings. Automatic cleanup also handles exceptions.
Do not apply transient mask priority overrides through SimulationInput on each
replay: that rejected prototype failed restoration during Fast Restart.

The first-prefix reproduction runner now owns this guard, but still uses its
historical default geometry and zero initial velocities. Extend its state and
horizon inputs before using the newly qualified state; a torque checkpoint alone
does not replace q, qd, geometry or attachments. Each changed-state run needs its
own state identity and independent acceptance evidence.

## Ordered Continuation

1. Preserve and inspect this candidate and its limitations before changing geometry.
2. Active length bindings are qualified by `native_evidence/native_geometry_probe_r2025b.json` and its reproduction probe: both upper arms use `UpperArmLength` in inches;
   both forearms use two halves of `LowerArmLength` in inches. The left/right
   upper-arm aliases do not drive these native solids.
3. Calibrate constant lengths and bounded anatomical marker attachments, rebuilding
   KinematicsSolver after geometry changes. Estimate a consistent initial velocity.
4. Warm-start and extend prefixes from the same time-zero state, progressively
   allowing nonconstant continuous polynomial efforts. Retain all earlier samples.
5. Validate fresh forward replays, all actuator channels, solver refinement and the
   entire capture before claiming a representative tour swing.

The native adapter must project original logged joint states before interpolating
marker positions. Interpolated joint angles can violate the closed chain; the
preserved failed experiment demonstrates this. Do not replace failures with a
finite penalty or weaken constraint acceptance without quantified evidence.
