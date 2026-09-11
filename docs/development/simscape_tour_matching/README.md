# Tour-Average Simscape Forward-Dynamics Matching

R2025b qualification: all 22 native tests pass on DeskComputer in MATLAB 25.2.0.3177638 (R2025b) Update 5. The actual 20 ms forward simulation succeeds with finite joint states and club/grip positions (95.89 s). Source revision: `4191ca3aca126216ea02e26e485ff72908f3ad2b`. Evidence: `native_evidence/native_r2025b_suite.json` and `native_evidence/baseline_r2025b.json`. Subsequent milestones qualify all 27 sensed actuator efforts and a 0.4 s exploratory fit; full-swing fitting remains unqualified. See [Current Handoff](../../../AGENT_HANDOFF.md).

## Required MATLAB Release

MATLAB R2025b is the required execution, model-save and validation release for the Simscape golf model and tour-average matching epic #9921. The user has the complete required licensed feature set in R2025b. R2026a is not a requirement; do not select it from PATH or use its successful probes as R2025b acceptance evidence. On DeskComputer and ControlTower launch `C:/Program Files/MATLAB/R2025b/bin/matlab.exe` explicitly. Preserve historical reports with their actual release; run acceptance checks in R2025b. The full forward-dynamics matching goal remains active with this constraint.

Epic: [#9921](https://github.com/D-sorganization/UpstreamDrift/issues/9921).
Dependency: kinetics handoff #9714; separate reference-model work #9914.
Foundation implementation: #9924.

## Qualification Status

The full swing is not yet qualified. DeskComputer has completed and independently replayed a 0.4 s quadratic forward-dynamics candidate with 3.119775 mm marker RMS. The next 0.5 s prefix is running; see AGENT_HANDOFF.md for the current checkpoint. The local
MATLAB R2024a, R2024b, R2025a and R2025b
directories lack `bin/matlab.exe`; MATLAB is absent from PATH and the Python
engine package is absent. SSH over Tailscale to DeskComputer and ControlTower
has since been verified. Use the explicit R2025b executable on either host.
Earlier R2026a probes are historical evidence only. See [Remote Execution](REMOTE_EXECUTION.md)
for the machine-transfer handoff and current remote state.

## Measured Capture Audit

`data/C3D_TA_Driver.c3d` contains 654 frames at 360 Hz, covering
0 through 653/360 = 1.813888889 s. POINT units are metres; there are 38 labels
and no EVENT group. Phase boundaries must be explicitly identified and saved;
do not describe heuristic events as recorded measurements.

Residual validity and finite XYZ jointly determine usable observations.
`RShoulderTop` has only 128 valid samples; each of the three `Marker_2:2:*`
markers has 618, and each `Marker_3:3:*` has 633. Most anatomical markers
have 654. `Marker_0:0:0` has 649 valid samples and substantial displacement;
the existing loader's stuck-sentinel comment is not true of this capture.
Its anatomical/club meaning remains unqualified. Unnamed markers and cluster
markers must not silently become clubhead, grip or joint centres.

Y is the apparent vertical capture coordinate; retain the source convention
until anatomical checks qualify the existing right-handed Y-up to Z-up map.
Joint centres differ from skin markers. Fit fixed, body-local offsets or use
qualified reference-model offsets rather than equating them.

## Native Model Contract

The actual `HexPolyInputFunction.m` evaluates
`A*t^6 + B*t^5 + C*t^4 + D*t^3 + E*t^2 + F*t + G` in seconds.
The shared Python evaluator takes ascending powers. Convert explicitly at the
Simscape boundary; reversing names without adjusting the time scale is wrong.
`ModelingMode == 3` selects the polynomial path in the archived SLX charts.
Actual joint actuation settings and motion-source modes still require a live
model inspection: setting a workspace selector alone does not prove dynamic
actuation. Preserve documented damping and record its contribution separately.

`simulate_with_coefficients.m` is the existing forward-call authority.
Its output extractor can synthesize a time grid and fill unavailable joint
signals with NaNs, while the wrapper allows a success status. A matching run
must independently require actual time coverage and finite simulated signals.
Physical-clock resampling is now tested: logs retain their own timestamps,
numeric arrays require matching solver timestamps, and uncovered samples remain
NaN. Native joint logs are degrees; direct joint-bus extraction now converts
them to the canonical radian contract and preserves translation coordinates in
metres. The actual baseline has finite q/qd. The #9927 referenced-joint target and
priority repairs pass native tests and actual model execution; actuator effort
and full marker-frame qualification remain open.
See [native evidence](native_evidence/README.md).
Its current canonical output does not expose all anatomical marker transforms.
Extend the existing extraction path before claiming a full-body fit.

## Incremental Identification Protocol

1. Hash capture, SLX dependencies, input MAT and marker map. Record source
   revision, MATLAB/toolbox versions and solver settings in every run.
2. Qualify registration and marker correspondence. Estimate a feasible initial
   state and constant geometry; establish marker uncertainty and the geometric
   residual floor. Save bounds and a numerical acceptance target before fitting.
3. Start with the initial short prefix. Optimize torque coefficients and only
   identified geometry parameters within explicit bounds. Normalize polynomial
   time using one fixed full-capture duration, never the changing prefix length.
4. Extend the prefix using the previous parameters as a warm start. Reintegrate
   every prefix from the same initial state and include all earlier observations
   in its residual. Do not reset a state to mocap at a window boundary.
5. Refit earlier coefficients when necessary and measure prior-prefix regression.
   If a single sixth-order polynomial is inadequate, use explicitly C1-matched
   piecewise polynomials and qualify their native Simscape implementation.
6. Independently replay the complete torque input from the saved initial state,
   with tighter solver tolerances and no kinematic prescription. Report unweighted
   marker-distance RMSE, p95, maxima and per-marker/per-phase errors alongside the
   weighted optimization objective. Record continuity, torque, velocity and joint
   constraints, optimizer failures, and the remaining model mismatch.

## Scope Limits

The model's pelvis-rooted upper-body chain cannot represent every measured leg
or head motion. A marker must either have a qualified model attachment, be
explicitly diagnostic-only, or require a tracked topology extension. Fixed-length
rigid segments cannot exactly follow all skin-marker deformation or inconsistencies
introduced by averaging captures. Numerical optimization cannot establish a
global best fit; use repeatable restarts and report attained error and sensitivity.

## Reproduction and Current Tests

From the repository root, with the pinned Tools submodule initialized:

```powershell
python3 -m src.shared.python.motion_matching.capture_audit data/C3D_TA_Driver.c3d driver_audit.json
python3 -m src.shared.python.motion_matching.capture_audit data/C3D_TA_Iron.c3d iron_audit.json
python3 -m pytest tests/motion_matching/test_prefix_fit.py tests/motion_matching/test_capture_audit.py -q
```

Output filenames must be new. Checked-in audit JSON files retain the input hashes
and source revision. They are capture evidence, not optimization results.
The ingestion authority is `motion_pipeline.sources.C3DAdapter`; no new parser,
gap filler or reference-model implementation is introduced.

`prefix_fit.fit_prefixes` accepts a forward oracle returning mapped marker
positions at the requested times. Supply a fixed initial state in the oracle,
constant geometry in the parameter vector, finite bounds and a full-duration
prefix schedule. The optional checkpoint callback receives independent,
read-only parameter snapshots and physical-distance errors at each stage.
The numerical acceptance flag covers final RMSE and optimizer convergence only.
Checkpoint persistence/resume, physical constraints, regularization, native
marker extraction and the live MATLAB oracle are subsequent implementation work.

Initial TDD evidence: missing-module/symbol failures were observed before
implementing prefix fitting, Bernstein conversion and capture auditing. Tests
cover an analytically integrated unit mass under a constant force, masked target
observations, bounds, unattainable targets, missing simulation output, immutable
checkpoints, native coefficient order, torque bounds and synthetic C3D residuals.
These tests do not substitute for live MATLAB validation.

## Scientific Basis

The model must provide joint torques while computing motion dynamically;
see MathWorks' [joint actuation contract](https://www.mathworks.com/help/sm/ug/joint-actuation.html).
The native power polynomial is optimized through degree-six Bernstein controls
over one fixed full-capture interval. Their nonnegative partition of unity gives
a continuous-interval bound from the control values; see the
[B-form convex hull property](https://www.mathworks.com/help/curvefit/construct-and-work-with-the-b-form.html).
Converting to native power coefficients preserves the curve, but large converted
coefficients require numerical conditioning checks in the actual solver. The
bound applies inside the configured interval only.

## Capture Replay Configuration

Use `opts = capture_fit_sim_options(duration_s)` before adding the fixed
geometry and initial-state fields to `opts.input_overrides`. This reuses the
canonical default options and existing forward wrapper, disables cache and
FastRestart for independent qualification, and keeps the torque gate at one
before and after its step time. It does not change saved model defaults.
Three native R2025b tests pass, including a 1.81 s replay across the former
one-second cutoff. Geometry/initial-state calibration and physical bounds
must still be supplied by the fit manifest.

## Qualified Native Body Frames

`[ks, schema] = build_golf_kinematics()` constructs a snapshot of the loaded
model geometry. It resolves coordinate IDs by block path and primitive and
registers 15 world-relative sensor frames. Add targets, guesses and outputs
explicitly using `schema.q_ids` and `schema.frame_ids`; positions use metres
and angles use radians. Load the model and its runtime helper paths first.
A fresh native forward replay verifies all 15 frames at three times to 1e-8 m;
saved-baseline discrepancies were at most 2.054e-13 m. This solver is for pose
calibration and seeding only: its solution does not validate contacts, joint
limits, actuator efforts or dynamics.
