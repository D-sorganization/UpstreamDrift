# Native Port Implementation Checkpoint

## Analytic Joint-Torque Refinement Is Running

Updated 2026-09-12 UTC. Previous goal turn made progress by independently
replaying the improved root-force result and qualifying one sensitivity direction.
This turn qualifies and connects the 81-control block to the shared optimizer.
Current live run: native-analytic-fit-9967-01 on ControlTower, unified session
12513, WSL PID 1954758. First optimizer trial reduced terminal RMS from 84.0761
to 75.9347 mm (whole 28.3953 mm, early 10.7753 mm, club 47.7935 mm). These are
live trial metrics, not a terminal/independently accepted result.

NativeEffortProfile.bernstein_control_jacobian reuses the shared Bernstein
conversion and includes world-to-base force rotation. Its columns are native
coordinate-major, ascending selected Bernstein index. Six tests were red then
green for rotation, force/torque separation, ordering and invalid basis inputs.
Actual 81-column audit at run 03 with single-thread BLAS took 12.4247 s.
Primal marker difference was 1.16159e-8 m. Selected independent checks: world-Y
B4 relative error 1.474e-6; LSInputX B4 8.046e-6 at 3e-5 Nm; RWInputX B4
1.928e-5 at 3e-5 Nm. Both torque checks also passed at 1e-4 Nm.

The first block audit (native-trajectory-sensitivity-9967-02.json) deliberately
remains failed: its wrist finite-difference check at 1e-6 Nm differed by 0.226%.
Intermediate physical probes and tighter sensitivity tolerance passed in
native-trajectory-sensitivity-9967-03.json. Do not hide the failure or claim every
column was independently finite-differenced. Full arrays are in same-name NPZs
on ControlTower and in local simscape-tour-checkpoints; receipts hash each NPZ.
Runtime for both: /home/dieterolson/native-sensitivity-9967-02. Audit 03 used
the separate copied check_native_trajectory_sensitivity_9967_03.py, whose actual
hash is in the receipt, with --all-controls --rtol 1e-10 --atol 1e-12,
--probe-coordinate LSInputX --probe-coordinate RWInputX --probe-step 0.0001
--probe-step 0.00003. OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 were explicit.

New native_sensitivity.replay_marker_sensitivities is the reusable adapter.
It validates ordinary continuous replay, integrates the full selected block,
checks sampled closure and agreement of primal markers, then returns physical
N/Nm derivatives. A two-kilogram free-mass analytic sextic response test went
red then green. Native adapter audit exited zero (session 11670): every column
matches audited block 03 exactly, total 15.9810 s including independent replay,
sensitivity 12.7817 s. Receipt native-sensitivity-adapter-9967-01.json.

Shared PrefixFitOptions now accepts optional marker_jacobian; it applies the
same missing-marker, marker-weight, time-weight and terminal-weight operations
as its residual. Independent central differences of the actual residual validate
this composition. Active yaw penalties or arbitrary regularization explicitly
reject this new callback until their derivatives are implemented; existing
finite-difference execution is preserved. The current native objective has
neither active penalty. Existing positional option order is preserved.
Combined adapter, effort-basis, control-subspace and shared-prefix tests:
53 passed. Direct mypy for native adapter, effort profile and shared fitter passed.

Current runner --analytic-jacobian caches by canonical candidate plus full clock
hash and writes jacobians.jsonl separately. Each uncached Jacobian entails one
additional primal replay and one sensitivity integration; count those as well
as evaluations.jsonl when comparing cost. First full-clock Jacobian cost 16.5464 s
and agreed with primal markers within 3.42e-9 m. No feedback or interior resets.

Run command uses the usual Python venv and model/target paths below, candidate
/mnt/c/Users/diete/native-root-force-9967-02/returned-candidate.json,
--shaping bernstein456 --amplitude-scale 10 --analytic-jacobian --max-nfev 8,
and --output_dir /mnt/c/Users/diete/native-analytic-fit-9967-01. All 81 force/torque
controls are active; corrections are +/-2 N/Nm relative to the root-force return.
Original early, whole, terminal, club and convergence gates remain unchanged.
Dedicated runtime /home/dieterolson/native-analytic-refinement-9967-02;
source bundle native-analytic-refinement-bundle-9967-02.zip SHA-256
148846240f77e2f501476644ba6c31d26da0229bca7803aeef840eda64327832.
Bundle 01 (adapter audit) SHA-256
5e7a9ca98aa94efea484d44b3ae433cb23bc97c1587a5eb7f3e4325be3dc9b80;
bundle 02 only reorders the new optional field to preserve positional API and
clarifies its documentation. All bundles carry exact per-file hashes and are
archived locally and remotely. Namespace-only deployment remains the scope.
Next: inspect session 12513, archive terminal logs including Jacobian ledger,
and independently replay/plot the returned candidate before promotion.

## Root-Force Run 02 Finished and Independently Reproduced

Updated 2026-09-12 UTC. Session 94653 exited zero after 233 actual forward
evaluations. No optimization or audit job from this workstream remains live.
The returned candidate canonical hash is
2afaf8b21a05a44b071e7328e2d624bba5f6a999aa85920b53b61b43961e6673.
A fresh ControlTower process reproduced all four reported marker metrics exactly:

| Metric                         | Run 03  | Root-Force Run 02 |
| ------------------------------ | ------- | ----------------- |
| Whole RMS (mm)                 | 30.8867 | 29.7657           |
| Early RMS Through 0.6 s (mm)   | 10.8177 | 10.9214           |
| Terminal RMS at 0.8 s (mm)     | 94.8294 | 84.0761           |
| Terminal Club Cluster RMS (mm) | 66.0119 | 55.5631           |

The club threshold is now met, but whole and terminal gates fail. The optimizer
exhausted its 24-evaluation budget; convergence and numerical acceptance remain
false. No full-swing or native R2025b acceptance of this candidate is established.
This is the current returned candidate with the best recorded weighted fit score,
not an accepted representative tour swing. Preserve earlier candidates as seeds.

Artifacts: native_evidence/root_force_9967_02/ contains returned-candidate.json,
returned.json, best.json, config.json, all evaluations, independent-replay.json,
visual-replay.npz and marker-errors.png. Original raw copies remain in local
simscape-tour-checkpoints/native-root-force-9967-02 and ControlTower
C:/Users/diete/native-root-force-9967-02. Candidate file byte SHA-256 in the raw
independent receipt is 34357becc44fa891a402a0f68e0a795d8304074bc8cdcfb5f419157b7de2680e;
canonical identity is insensitive to repository JSON formatting.

![Root-Force Run 02 Observed Marker Errors](native_evidence/root_force_9967_02/marker-errors.png)

Independent full adapter/residual evaluation took 3.55486 s, integration 3.40878 s;
sampled closure pose/rate maxima 8.24429e-12 / 2.63179e-10. Same 25 markers,
289 samples and 7225 observations. Plot was rendered and visually checked. The
benchmark receipt retains its historical baseline-only qualification string;
that string does not erase this candidate's actual optimizer history.

Three force controls are near bounds: TranslationInputX B6 +24.9990 N,
TranslationInputY B5 -24.9984 N and B6 -24.9980 N, relative to run 03. This is
evidence that the selected subspace/bounds restrict this local solve, not a reason
to widen every effort indiscriminately. Arms still have the largest individual
terminal errors. All 24 torque profiles were frozen in this experiment.

Next bounded assignment: qualify the full selected control sensitivity block
using the new shared integrator and native derivative methods (details below).
At minimum validate the known world-Y B4 direction and independent torque
directions, compare primal/closure accuracy, then benchmark against the actual
81-column replay cost. One-column integration is accurate but took 32.9 s;
do not assume all-column sensitivity is faster. If useful, add an optional
Jacobian path to shared fit_prefixes with exact residual masks/weights/terminal
terms and cache by full candidate/clock identity. Otherwise use physically
qualified finite differences. Resume joint-torque refinement from this returned
candidate; preserve all raw marker errors and the known head/back rigidity limit.

## Complete Native Sensitivity Direction Verified

Updated 2026-09-12 UTC. Previous goal turn made progress: force-scale audit,
corrected search and committed evidence. This turn implements and verifies the
next sensitivity stages while root-force run 02 remains live (session 94653).
Its latest observed evaluation 201 had terminal RMS 84.3354 mm, whole RMS
29.7553 mm, early RMS 10.9194 mm and club RMS 55.9030 mm. These are evaluated
trial metrics, not a returned/independently accepted candidate.

NativePinocchioModel.marker_derivatives reuses shared project_markers validation
and Pinocchio aligned frame Jacobians, accounting for rotated marker offsets and
native coordinate-column order. Tree partial derivatives are not projected onto
the weld manifold; the dynamics sensitivity supplies that constraint consistency.
Three boundary tests were red then green. Actual native audit on all 25 markers
and 27 columns at 0,0.6,0.8 s passed three central step sizes; maximum discrepancy
at step 1e-6 was 5.84e-10. Receipt native-marker-derivatives-9967-01.json.

New shared forward_sensitivity.integrate_sensitivities integrates x'=f and
S'=df/dx\*S+df/dp through existing integrate_forward. It validates dimensions and
finite values, preserves initial sensitivities, and returns detached readonly
state/sensitivity arrays. Analytic polynomial-acceleration and exponential
state-coupling tests went red then green. Combined candidate, local derivative
and sensitivity tests: 28 passed. Direct mypy on native_model.py and
forward_sensitivity.py passed; ruff passed.

The complete native audit (session 42709, terminal exit zero) integrated one
world-force B4 sensitivity through 0.8 s and compared the actual marker derivative
to the independently recorded small-force central replay. Relative discrepancy
1.46013e-6; primal marker maximum difference 2.39944e-10 m; sampled closure bound
9.37455e-11. Receipt native-trajectory-sensitivity-9967-01.json. This qualifies
ONE direction at run 03, not the full optimizer Jacobian. All inputs and actual
source hashes are preserved. No optimizer uses this new path yet.

Timing: augmented integration 32.9031 s, 112466 RHS evaluations, versus ordinary
replay integration 3.15391 s. Do not claim a speedup for one direction. Next audit
must measure the whole parameter block and physically adequate sensitivity
tolerances, preserve primal/closure accuracy, and compare actual total time with
the numerical Jacobian. Tight error control includes sensitivity variables and
can cost substantially more than a primal replay. Shared fitter residual masks,
weights and terminal terms also need a verified derivative before integration.

Reproduction runners: check_native_marker_derivatives.py and
check_native_trajectory_sensitivity.py in native_evidence/reproduction.
Both require --model and --candidate paths used below plus NEW --output paths;
the trajectory audit additionally requires --reference
/mnt/c/Users/diete/native-force-difference-9967-01.json. Runtimes:
/home/dieterolson/native-marker-derivative-9967-01 and
/home/dieterolson/native-sensitivity-9967-01, respectively. Same Pinocchio venv.
Source bundles archived locally and on ControlTower:
native-marker-derivative-bundle-9967-01.zip SHA-256
a69ad5a2c92aa3b21d4be614c729c49760ded337f0bdbe8fd2af8b49ee4cb99d;
native-sensitivity-bundle-9967-01.zip SHA-256
6ca43493c566e5f963baf5b09ca393ffbf30673cecf284bf4f4b85d646abe05a.
Each contains a per-file source manifest. These remain isolated namespace
deployments, not full application qualification. Next immediate action: observe
root-force run 02's exact handle and independently replay its terminal return.

## Physical Difference Step Was Too Large; Corrected Search Is Live

Updated 2026-09-12 UTC. This supersedes the root-force live status below.
Root-force run 01 (session 81736) exited zero after 23 actual evaluations and
returned the exact run 03 candidate unchanged. It exhausted 12 optimizer
evaluations without an accepted step. Complete artifacts are preserved in
native_evidence/root_force_9967_01/. Do not restart this terminal handle.

The native finite-difference audit exited zero (session 32416). At fixed run 03,
perturbing world force coordinate 1, Bernstein control 4, gives the following
central marker derivative norms at output times 0,0.6,0.7,0.8 s:

| Physical Step (N) | Derivative Norm (m/N) | Relative Change From Previous |
| ----------------- | --------------------- | ----------------------------- |
| 0.125             | 6.24575               | N/A                           |
| 0.0125            | 38.01678              | 1.09295                       |
| 0.00125           | 47.39844              | 0.24194                       |
| 0.000125          | 47.50927              | 0.00355563                    |
| 0.0000125         | 47.51057              | 0.0000386148                  |

The original search step 0.125 N was demonstrably outside this local linear
regime. This is one direction at one candidate; it does not qualify all columns
or prove that corrected steps alone solve matching. See immutable receipt
native-force-difference-9967-01.json and reproduction/check_native_force_difference.py.
It preserves the full derivative arrays, central/even residuals and source hashes.

Root-force run 02 is now live: session 94653, WSL PID 1941356, output
C:/Users/diete/native-root-force-9967-02. Same run 03 base, frozen torques,
nine variables and +/-25 N bounds, but --finite-difference-step 1e-7 gives
0.0000125 N at the initial point. --max-nfev 24. Initial evaluation again
reproduced run 03 exactly; no acceptance claim yet. The runner now exposes the
existing shared fitter's validated step option instead of hardcoding 0.001.
Combined subspace/shared-prefix tests: 38 passed, including existing invalid
difference-step contracts. The earlier control mapping had red-to-green tests.

Dedicated runtime: /home/dieterolson/native-root-force-runtime-9967-02.
Use the same model/candidate/target paths and Python executable below, adding
--finite-difference-step 1e-7 --max-nfev 24 and a NEW output path to reproduce.
Bundle native-root-force-bundle-9967-02.zip, SHA-256
f4b05cda0aaa5d2ff17cfaae0c560a30e3d30c89b6ecc667bc30be4ac77a43d6,
is archived locally and on ControlTower with per-file hashes. It preserves the
same namespace deployment boundary. Next action: inspect session 94653, preserve
terminal results, independently replay any returned improvement, then qualify
all relevant derivative directions before expanding the control search.

## Root-Force Authority Verified and Isolated Search Started

Updated 2026-09-12 UTC. The previous goal turn made progress by committing and
pushing run 03 and refreshing the OpenSim plan. Current native force-authority
audit exited zero on ControlTower (session 44824). See
native_evidence/native-force-authority-9967-01.json and reproduction runner
check_native_force_authority.py. It uses the existing run 03 namespace runtime,
Pinocchio 4.1.0, and the qualified model with mass 77.60581783574678 kg.

For the existing +/-2 N Bernstein controls 4,5,6, maximum terminal COM correction
per world axis is 2*T^2*(3+2+1)/(56\*M) = 1.767172371 mm at T=0.8 s.
Three native replays (baseline, run 03, and simultaneous positive root-force
perturbation) verify the double-integrated force balance to 1.69e-13 m.
Run 03's COM correction also matches its polynomial force integral to 5.53e-14 m.
The static pose diagnostic's terminal COM is displaced from baseline by
(+6.0832,-8.1619,+2.3778) mm. Static COM is neither a measured target nor a necessary
condition for the best marker fit; this supports testing wider force authority,
not claiming that root forces alone explain all marker error.

An isolated root-force search is now live: unified session 81736, WSL PID
1937679; output C:/Users/diete/native-root-force-9967-01 on ControlTower.
It starts exactly from run 03's returned candidate, freezes all 24 torque
polynomials, and changes nine world-force Bernstein controls 4,5,6. Bounds are
dimensionless [0.8,1.2] with scale 125, hence +/-25 N corrections relative to run 03. These are exploratory bounds, not inferred physiological limits. The early
RMS and final acceptance gates remain unchanged. max_nfev=12, relative numerical
Jacobian step 0.001 (0.125 N at the starting point). Initial replay reproduced
run 03's canonical hash and every metric. Early probes show strong nonlinear
sensitivity to 0.125 N perturbations; qualify a smaller physical difference step
before interpreting slow convergence as lack of controllability.

Deployment: /home/dieterolson/native-refinement-9967-04, Python executable
/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python. Source bundle is
native-refinement-bundle-9967-04.zip in local simscape-tour-checkpoints and
ControlTower C:/Users/diete; SHA-256
773f276679f2a785f39f1ba7ace20112eb7061337e2e125c626ab4929e7e789e.
Its manifest hashes every deployed module; full application remains unqualified.
The first attempted output native-refinement-9967-04 already contained another
completed one-step run and was correctly refused; that output was not changed.
Do not confuse it with this newly deployed runtime or the distinct root-force run.

Reproduce the current run by invoking the deployed refine_native_candidate.py
with --model /mnt/c/Users/diete/native_geometry_spec_9967.json,
--candidate /mnt/c/Users/diete/native-refinement-9967-03/returned-candidate.json,
--target /mnt/c/Users/diete/driver_marker_payload_9967.json,
--shaping bernstein456 --root-forces-only --amplitude-scale 125 --max-nfev 12,
and a NEW --output_dir. Never restart merely because an observation times out.

New root-only parameter mapping tests were red (missing function), then green:
23 combined subspace/candidate tests pass. They verify frozen torques and early
controls, historical full-space ordering, invalid scales and wrong dimensions.
Runner uses the same shared fit_prefixes and candidate/Bernstein conversion APIs.
Native force audit includes independent integral and simulator assertions.
Next: inspect this exact live handle, preserve terminal evidence, audit physical
finite-difference step size, and continue native trajectory-sensitivity gates.

## Run 03 Finished: Independent Replay and Typing Check Complete

Updated 2026-09-12 UTC. This supersedes live-run statements below. Session
17132 exited zero after 249 forward evaluations. The returned candidate is
60bbe60ec6a57f7a5c4dd277b15de96fa1760ca006bad6509cb215c3e743e379.
A fresh ControlTower process reproduced every reported marker metric exactly:
whole RMS 30.8867 mm, early RMS 10.8177 mm, terminal RMS 94.8294 mm and terminal
club cluster RMS 66.0119 mm. Whole/terminal/club gates fail; optimizer budget
was exhausted, so numerical acceptance and convergence remain false.
No replacement optimization has been launched in this workstream.

Fresh adapter/residual evaluation took 3.43346 s. Maximum closure pose/rate
residuals were 8.06172e-12 / 1.37587e-10. All run artifacts, source config,
ledger, separate best and returned candidates, independent receipt, portable
arrays and chart are in native_evidence/refinement_9967_03/. Original copies
remain on ControlTower and in local simscape-tour-checkpoints. Benchmark's
historical baseline qualification string is preserved verbatim and does not
mean the supplied candidate had no prior optimizer history.

![Run 03 Observed Marker Errors](native_evidence/refinement_9967_03/marker-errors.png)

The shaping expansion improved club error but only modestly improved terminal
all-marker error under this small budget. It does not establish convergence,
a global optimum, or failure of global sextic controls. There were just four
optimizer function evaluations plus numerical Jacobian probes/replays. Retain
all raw marker errors, including the known rigid head/back model mismatch.

A direct native_model.py mypy invocation exposed incomplete local Pinocchio
stubs and pre-existing constructor variable/field typing errors that the
normal push hook skipped. Fixed the optional C++ boundary using an explicitly
dynamic import, distinguished joint specifications from numeric joint IDs and
typed the frame map. No dynamics formula changed. Direct check now passes:
python3 -m mypy src/engines/physics_engines/pinocchio/python/native_model.py
--follow-imports=silent. Derivative/topology/URDF-binding targeted suite:
12 passed (suite-marker advisory on two existing tests remains report-only).
The local derivative receipt below predates this typing-only source revision;
it must retain its actual deployed source hashes.

Next: audit force-versus-torque correction scales in physical units before
another expensive search. Current +/-2 bounds were exploratory corrections,
not measured physical limits; equal numerical values in N and Nm are not equal
physical authority. Quantify their attainable COM impulse/displacement and
compare to feasible pose diagnostics before choosing new scales. Separately
continue the staged trajectory-sensitivity validation below to improve
Jacobian accuracy and computation cost. Do not claim a trajectory speedup
from the local acceleration derivative timings alone.

## Local Constrained-Dynamics Derivatives Verified

Updated 2026-09-12 UTC. Previous goal turn made progress by launching expanded
run 03. It remains live (session 17132, WSL PID 1911742); do not replace it.
This turn independently probed installed Pinocchio 4.1.0's
computeConstraintDynamicsDerivatives, which requires constraintDynamics first.
New public acceleration_derivatives refreshes dynamics, maps scalar native
coordinate order explicitly, validates matrices, and returns detached readonly
q/rate/primitive-effort derivative blocks. Two unit tests went red then green;
combined derivative/closure boundary suite: 4 passed. Commit 81f848c54.

Actual ControlTower audit on run 02's continuous states at 0,.60,.80 s exited
zero. It reuses shared estimation.residuals.finite_difference_jacobian rather
than another finite-difference implementation. At central step 1e-5, maximum
scaled discrepancies across q/v/effort blocks and all three states are below
9.59e-5. Step 1e-4 shows q nonlinearity up to 2.03e-4 scaled discrepancy;
step 1e-6 is noisier, up to 1.29e-3. Do not assume smaller finite differences
are automatically more accurate. Scaling is abs(error)/max(1,abs(reference)),
with blocks reported separately in native units.

Warm analytic calls took ~0.214–0.224 ms versus ~8 ms for the full local central
finite-difference matrix; the first analytic call took 4.23 ms. These are LOCAL
acceleration timings, not end-to-end trajectory-gradient or fitting speedups.
No analytic Jacobian is used by the running optimizer. Receipt:
native_evidence/native-derivatives-9967-01.json; reproduction runner:
native_evidence/reproduction/check_native_derivatives.py. Original copies are
on ControlTower under C:/Users/diete and in local simscape-tour-checkpoints.
Independent runtime /home/dieterolson/native-derivative-9967-01 and source bundle
native-derivative-bundle-9967-01.zip preserve hashes; run 03's deployed native
module was not modified. This remains a namespace diagnostic deployment.

### Next Sensitivity Implementation Stage

1. Add a native marker-position Jacobian with explicit scalar coordinate order.
   Reuse Pinocchio frame Jacobians and existing marker offsets/projection.
   Test angular cross-product sign on an offset point, ordering, finite values
   and detached arrays; compare actual FK finite differences on native states.
2. Implement an augmented state/sensitivity integrator reusing integrate_forward.
   Test an analytic polynomial-forced system (for example q''=theta\*t^2 gives
   dq/dtheta=t^4/12 from fixed initial state), full absolute clock, failed RHS,
   fixed initial sensitivities and non-finite output. Do not introduce resets.
3. For native fitting, form Sdot=A*S+B*du/dtheta from the qualified local blocks.
   Include the world-force to native-base rotation and the actual Bernstein
   control basis in du/dtheta. Preserve the original candidate time origin.
4. Independently compare integrated marker derivatives against several complete
   perturbed forward replays at multiple step sizes. Check primal replay error,
   closure, sensitivity finiteness and total wall time. Local derivative checks
   alone do not qualify this trajectory Jacobian or its behavior near singular
   coordinate charts. Record thread settings and avoid noisy small-matrix BLAS
   oversubscription if observed; do not alter running job environments.
5. Only then add an optional Jacobian to the shared prefix fitter, matching ALL
   enabled residual terms and masks exactly. Unsupported yaw/regularization
   derivatives must fail explicitly rather than silently disappearing. Cache
   primal/Jacobian pairs by the full immutable candidate/clock identity and
   count actual integrations separately from optimizer callbacks.
6. Benchmark equal-budget fitting from a preserved seed before claiming faster
   convergence. Keep finite-difference fitting available as a reference.

Run 03 latest saved best exploratory sample was evaluation 157: whole RMS
30.8489 mm, early 10.8104 mm, terminal 94.5785 mm, club cluster 67.3016 mm.
This is not the returned optimizer candidate or an accepted match. Some other
finite-difference probes have much larger terminal error; preserve the ledger
and wait for terminal exit before judging the result. The current 1e-3 control
perturbation has not been qualified as an accurate trajectory derivative.

## Run 03: Expanded Sextic Shaping Is Live

Updated 2026-09-12 UTC. Previous goal turn made progress by verifying native
Hub topology and measuring pair spacing over the whole capture. This turn
implemented tested Bernstein increment/recovery helpers and launched a bounded
forward optimization with the unchanged native model, markers and initial state.
Source commit 6efa138fd. The new candidate test first failed on missing helpers,
then all 16 candidate tests passed, including analytic B4/B5/B6 values and exact
identity of a pure t^6 correction in the two representations.

ControlTower WSL PID 1911742 is live in exec session 17132. Do not duplicate or
restart it while its actual handle/process lives. Initial evaluation exactly
reproduces run 02 candidate SHA
90d801cad282dba88170442dc7a57e4757aef04780bf267f311bd1a098f498b7
and all four errors (whole 31.5031 mm, early 10.9514 mm, terminal 96.8489 mm,
club cluster 78.6801 mm). No new improvement has yet been qualified.

Run 03 frees Bernstein controls 4, 5 and 6 for each of 27 effort channels,
81 parameters, in coordinate-major ascending-control order. Controls 0..3
remain zero. Correction controls are 10\*(x-1), x in [.8,1.2], relative to the
ORIGINAL baseline candidate; the run 02 returned candidate initializes x.
This retains a single global degree-six polynomial and bounds correction
magnitude by 2 N/Nm on [0,.80] through Bernstein convexity. It is not a bound
on total effort or extrapolation beyond .80. Geometry, initial q/qd, marker
assignments, capture identity, native integration and all acceptance gates
are unchanged. max_nfev=4, relative finite-difference step .001; 81 finite-
difference perturbations per Jacobian add calls beyond the four-step budget.

Runtime: /home/dieterolson/native-refinement-9967-03/refine_native_candidate.py
Python: /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python
Output: C:/Users/diete/native-refinement-9967-03
Arguments: --model /mnt/c/Users/diete/native_geometry_spec_9967.json
--candidate /mnt/c/Users/diete/native_replay_candidate_9967.json
--target /mnt/c/Users/diete/driver_marker_payload_9967.json
--restart-candidate /mnt/c/Users/diete/native-refinement-9967-02/returned-candidate.json
--shaping bernstein456 --output_dir /mnt/c/Users/diete/native-refinement-9967-03
--max-nfev 4

Archive native-refinement-bundle-9967-03.zip is preserved locally under
simscape-tour-checkpoints and on ControlTower under C:/Users/diete; its
source_manifest.json hashes the bundled modules and runner. This remains a
namespace diagnostic deployment rather than a full application import claim.
Config/ledger/best/returned/failure behavior is retained from previous runs.

Next: poll session 17132 and the actual worker; do not infer termination from
a missing early output file or a polling timeout. At terminal exit archive all
artifacts, independently replay the returned candidate, report every gate and
per-marker/local-window behavior. Preserve the run 02 accepted-as-seed identity
if new exploratory parameters worsen early retention. No full swing, changed
geometry, extra head DOF or new R2025b acceptance is established by this run.

## Full-Capture Hub Topology and Pair Audit

Updated 2026-09-12 UTC. Previous goal turn made progress by measuring native
closed-pose feasibility and fixed-offset rigidity. This turn verified the
native geometry inventory and measured spacing over ALL 654 capture samples.
Receipt: native_evidence/native-hub-pair-audit-9967-01.json, reproduced by
native_evidence/reproduction/check_hub_marker_pairs.py. It validates model,
candidate and capture hashes and preserves actual masks. All 15 Hub pairs
have 654 observed frames, through 1.813889 s, in this payload.

The native Hub frame belongs to COMRod's rigid aggregate, which contains the
Head, Neck, UpperTorsoTop and ZeroMassShoulderReference solids. The exported
native joint inventory has no separate head joint. Thus the current six-marker
Hub assignment is consistent with the rigid native topology; a different fixed
head offset cannot reproduce all observed head/back relative motion.

Across the capture, each of the three head-to-head pair distance ranges is
<=2.101 mm (standard deviation <=0.323 mm). HeadSide-to-BackLeft distance ranges
from 349.682 to 459.286 mm: span 109.604 mm, standard deviation 37.221 mm.
HeadFront-to-BackLeft spans 97.617 mm. Back-to-back pairs also vary, by up to
31.364 mm. The data support relative head/back motion and some within-back
constellation variation; this audit does not attribute that variation uniquely
to anatomy, averaging or marker measurement effects.

For any fixed separation, the best independently posed two-marker RMS is
std(observed distance)/2: 18.610 mm for HeadSide/BackLeft. This bound permits
recalibration of that pair's constant separation, unlike the previous fixed-
offset instantaneous bound. It applies ONLY to that pair across time and must
not be mislabeled as all-25-marker RMS or proof of failure of the 35 mm gate.
No rigid offset or constant segment-length change can eliminate this pair's
length variation. The analytical pair-bound tests went red then green; all
four rigidity tests pass, including masks and rigid-transform invariance.

### Next Native Matching Step

Keep the verified 27-input native-equivalent model as the baseline. The gate
has not been proved unreachable, and substantial forward error still exceeds
the best found static pose error. Resume native torque refinement with a
bounded expansion from only the t^6 correction to t^4/t^5/t^6 shaping directions,
while preserving one global sextic, original initial state, geometry and marker
assignments. Reuse bernstein_to_simscape and the existing candidate/forward
adapter. A convenient bounded experiment uses only Bernstein control entries
4, 5, 6 free, earlier entries zero, initialized from run 02's t^6 correction.
Test physical-time equivalence and restart/bound semantics before launching;
record that bounds on [0,.8] do not bound extrapolation to the full capture.
Use all raw observed markers and existing acceptance gates, plus local-window
and per-marker diagnostics. Do not optimize away the head errors by omission.

A separate neck/head model variant may improve representational capacity, but
it must be explicitly distinct from the native-equivalent baseline and must
modify/qualify the MATLAB R2025b model too. Do not silently introduce extra
Pinocchio joints and retain the old equivalence claim. This topology proposal
has been communicated for coordination; no such variant has been implemented.

## Later-Capture Rigidity Spot Checks

Updated 2026-09-12 UTC. The raw driver_marker_payload.json from preserved
prefix-1200ms-sextic-01 was checked against the candidate capture SHA. Despite
its historical folder name it contains observations through 1.813889 s.
The same fixed-attachment relaxation was evaluated at 1.0, 1.2, 1.5 and 1.8 s,
without forward simulation or extending the candidate's 0.80 s validity claim.
All-marker lower bounds: 27.4147, 19.0406, 21.0997 and 21.2227 mm respectively.
Observed modeled-marker counts: 25, 25, 22 and 25. Missing samples stayed masked.

Receipt native_evidence/native-marker-rigidity-full-9967-01.json records these
seven spot checks including the earlier .60/.70/.80 values. Its filename does
not imply every capture frame was evaluated. None of these sampled lower bounds
alone proves the 35 mm all-marker gate impossible. They identify an unavoidable
fixed-offset contribution which remains relevant throughout the capture.
Reproduce with check_marker_rigidity.py --candidate returned-candidate.json
--capture-payload driver_marker_payload.json --times 0.6 0.7 0.8 1.0 1.2 1.5 1.8
--output a_new_receipt.json. This is a geometry/observation diagnostic only.

The next action remains a documented attachment/native-topology audit and
multiframe calibration feasibility check, not dropping markers to force a pass.
The first push of the diagnostic commits was rejected by hooks reporting files
modified while this follow-up CLI work was being edited; retry normal hooks only
after committing this update. Do not bypass hooks or claim that first push passed.

## Native Pose Feasibility and Fixed-Marker Rigidity

Updated 2026-09-12 UTC. Previous goal turn made progress: run 02 finished,
independent forward replay passed, and artifacts were committed/pushed. This
turn implemented a tested static diagnostic and executed three native local
pose-fit experiments on ControlTower. All diagnostic processes exited zero;
no replacement forward optimizer is running in this workstream.

Three radii were examined, always centered on the original run 02 replay q:
01: translation +/-0.2 m, rotations +/-0.5 rad, replay seed;
02: +/-0.4 m / +/-1.0 rad, seeded from 01;
03: +/-0.8 m / +/-3.2 rad, seeded from 02. Geometry, attachments and six-DOF
grip closure remained unchanged. These are search envelopes, not newly imposed
physical native joint limits. Actual native limits were previously audited off.

| Time   | Forward RMS | Narrow Pose RMS | Wide Pose RMS |
| ------ | ----------: | --------------: | ------------: |
| 0.60 s |  32.7013 mm |      20.5761 mm |    20.5761 mm |
| 0.70 s |  55.4121 mm |      27.4923 mm |    27.4923 mm |
| 0.80 s |  96.8489 mm |      39.4278 mm |    36.8367 mm |

All final local solves converged; closure maximum <=2.66e-13 and no active
search bounds in experiment 03. Experiment 02 had LFInput on its bound at
0.80 s, which is why the third probe was run. These are feasible static poses,
not dynamically realizable trajectories or proven global minima. The best
found 0.80 s pose still misses the 35 mm terminal gate. Do not declare that
no solution exists solely because these local searches missed it.

### Rigidity Finding

The current candidate assigns BackTop, BackLeft, BackRight, HeadTop, HeadFront
and HeadSide to one Hub frame with fixed offsets. A separate optimistic
lower-bound diagnostic gives every attachment frame an independent rigid pose,
removing ALL joint and grip constraints. It reuses the existing proper-rotation
Kabsch solver, permits translation/rotation but no scaling/outlier removal.

| Time   | All 25 Markers' RMS Lower Bound | Hub Six-Marker RMS Lower Bound |
| ------ | ------------------------------: | -----------------------------: |
| 0.60 s |                      10.8473 mm |                     21.8221 mm |
| 0.70 s |                      16.1695 mm |                     32.5402 mm |
| 0.80 s |                      21.4230 mm |                     43.1478 mm |

These are lower bounds only for the CURRENT fixed offsets/assignments and
observed equal-weight markers. They do not apply to recalibrated constellations
or a changed model variant. They establish that torque changes alone cannot
remove all error in the current head/back constellation. They do NOT prove
the 35 mm all-marker gate impossible: 21.423 mm is below it. The local constrained
pose result and the optimistic lower bound are deliberately separate evidence.
Do not silently drop head markers or add a neck joint to the native-equivalent
model. Audit mapping, native topology and raw C3D rigid-cluster consistency;
any calibrated/model variant must be explicit and requalified.

### Artifacts and Next Execution

Receipts are native_evidence/native-pose-feasibility-9967-01.json through -03.json
and native-marker-rigidity-9967-01.json. Later pose receipts include per-marker
errors, active bounds and seed-report hashes. The original remote receipts
remain under C:/Users/diete on ControlTower; local copies and source bundle
native-pose-bundle-9967-01.zip are under simscape-tour-checkpoints. Actual runtime
was /home/dieterolson/native-pose-9967-01 using the existing Pinocchio venv.
Script check_native_pose_feasibility.py reproduces each radius/seed experiment;
wide diagnostic copy on ControlTower is check_native_pose_feasibility_wide_9967.py.
The second/third runs used --seed-report pointing to the preceding receipt,
--translation-radius and --rotation-radius as above. Original deployment
manifest precedes these CLI additions; the committed runner is the updated
source for wide runs. Do not mislabel that bundle as the later runner revision.

Pure constrained-pose and rigidity tests went red on missing implementation,
then green; combined pose/rigidity/plot tests: 11 passed. Native solves provide
separate runtime evidence. The rigidity command check_marker_rigidity.py runs
locally from the candidate-only JSON and its visual-replay.npz; it validates
candidate/marker identity and refuses to overwrite the output report.

Next: extend rigidity diagnostics to the full capture (including raw observation
masks) before another long torque run. Review whether native head/back rigidity
and fixed attachments can meet the intended full-swing tolerances. Investigate
multiple pose seeds and bounded multiframe geometry/attachment calibration
with held-out poses; preserve the original model baseline and model/candidate
hashes. Do not fit independent lengths per time sample. Then resume torque
optimization from an explicitly selected calibrated/native variant and verify
both grip closure and native R2025b parity for any physical changes.

## Run 02 Finished and Independently Replayed

Updated 2026-09-12 07:38 UTC. This supersedes all live-run statements below.
Exec session 48221 exited zero after 338 evaluations; our optimization is now
terminal. No replacement job has been launched. Independent fresh-process
replay exactly reproduces the returned candidate's four RMS metrics.
Canonical candidate SHA256:
90d801cad282dba88170442dc7a57e4757aef04780bf267f311bd1a098f498b7.

| Metric                    | Run 01 Returned | Run 02 Returned | Acceptance Gate |
| ------------------------- | --------------: | --------------: | --------------: |
| Whole 0.80 s Prefix RMS   |      34.4259 mm |      31.5031 mm |           25 mm |
| Early Through 0.60 s RMS  |      10.5434 mm |      10.9514 mm |           12 mm |
| Terminal RMS              |      132.142 mm |      96.8489 mm |           35 mm |
| Terminal Club Cluster RMS |      177.079 mm |      78.6801 mm |           60 mm |

Numerical acceptance and optimizer convergence both remain false; max_nfev=12
was exhausted. Fresh replay plus residual took 3.40328 s; closure pose/velocity
maxima were 3.52597e-11 / 1.76122e-10. This is improvement, not a completed
prefix/full-swing match. No new R2025b acceptance run has been made.

Five controls are within 0.0001 of their original [0.8,1.2] optimizer bounds:
TranslationInputZ (upper), LScapInputX (lower), LSInputZ (upper), LWInputX
(lower), RWInputX (upper). The associated sixth-power correction at 0.80 s
is approximately +/-2 N or Nm. These are correction bounds relative to the
original candidate, not total physical-effort bounds. Earlier no-bound
observations do not apply to the final returned result. Improvement resumed
late in the run, so an earlier apparent plateau was not a proven minimum.

Artifacts: native_evidence/refinement_9967_02/ contains config, every evaluation,
best exploratory sample, optimizer return, candidate-only JSON, fresh replay
receipt, portable trajectory arrays and the new marker-error PNG. The original
remote folder C:/Users/diete/native-refinement-9967-02 and local checkpoint
archive remain intact. The benchmark's old baseline qualification string is
preserved as emitted; it denotes a measurement-only diagnostic, not absence
of an optimizer upstream. Canonical hashes identify semantics independently
of pretty-printed file bytes.

![Run 02 Observed Marker Errors](native_evidence/refinement_9967_02/marker-errors.png)

### Next Bounded Work

1. Recheck central claim and remote processes; do not infer another agent's
   worker is ours. Reproduce this returned candidate before using it as a seed.
2. Run a constraint-aware static marker-fit diagnostic at 0.60, 0.70 and 0.80 s
   with original geometry/attachments and both grips retained. Start from the
   corresponding replay states; use multiple local seeds if needed. Reuse the
   existing FK/projection and inspect existing Pinocchio IK APIs before adding
   code. Report optimization success and actual closure residual separately.
   A found feasible pose demonstrates reachability; failure of a local search
   does not prove an irreducible error floor. This is IK diagnosis, not inverse
   dynamics, and it does not qualify a forward swing.
3. Inspect finite-difference sensitivity across smaller steps before changing
   the model. Existing shared estimation.identifiability finite_difference_jacobian
   is available; do not duplicate it. Preserve actual module dependencies and
   source hashes if bundling it for the remote diagnostic.
4. If pose reachability is good and derivatives are stable, compare two small,
   explicitly separate experiments from the original-base-relative run 02 seed:
   selected saturated correction bounds widened modestly versus additional
   t^4/t^5 shaping directions within a declared global sextic basis. Do not
   expand all controls blindly or recenter original bounds at the new seed.
   Use unchanged marker gates, per-marker/local-window diagnostics, effort
   extrema and native closure checks. Bernstein controls can span the new
   directions using existing bernstein_to_simscape; conversion never changes
   the physical polynomial or original time origin.
5. Promote only after independent unsegmented replay and R2025b validation.
   Do not extend beyond 0.80 s based solely on the improving aggregate score.
   Preserve failed and partial experiments with their real status.

## Actual Forward-Replay Visual and Transition Diagnosis

Updated 2026-09-12 UTC. Run 02 remains live; do not restart it. The previous
continuation made progress by implementing/test-driving restart support and
launching the verified ControlTower worker. This turn added a reproducible
visual of the independently replayed, completed run 01 candidate (not live FD
samples). The new six-test marker report suite passes after its missing-module
red test. Pure plotting requires no Pinocchio runtime.

![Run 01 Observed Marker Errors](native_evidence/refinement_9967_01/marker-errors.png)

The visual exposes an important limitation of aggregate early retention:
instantaneous across-marker RMS is 14.5953 mm at 0.50 s, 30.3753 at 0.60 s,
50.5658 at 0.70 s, 74.1305 at 0.75 s and 132.142 at 0.80 s. The early-window
average 10.5434 mm does not imply close tracking at its endpoint. Do not state
that the first 0.60 s is exactly solved, or that divergence starts only at 0.70 s.
Inspect short-window/endpoint residuals as well as the unchanged acceptance
gates before extending the horizon. The curve is evidence of accumulating
trajectory error; it does not alone identify torque, geometry or singularity
as its cause. A constraint-aware kinematic floor remains necessary if torque
refinement stalls. Run 02 latest exploratory terminal RMS is about 103 mm;
no returned/accepted run 02 result has yet been verified.

Portable arrays, exact source receipt and PNG are committed beside run 01:
visual-replay.npz, visual-replay.json and marker-errors.png. Fresh forward
replay uses the same returned candidate and matches all earlier aggregate
metrics. The archive contains absolute times, SI target/prediction, validity
mask, labels and canonical candidate hash. Missing observations stay missing.

Reproduce: run native_evidence/reproduction/benchmark_native_marker_residual.py
with the original model/target and run 01 returned-candidate.json, a new
--output receipt path and --trajectory-output archive.npz. Then run
native_evidence/reproduction/plot_native_marker_replay.py --trajectory archive.npz
--output plot.png with repository root on PYTHONPATH and MPLBACKEND=Agg.
Both commands refuse to overwrite their output artifacts. The chart shows
actual forward marker errors, not IK/prescribed motion or engine-parity error.

## P4 Restart Run 02 Is Live

Updated 2026-09-12 UTC. Previous goal turn made progress: run 01 finished,
independent replay confirmed improvement, artifacts and turnover were pushed.
This continuation added tested restart support in commit 90def342a. All 15
native-candidate tests pass (new restart tests first failed on missing helper).
Restart rejects non-control identity changes; its original +/-2 correction
bounds remain centered on the original base, not on the improved seed.

Actual ControlTower WSL worker PID 1885136 is live in exec session 48221.
Do not restart or duplicate while that handle/process remains live. Run 02
first evaluation exactly reproduced run 01 returned candidate SHA and all four
RMS metrics, confirming the restart did not change the saved controls.
No coordinates were initially near the original bounds (within 0.0001).

Runtime: /home/dieterolson/native-refinement-9967-02/refine_native_candidate.py
Python: /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python
Output: C:/Users/diete/native-refinement-9967-02
Arguments: --model /mnt/c/Users/diete/native_geometry_spec_9967.json
--candidate /mnt/c/Users/diete/native_replay_candidate_9967.json
--target /mnt/c/Users/diete/driver_marker_payload_9967.json
--restart-candidate /mnt/c/Users/diete/native-refinement-9967-01/returned-candidate.json
--output_dir /mnt/c/Users/diete/native-refinement-9967-02 --max-nfev 12
Run in ControlTower-Runner WSL through SSH controltower; preserve original
0.80 s basis and native integration tolerances. Full command remains visible
with pgrep -af refine_native_candidate.py. Finite-difference evaluations are
additional to max_nfev; expect several hundred replays, not twelve.

Source bundle native-refinement-bundle-9967-02.zip is archived locally under
C:/Users/diete/Repositories/simscape-tour-checkpoints and on ControlTower under
C:/Users/diete. source_manifest.json hashes all bundled modules and the runner.
The package remains a namespace-only diagnostic deployment, not proof of full
application import integration. Each evaluation now saves parameters and
near_bound_count; returned optimizer parameters are saved explicitly.

Next: poll session 48221, inspect actual PID and output ledger. On terminal
exit archive config, evaluations, best and returned/failure without overwriting
run 01. Independently replay the returned candidate and compare all gates.
If it stalls, use recorded bound activity and sensitivities to decide the next
parameterization; do not simply increase budget. R2025b acceptance remains
pending for any improved candidate. Gemini jobs have not been interrupted.

## P4 Refinement Finished: Improved Seed, Acceptance Failed

Updated 2026-09-12 UTC. This section supersedes the live-run section below.
Session 65632 exited zero after 86 evaluations; our refinement worker is done.
Returned candidate canonical SHA256:
2ebf34b2d77c56201b7a8f5d97d41c0cde3e7f79709008a5ec02a746ffd2b9a7.
Independent fresh-process replay reproduces all four marker metrics exactly:

| Metric                    |   Original |   Returned |  Gate |
| ------------------------- | ---------: | ---------: | ----: |
| Whole Prefix RMS          | 43.8225 mm | 34.4259 mm | 25 mm |
| Early Through 0.60 s RMS  | 9.86004 mm | 10.5434 mm | 12 mm |
| Terminal RMS              | 202.559 mm | 132.142 mm | 35 mm |
| Terminal Club Cluster RMS | 265.370 mm | 177.079 mm | 60 mm |

Terminal error improved about 34.8%, but this is NOT an accepted match.
Optimizer convergence is false: max_nfev=3 exhausted. No claim about a full
swing or this new candidate's R2025b agreement is made. The fresh replay took
3.47455 s including residual evaluation; closure pose/velocity maxima were
3.31405e-11 / 9.69919e-11. It used 289 timestamps, 25 attached markers and
7,225 observed samples through 0.80 s, not all 38 capture labels.

Complete small artifacts are committed in native_evidence/refinement_9967_01/:
config, every evaluation, best exploratory sample, returned candidate, separate
candidate-only JSON, and independent replay receipt. The reused benchmark's
qualification string says baseline; here it means a measurement-only replay,
not a statement that no optimizer preceded the candidate. Original receipt is
preserved verbatim. Candidate file hashes differ from canonical semantic hashes.
Raw archive and identical remote run folder remain at the paths below.

### Next-Agent Execution Prompt

1. Read this checkpoint, PINOCCHIO_EXECUTION_PLAN.md and repository rules.
   Inspect git status and central issue #9967 lease. Preserve other agents'
   running jobs. Our run 01 is terminal; never overwrite its output directory.
2. Reproduce returned-candidate.json using the committed benchmark runner and
   exact model/target hashes in independent-replay.json. Use the qualified
   ControlTower namespace bundle and native Pinocchio venv recorded below.
   All four RMS values must reproduce before another optimization starts.
3. First continue the same 27 t^6 correction directions from the original
   base with a larger bounded evaluation budget (e.g. max_nfev=12), initializing
   from the returned correction. Do not recenter bounds on the new seed:
   preserve original absolute +/-2 N/Nm envelope at 0.80 s. Add restart-vector
   support with a failing test, then minimal implementation. Preserve fixed
   0.80 s basis and original q0/qd0, geometry, frames and capture identity.
4. Record bound activity and cost reduction per evaluation. If progress stalls
   or bounds saturate, stop increasing budget blindly. Compare transition IK
   feasibility and residuals per marker; inspect force/torque sensitivity and
   conditioning. Test t^4/t^5 directions with explicit scaling and early-motion
   constraints in a separate run; these can disturb early tracking more strongly.
5. Keep best early-feasible evaluation separate from optimizer return and
   accepted result. Early <=12 mm alone is not acceptance. Enforce whole,
   terminal, club and yaw gates before promotion. Preserve failure diagnostics;
   do not turn integrator failures into a constant finite penalty.
6. Before horizon extension, replay an improved accepted prefix in R2025b with
   the qualified tight settings. Extend in small steps from a single initial
   state, retaining a fixed global polynomial basis and all earlier gates.
   Produce time-error plots and target/model overlays from actual replay;
   a visualization of target or prescribed kinematics is not forward-fit proof.
7. Coordinate Gemini's boundary-time and acceptance fixes already posted on
   #9964. Do not use its current multiple-shooting result as acceptance until
   exact boundary time, closure-consistent state defects and a gate-passing
   unsegmented replay are verified. Separate MuJoCo/Drake native parity remains
   required; URDF tree alone omits the grip loop and cannot establish parity.
8. Commit source, tests, small receipts and updated turnover after each bounded
   stage. Record live handles while running and terminal exit when complete.
   OpenSim has separate epic #10003 and staged handoff; implementation begins
   with OS-0 qualification after the user's requested planning check-in.

## P4 Bounded ControlTower Refinement Is Live

Actual native refinement launched in exec session 65632; verified WSL worker
PID 1859278 on ControlTower. Do not duplicate/restart while this handle lives.
Runner: /home/dieterolson/native-refinement-9967-01/refine_native_candidate.py
under the existing Pinocchio venv. Output directory:
C:/Users/diete/native-refinement-9967-01. Its config.json, evaluations.jsonl,
best.json, returned.json or failure.json preserve progress and distinct outcomes.
At the latest observation, evaluation 4 completed; integration is ~3.3–3.4 s.
These initial evaluations are finite differences, not a converged improvement.

This bounded experiment uses the packaged .80 s candidate, fixed geometry,
initial state and 25 attachments, adjusting only the t^6 term of each of 27
inputs. Correction is 10*(x-1)*(t/.8)^6 with x in [.8,1.2], i.e. at most +/-2
N or Nm at .8 s. The basis is fixed for the run and final inputs remain global
sextics. Shared fit_prefixes executes one full prefix, max_nfev=3 (finite
differences add calls), relative diff step 1e-3, terminal weight 10. Numerical
acceptance still requires whole/terminal/yaw gates plus explicit early <=12 mm
and club cluster <=60 mm; final R2025b acceptance remains separate.

New increment_native_candidate helper reuses normalized_to_simscape; its new
test progressed red-to-green and all 11 candidate tests pass. It creates a new
identity without retiming the original controls. The runner is committed under
native_evidence/reproduction/refine_native_candidate.py. Source bundle with
module hashes is archived locally as native-refinement-bundle-9967-01.zip and
deployed separately from the qualified baseline bundle. Each evaluation is
logged, best early-retaining score is saved atomically, and returned optimizer
candidate is kept separate. Exceptions preserve failure evidence; no finite
penalty plateau masks solver failure.

Next: poll session 65632 and inspect last JSONL records/best checkpoint. Wait
for terminal exit before treating returned.json as complete. Archive outputs,
independently replay any improved candidate, compare all gates, then decide
whether to widen correction bounds/add t^4/t^5 directions. Gemini's separate
DeskComputer multiple-shooting run is untouched; coordination findings below
still apply to its boundary and acceptance behavior.

## P4 Capture Baseline and Active-Agent Review

All commits through e618aa3d4 are pushed; corrected type checks and remaining
push hooks passed. Actual ControlTower candidate-to-C3D evaluation completed
exit zero using the packaged adapter on 289 capture timestamps through .80 s.
25 modeled attachments / 38 capture labels, 7,225 observed marker samples:
whole RMS 43.8225 mm, early through .60 s RMS 9.86004 mm, terminal RMS
202.559 mm, terminal six-marker club-cluster RMS 265.370 mm. Adapter plus
residual took 3.48981 s, integration 3.35972 s. Receipt is
native_evidence/native_c3d_baseline_9967.json; runner is
native_evidence/reproduction/benchmark_native_marker_residual.py. This is the
packaged candidate baseline, NOT a newly optimized or accepted swing.

Inputs on ControlTower: native_geometry_spec_9967.json,
native_replay_candidate_9967.json, driver_marker_payload_9967.json. The target
payload came from local archive prefix-1200ms-sextic-01/driver_marker_payload.json
and carries the same driver source SHA. Baseline uses its first .80 s and
explicit validity mask; it makes no claim about all-marker/full-capture coverage.

Gemini's latest plan and actual DeskComputer run_real_ms_080s.py were inspected.
The runner now genuinely calls fit_multiple_shooting with 189 controls + 54
node-state variables. Active Python launcher/worker PIDs 33608/76992 and MATLAB
73720 were observed; do not interrupt them. This supersedes the earlier
single-shooting-only finding for this NEW runner, not historical evidence.
New defect: it simulates win_duration+.001 and returns final q/qd as the
boundary state, advancing nominal .60 to .601 s. Its runtime library accepted
flag still allows any finite replay RMS; the runner copies it even if gates
fail. Node closure/assembled-state identity also needs explicit verification.

Findings and corrective instructions were sent to issue #9964:
https://github.com/D-sorganization/UpstreamDrift/issues/9964#issuecomment-5644311396
and Gemini brain CODEX_LIVE_MS_REVIEW_20260912.txt. Read-only source snapshots
are archived as review_run_real_ms_080s_20260912.py and
review_multi_shooting_fit_20260912.py. No external fitting process was changed.
Next: coordinate correction of exact node boundaries/acceptance, then connect
the qualified ControlTower adapter to a bounded reproducible optimization run
with immutable sextic basis and preserved best feasible candidate. Do not
duplicate the active DeskComputer run or call its 12-hour schedule a guarantee.

## P3 Actual URDF Round Trip Passed on ControlTower

NativeUrdfModel now calls Pinocchio buildModelFromXML on the exported tree,
restores native gravity/unbounded limits, maps scalar q/v and body/reference
frames, and attaches the original weld via the shared closure initializer.
It does NOT rebuild bodies or inertias from the source JSON. Mandatory sidecar
validation binds URDF/source hashes, coordinate inventory, closure, gravity and
frame mappings before native construction. Eight binding tests progressed
red-to-green; the combined binding/accessor/adapter suite passes 14 tests.
Ruff passes on all changed code and the reproduction runner.

Actual ControlTower diagnostic completed exit zero and passed. Total mass is
77.60581783574676 kg; COM difference 1.11e-16 m; maximum six-state frame-matrix
difference 1.11e-15; maximum scaled moving-state acceleration difference
2.63e-10; maximum scaled baseline/27-unit-input response difference 2.49e-11.
The URDF-loaded closed model replayed the saved 0.80 s candidate: q difference
6.46e-8 and qd difference 7.21e-5 versus qualified diagnostic, integration
4.096 s, maximum closure pose/rate residuals 2.62e-11 / 2.47e-9. Mixed-coordinate
differences are not marker metres. Receipt: native_evidence/native_urdf_roundtrip_9967.json.

Reproduction: check_native_urdf_roundtrip.py in native_evidence/reproduction;
arguments --urdf, --sidecar, --model, --candidate, --fixture, --reference, --output.
The source bundle is archived locally as native-urdf-bundle-9967-01.zip and
extracted on ControlTower to /home/dieterolson/native-urdf-9967-01. It preserves
module paths with a namespace-only package graph and source manifest. Inputs
are native-golf-9967-01.urdf/.sidecar.json, native_geometry_spec_9967.json,
native_replay_candidate_9967.json, native-rollout-reference-9967-01.json and
native-continuous-800ms-9967-tight-01.json, all in ControlTower's user directory.
Output native-urdf-roundtrip-9967-01.json is archived locally and summarized here.

Next: wire a repeatable matcher to the qualified candidate/adapter, preserve
absolute-second global sextics and assess full observed-swing tracking from
the saved best candidates. Also qualify MuJoCo/Drake loaders against the same
URDF+sidecar when extending interchange; their native execution is not proven
by Pinocchio's result. The isolated bundle is not full application packaging.
No native process remains live. Full-swing C3D acceptance remains outstanding.

## P3 Native Tree Exported and Parsed

`motion_matching/native_urdf.py` now exports the native spec through the shared
17-digit writer. It explicitly represents 27 scalar primitives, 31 original
solids and 16 reference frames; primitive/body/frame helper links have zero
mass. Individual solid inertias and transforms are retained rather than
recomputed from approximate shapes. Weld closure and original names/maps are
preserved in a mandatory sidecar. Four tests progressed red-to-green; Ruff
passed. These tests cover mass accounting, zero passive damping, metadata,
missing closure/coordinates and invalid transforms.

Actual ControlTower Pinocchio loaded the exported URDF successfully, with
nq=nv=27 and total mass 77.60581783574676 kg. Session 65331 exited zero.
Receipt: `native_evidence/native_urdf_parse_9967.json`. This is parsing and
mass evidence ONLY: closure was not attached and dynamics remain unqualified.
Artifacts on both local checkpoint archive and ControlTower user directory:
`native-golf-9967-01.urdf`, `native-golf-9967-01.sidecar.json`.
Reproduce locally via export_native_urdf(original_model_bytes); write XML with
write_bytes(xml.encode()) to preserve the sidecar's exact urdf_sha256.

The installed parser rejects literal infinite limits (negative probe archived
as probe_urdf_infinite_limits_9967.py). URDF uses finite binary64-maximum limit
placeholders. The sidecar requires restore-unbounded-before-dynamics; qualified
loaders MUST restore infinite position/velocity/effort limits and attach the
native weld before execution. No generic +/-pi, +/-1 m, 1000 Nm or damping
defaults are inherited. General parser limit rules are documented upstream:
https://github.com/ros/urdfdom . The actual installed parser probe governs this
runtime; other engines need their own compatibility checks.

Next single action: implement a sidecar-bound Pinocchio URDF loader that checks
URDF/model identity, maps all native scalar q/v indices and frame names, sets
gravity and restores unbounded limits, builds the 6D weld from the mapped body
frames, then uses existing NativePinocchioModel public replay operations.
Do not rebuild dynamics from the JSON instead of loading the URDF when claiming
round-trip parity. Verify mass/COM, six native FK/acceleration states, input
pulses, then the saved candidate replay. No process remains live.

## P3 Writer Precision Prepared

All prior commits through 20fea65c1 are pushed; normal push checks passed.
The existing shared model-generation URDFWriter now accepts numeric_precision
from 1 through 17, retaining the previous default of 6. All its numeric XML
fields use that option. Native export must request numeric_precision=17 and
expand_composite_joints=False, with explicit massless primitive links.

Five new tests progressed red-to-green, including exact binary64 XML round
trip for mass, inertia, COM and joint origin, preservation of zero link mass,
and rejection of invalid precision. The combined new and existing URDF
round-trip suite passes 34 tests. Ruff lint/format passed. No native tree has
yet been exported/reloaded; precision support alone is not P3 acceptance.

Next single implementation: convert the native spec into shared Link/Joint
objects with explicit primitive and fixed solid/frame links, preserve all
native transforms and per-solid inertias, and export through this writer.
Record the explicit weld and actuation/gravity/provenance metadata separately.
Audit URDF joint-limit representation: do not silently invent physical limits
or inherit the generic exporter's effort/velocity bounds. Tests must reject
lost primitives/closure and artificial intermediate mass; actual Pinocchio
reload must preserve mass, COM, FK, input responses and continuous replay.
No native process remains live.

## P2 Packaged Adapter Passed on ControlTower

Actual execution completed with exit code zero in the existing ControlTower
WSL environment: Python 3.12.3, Pinocchio 4.1.0, NumPy 2.5.3, SciPy 1.18.1.
On the saved 867-sample clock through .80 s, the packaged adapter reproduces
qualified diagnostic q and qd exactly (maximum differences both zero).
Total adapter call took 3.80991 s; integration took 3.56810 s. Maximum sampled
closure pose/rate residuals are 1.23770e-11 / 1.69879e-9.
Receipt: `native_evidence/native_adapter_check_9967.json`.

Deployment is explicitly a namespace-only diagnostic bundle, not the complete
application package graph. Eight production modules are unmodified and hashed
in its source manifest. Local archive `native-adapter-bundle-9967-01.zip` lives
under simscape-tour-checkpoints; remote copy is in ControlTower's user directory,
extracted to `/home/dieterolson/native-adapter-9967-01`. The runner source is
`native_evidence/reproduction/run_packaged_native_candidate.py`, copied to the
bundle root. Invoke it using the established Pinocchio venv Python, arguments
--model /mnt/c/Users/diete/native_geometry_spec_9967.json
--candidate /mnt/c/Users/diete/native_replay_candidate_9967.json
--reference /mnt/c/Users/diete/native-continuous-800ms-9967-tight-01.json
--output a NEW receipt path. The original receipt output was
`/mnt/c/Users/diete/native-adapter-check-9967-01.json`; never overwrite it.

Next is P3 interchange. Existing shared model-generation URDFWriter is reusable
but writes inertias with only six significant digits and its composite-joint
expansion introduces INTERMEDIATE_LINK_MASS. Preserve native precision and
explicitly expand massless primitive links instead of using that default
expansion unchanged. Existing dtack exporter also imposes generic effort and
velocity bounds, so it is not an exact native interchange path as-is. Add red
round-trip tests before adapting the shared writer. Preserve closure, gravity,
native coordinate/effort identity and provenance in a sidecar, then prove actual
Pinocchio reload pulse/trajectory parity. No full-swing acceptance is implied.
No diagnostic job remains live.

## P2 Replay Adapter Implemented: Native Execution Pending

`src/engines/physics_engines/pinocchio/python/native_replay.py` now consumes
exact model bytes, revalidates the candidate identity, checks the root force
primitive inventory, rejects partial/nonfinite clocks, and integrates once
with NativeEffortProfile and integrate_forward. It checks initial and sampled
weld pose/velocity residuals through the new public NativePinocchioModel
closure_errors accessor, and projects markers through the shared helper.
Its result contains read-only samples, candidate hash and sampled closure
maxima. It does not assert C3D fit acceptance or check every internal integrator
trial state's closure; only the initial and requested output states are gated.

Two accessor tests and four adapter tests progressed red-to-green. Analytic
sextic forcing, changed model bytes, broken initial closure and partial coverage
are covered. The combined candidate/effort/integration/projection/adapter suite
passes 34 tests; Ruff passed after import sorting. Fake-engine contract evidence
does not replace actual Pinocchio execution. No native job is live.

Next single action: stage an isolated runtime bundle on ControlTower containing
unmodified native_model.py/native_replay.py and shared native_candidate.py,
native_effort_profile.py, polynomial_torque.py, validate_theta.py,
continuous_forward.py and marker_projection.py, preserving module paths and
source hashes. Avoid importing the unrelated full launcher/UI package graph.
Use the existing Pinocchio 4.1.0/SciPy environment and exact original model bytes
bound by native_replay_candidate_9967.json. Run replay_candidate on the saved
native clock (or first a capture-rate full 0.80 s clock), compare against the
qualified diagnostic, and save dependency/source/candidate/runtime receipts.
If deployment uses a namespace-only diagnostic bundle, say so explicitly and
do not claim full application-package integration. P3 URDF reload remains next.

## P2 Candidate Package Implemented

`motion_matching/native_candidate.py` provides a versioned validated JSON
snapshot with content identity. Ten tests progressed from missing-module red
to green: detached snapshots, input-sensitive identity, wrong model hash,
coordinate reordering, coefficient/time/force-frame ambiguity, invalid initial
state/marker offsets/duration and missing provenance. Ruff passed. The snapshot
is a package contract, not a native-physics certificate or engine adapter.

`native_evidence/native_replay_candidate_9967.json` packages the saved 0.80 s
candidate with 27 coordinates and 25 fixed seed attachments. model_sha256 is
the BYTE hash of the original `native_geometry_spec_9967.json` in the local
checkpoint archive / ControlTower user directory, not a prettified re-export.
source_sha256 identifies `native-rollout-reference-9967-01.json`; capture_sha256
identifies the driver C3D. The package owns q0/qd0 from native measured initial
state, native highest-power-first coefficients and absolute-second duration.
Recompute candidate identity through the parser after serialization; do not
confuse the snapshot's canonical hash with a pretty-printed file-byte hash.

Next single implementation: native engine replay adapter consuming this package
and model bytes, verifying model hash, coordinate/primitive inventory, marker
frames and initial grip pose/velocity closure before integrating. Reuse
NativeEffortProfile, integrate_forward and project_markers. Reject partial
coverage and nonfinite outputs; include final closure checks and immutable
receipts. NativePinocchioModel needs a public closure-residual accessor so the
adapter does not inspect its internal constraint-data objects. Test these
contracts with a fake engine, then execute this exact package on ControlTower
and compare with the qualified diagnostic before optimizer integration.
No native job is live. P2 adapter and P3 URDF interchange remain incomplete.

## Second Tolerance Comparison Complete: Proceed to P2

All jobs are terminal, exit zero: export 31614, continuous replay 19541,
summary extraction 20657 and marker diagnostic 1659. Second native export has
148,148 samples through 0.80 s. Coefficients, geometry and initial q/qd are
exactly unchanged from the first tight export. Native tolerances alone tighten
from 1e-10/1e-12 to 1e-11/1e-13, with MaxStep still 0.0001 s.

Maximum projected marker difference falls from 6.17217e-6 to 1.35908e-6 m;
time-weighted RMS falls from 4.66791e-7 to 1.02711e-7 m. Six independent
native KinematicsSolver samples confirm the latter result, including transition
and the terminal state. Pinocchio integration takes 3.972 s; maximum mixed
q/qd discrepancies are 1.35352e-5 / 0.0146977. This is converging continuous
agreement for this exact 0.80 s candidate, not full-swing C3D acceptance or
equivalence under every possible parameter change. Further tightening of the
same candidate is not the next priority: errors are far below fitting budgets.

Receipts: `native_evidence/native_marker_convergence_second_9967.json`,
`native_continuous_convergence_second_9967.json`, and
`native_tight_replay_second_9967.json`. Full native -02 export is on both hosts
and in the local checkpoint archive. Full Pinocchio -02 replay remains on
ControlTower under the same user directory and naming convention as -01.
Input hashes are preserved in receipts. No live diagnostic needs restarting.

The native effort converter also exactly reproduces the original diagnostic's
forces/torques at 1,001 times over [0,.8]; receipt is
`native_evidence/native_effort_conversion_9967.json`. Next: P2 candidate identity
and closed-chain engine adapter, with tests rejecting coefficient/time/frame
ambiguity, followed by native-qualified replay using the shared converter.
Then P3 URDF tree plus closure/actuation metadata and actual reload checks.
All commits through 948856377 were pushed with normal checks passing.

## Second Native Replay Terminal Success

Session 88805 has now exited successfully (code zero). Its MAT is complete at
1,889,946,594 bytes; companion report is copied to the local checkpoint archive
as `native-tight-replay-9967-02.mat.json`. The live job is now EXPORT session
31614, explicit R2025b on DeskComputer. Log is `native-tight-export-9967-02.log`;
output is `native-tight-reference-9967-02.json`, both under SimscapeTour9921.
Poll this existing handle; on success copy the exported JSON to the archive
and ControlTower, then rerun the continuous and independent marker diagnostics
against this second reference. Prior live-replay notes below are historical.

## Native Effort Profile Contract Started

The prior 5c6ec3c8c commit is pushed; all its normal push checks passed.
New `motion_matching/native_effort_profile.py` explicitly converts the native
highest-power-first absolute-second sextic to the existing shared evaluator's
lowest-power-first representation, and rotates only the first three world
forces into the native hip base. It owns copies of coefficients/rotation and
does not normalize by replay horizon. Five tests progressed from missing-module
red to green; Ruff lint/format passed. It is a P2 building block, not yet wired
into the qualified native replay runner or a complete candidate package.

Second native tolerance run session 88805 is still live at this checkpoint.
Latest worker PID 81844 CPU=519.84 s and its new MAT file was being written
(891,468,896 bytes observed). Do not consume a partially written artifact or
restart on the basis of its size. Await terminal success, then export and
compare as described below. No error output has been received.

## Independent Native Cartesian Samples and Second Tolerance Run

ControlTower session 16047 completed, exit zero. The marker diagnostic now
also projects the independently exported R2025b KinematicsSolver transforms
directly; its reference side does not call Pinocchio FK at those six states.
Maximum marker distances at t approximately 0, .4, .6, .7, .75, .8 seconds:
1.75e-15, 1.74e-9, 5.72e-8, 6.49e-7, 2.13e-6, 6.17e-6 metres.
Receipt: `native_evidence/native_independent_marker_difference_9967.json`.
All-clock shared-FK metrics and the six independent samples have distinct
scope in that receipt. This does not establish full-swing C3D fitting.

Second cold R2025b tolerance run is LIVE at this checkpoint: exec session
88805, DeskComputer launcher PID 94352 and worker PID 81844. Existing PID
73720 is unrelated. Command is the same diagnostic with fourth argument 0.1:
RelTol=1e-11, AbsTol=1e-13, MaxStep=0.0001. Log/output stem is
`C:/Users/diete/SimscapeTour9921/native-tight-replay-9967-02`.
No success is claimed until that handle exits successfully. Poll it rather
than starting a duplicate. The helper now accepts a positive finite scale
at most one; its default preserves the first tight run's settings.

Next: after successful terminal exit, export this second MAT with the same
`export_native_pose_samples(...,spec_path,true)` invocation to a new -02 JSON.
Compare identical polynomial inputs against it and quantify native tolerance
convergence in marker metres. Preserve the prior references and all settings.

## Cartesian Trajectory Difference Through 0.80 Seconds

ControlTower diagnostic session 67568 completed with exit code zero. Using
the fixed 25 seed attachments, the qualified native-derived FK map projected
both trajectories at all 68,205 identical native timestamps. Maximum marker
distance is 6.17217e-6 m (0.00617217 mm); time-weighted RMS is 4.66791e-7 m
(0.000466791 mm). Receipt: `native_evidence/native_marker_difference_9967.json`.
This is engine-trajectory Cartesian agreement through 0.80 s, not a C3D fit
score or independent native Cartesian output verification. The shared FK
map has separate six-pose native verification; do not conflate the scopes.

New shared `motion_matching/marker_projection.py` follows a red missing-module
test, then six passing tests for rigid transforms, attachment order and invalid
offsets/frames. Ruff lint and formatting passed. Reproduction runner is
`native_evidence/reproduction/check_continuous_marker_difference.py`; its CLI
requires module/projection/spec/seed/reference/replay/output paths, retains
input hashes, rejects an existing output and requires identical sample clocks.
It time-weights squared error to avoid bias from adaptive native sample density.

Next: independently validate native Cartesian outputs and one further native
tolerance level, then qualify the reusable matching adapter and URDF plus
closure/actuation sidecar round trip. Do not silently replace the closed model
with a URDF tree. Full observed-swing optimization and independent native
acceptance remain outstanding. No diagnostic process remains live.

## Tight Native Replay Completed

The replay below finished with exit code zero, as did export session 73644
and ControlTower replay session 61132. None of these jobs remains live.
The subsequent live-process notes are historical. Native report is
`native_evidence/native_tight_replay_report_9967.json`; Pinocchio result is
`native_evidence/native_continuous_tight_reference_9967.json`.

Loaded R2025b defaults were ode23t, RelTol=1e-3, AbsTol=1e-5, MaxStep=0.001.
Tight replay used 1e-10, 1e-12, 0.0001 respectively and took 169.087 s
inside the replay wrapper. Export produced 68,205 raw native samples through
0.80 s. On this reference, Pinocchio integration took 3.825 s and maximum
q/qd discrepancy dropped to 6.21233e-5 / 0.0673489, from 0.671626 / 491.816.
These are mixed-coordinate maxima, not marker metres. This strongly supports
a numerical contribution to the earlier discrepancy, but does not alone prove
historical solver settings or final continuous equivalence. Marker metrics and
another native tolerance-convergence run remain required.

Raw tight MAT (904,020,813 bytes) remains on DeskComputer at the stem below.
`native-tight-reference-9967-01.json` is preserved there, in the local
`simscape-tour-checkpoints` archive, and ControlTower's user directory.
Full Pinocchio output remains at
`C:/Users/diete/native-continuous-tight-reference-800ms-9967-01.json` on
ControlTower; compact summary is archived locally and committed. Input hashes
are included in the compact receipt. Do not overwrite these evidence files.

## Tight Native Replay Launch History

September 12 UTC: isolated DeskComputer R2025b replay launched using
`native_evidence/reproduction/replay_native_tolerance_diagnostic.m`.
It uses the shared forward wrapper with the saved 0.80 s coefficients,
geometry and initial targets, rtol 1e-10, atol 1e-12 and maximum step 0.0001 s.
No source model is saved. It records loaded model defaults separately from
requested settings; defaults are not asserted to be historical run settings.

Live handle at this checkpoint: exec session 65188; MATLAB worker PID 87304,
launcher PID 37060 on DeskComputer. Existing PID 73720 is not ours and must
not be interrupted. Latest process check showed worker CPU 114.95 s; no
output MAT yet. Poll the same handle/process before any restart.
Log/output stem: `C:/Users/diete/SimscapeTour9921/native-tight-replay-9967-01`.
Source MAT: `prefix-800ms-sextic-01/final_native_replay.mat` in that directory.
Runtime root: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime`.

Next after successful exit: retain MAT and companion report, run
`export_native_pose_samples` with full_trajectory=true and the same native
geometry spec, then compare Pinocchio on the exported native clock. An output
file alone is not proof of successful terminal execution. The new diagnostic
is not an optimization run and cannot qualify full-swing tracking.

## Continuous Integration Convergence: September 12 UTC

The requested OpenSim planning check-in has been delivered. OpenSim code
implementation remains unstarted. Pinocchio continuation completed three
additional native ControlTower runs, all terminal with exit code zero.
Receipt: `native_evidence/native_continuous_convergence_9967.json`, including
input hashes, artifact hashes, solver settings and per-coordinate errors.
Raw JSON is preserved in `C:/Users/diete/Repositories/simscape-tour-checkpoints`
and `C:/Users/diete` on ControlTower, using the receipt's exact filenames.

Reducing maximum step from 0.001 to 0.00025 s and tightening rtol/atol from
1e-9/1e-11 to 1e-11/1e-13 changes the 0.5992645 s Pinocchio trajectory by
at most 6.84e-10 in q and 1.33e-8 in qd. Native-reference differences remain
0.00278474 in q and 0.0699314 in qd. These are mixed generalized-coordinate
maxima, not marker metres or a single angular unit.

Both integrations also reach 0.80 s. Baseline versus tight differences are
1.46e-7 in q and 1.62e-4 in qd, whereas tight versus saved native reference
differences grow to 0.671626 and 491.816 respectively. Tight integration takes
3.71 s, with maximum grip pose residual 1.24e-11 and velocity residual
1.70e-9. This establishes continuous execution and numerical self-convergence
at these settings, NOT native trajectory parity or a swing match.

Next action: inspect the saved native replay's actual solver configuration,
coefficient provenance and sampling; independently replay identical inputs
in explicit R2025b with tighter solver tolerances. Compare marker errors and
both engines' step convergence. Do not optimize against this unresolved
reference discrepancy or attribute it to a physical mismatch without evidence.
The default `capture_fit_sim_options` delegates solver settings to
`default_sim_options`; its source alone does not establish saved-run settings.

## Program Expansion and First Continuous Diagnostic

OpenSim is now part of the working program under
[Epic #10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003).
A parallel planning agent committed/pushed its detailed seven-stage epic and
handoff as 0517dea90 in `Worktrees/UpstreamDrift-opensim-10003`, branch
`docs/10003-opensim-matching-epic`. OpenSim implementation has not started;
the requested user check-in is the next program milestone. Pinocchio's
analogous lower-agent plan is `PINOCCHIO_EXECUTION_PLAN.md` in this directory.

The shared `continuous_forward.py` helper was developed red-to-green with
seven unit tests, including analytic nonconstant sextic forcing, absolute
time, invalid clocks and invalid derivatives. It integrates one explicit
Euclidean state equation from t=0 without target injection or feedback.
Native adapters remain responsible for physical constraints and validation.

Actual ControlTower Pinocchio first-prefix run completed in 0.739 s for
9002 derivative evaluations, ending at the last native sample before 0.60 s:
0.5992645005689832 s. Its maximum grip pose residual is 3.68e-11 and velocity
residual 1.15e-10. Maximum coordinate difference from the saved native replay
is 0.002785 and rate difference 0.06994 (mixed coordinate units; inspect
per-coordinate values). This is preliminary evidence, not an accepted
continuous-parity result or an end-to-end optimizer speedup benchmark.

Raw result: `simscape-tour-checkpoints/native-continuous-600ms-9967-01.json`
locally and `ControlTower:C:/Users/diete/native-continuous-600ms-9967-01.json`.
Compact summary: `native_evidence/native_continuous_prefix_diagnostic_9967.json`.
Reference: `native-rollout-reference-9967-01.json` in the local archive and on
both remote hosts (under SimscapeTour9921 on DeskComputer, user root on
ControlTower). It preserves exact native time/q/qd and saved polynomial
coefficients, exported by a fresh R2025b process from the existing MAT replay.
No new optimized or tighter-tolerance native replay has been run yet.

Reproducer: `check_native_continuous_rollout.py` with module/integrator/spec/
fixture/output paths and `--duration 0.6`. Source identities are in the raw
receipt. ControlTower's isolated Pin environment now additionally has SciPy
1.18.1. Next: derive marker errors, perform step/tolerance convergence, and
compare a fresh tighter-tolerance R2025b baseline before extending the horizon.
All native export and first-prefix diagnostic jobs from this checkpoint are
terminal; do not restart them as if still active.

## Current Result: Acceleration Parity Repaired and Verified

The failure below is now resolved for the tested states and inputs. The port
added joints in breadth-first order, interleaving left/right subtrees. This
violated Pinocchio's compact subtree indexing assumption: even with positive
individual body inertias, CRBA produced a negative mass-matrix eigenvalue
(-0.41694). The constrained stationary mobility was also indefinite.
The depth-first requirement is explicit in the
[Pinocchio Model API](https://docs.ros.org/en/ros2_packages/rolling/api/pinocchio/generated/structpinocchio_1_1ModelTpl.html).

`NativePinocchioModel` now orders the input tree depth-first before adding
joints. It preserves coordinate identities, geometry, masses, inertias, effort
mapping and closure. Invalid/disconnected trees are rejected. New unit tests
failed before implementation and pass afterward (2 tests). No native physical
parameters were changed to achieve agreement.

Actual execution evidence after correction:

- DeskComputer R2025b ran baseline plus all 27 unit input cases, each at exactly
  the same assembled initial q and zero qd. All cases completed and include
  complete actuator-log audits. Raw case files are in
  `C:/Users/diete/SimscapeTour9921/native-input-pulses-9967-01` on DeskComputer,
  copied to local `simscape-tour-checkpoints/native-input-pulses-9967-01` and
  ControlTower `C:/Users/diete/native-input-pulses-9967-01`.
- Actual ControlTower Pinocchio stationary response comparison passes with
  `--require-parity`: baseline max discrepancy 5.10e-12 and maximum scaled
  response discrepancy 9.68e-12. Receipt: `native_input_pulse_parity_9967.json`.
- The same six moving-state acceleration tests through 0.80 s now pass.
  Maximum absolute discrepancy 3.63e-9, maximum scaled discrepancy 2.54e-10.
  Receipt: `native_dynamics_parity_9967.json`. The original geometry comparison
  still passes. These are mixed generalized-coordinate diagnostics; do not
  label a mixed maximum exclusively as rad/s^2 or m/s^2.
- Corrected mass matrix minimum eigenvalue is positive (2.259e-6). Constrained
  mobility minimum eigenvalue is -1.79e-13, consistent with roundoff at the
  constrained null directions. `native_mass_matrix_parity_9967.json` records
  the dense KKT diagnostic and component inertias; it is not itself a full
  native mass-matrix measurement.

Reproduction: `export_native_input_pulses.m`, `check_native_input_pulses.py`,
and `diagnose_pinocchio_pulse_solver.py` under `native_evidence/reproduction`.
The pulse checker requires module/spec/cases/output paths and accepts
`--require-parity` to fail if agreement or common-state conditions are violated.
All native checks used production source copied to
`ControlTower:C:/Users/diete/native_model_9967_dfs.py`. The pulse receipt records
its hash and all 28 native case hashes. Raw historical failed receipts remain
in `simscape-tour-checkpoints`; do not mistake them for the latest results.

Next: implement and verify continuous forward integration with a single initial
state and native polynomial effort mapping. Compare native time histories,
closure residuals and solver-step convergence through transition, then extend
to the full observed swing. Preserve no-feedback/no-target-reset acceptance.
Pinocchio is now qualified for these sampled acceleration tests, not yet for
continuous-rollout equivalence or full-swing matching. The portable native JSON
retains the closure; plain URDF alone cannot carry the entire closed-loop
execution contract. Export a tree plus explicit closure/actuation sidecar when
adding interoperability, and test each engine's reconstruction independently.

## Historical Failure: Acceleration Parity

The next actual native check found a material dynamics mismatch. Do not run
optimization on this Pinocchio model as a native-equivalent oracle yet.
`native_evidence/native_dynamics_check_9967.json` preserves the failure:
maximum absolute generalized acceleration discrepancies at the six samples
are approximately 54.49, 21.24, 277.24, 396.25, 13267.89 and 22555.82
(translation channels in m/s^2, rotation channels in rad/s^2; do not present
the combined maximum as a single physical unit). FK still passes.

Evidence that narrows the issue:

- All 27 native actuator logs are present. Existing
  `audit_golf_actuator_torques` agrees with the polynomial efforts after rotating
  the three world forces into hip-base axes: max force error 3.41e-13 N,
  max torque error 2.84e-14 Nm. This verifies the existing log audit, not an
  independent virtual-work/pulse qualification of every actuator route.
- Full primitive audit (including unprefixed Revolute parameter names) records
  zero stiffness/damping, disabled limits, InputTorque and ComputedMotion in
  `native_passive_joint_audit_9967.json`. This is uncompiled inventory evidence.
- Native total mass 77.60581783574678 kg versus Pinocchio 77.60581783574676 kg.
  Native initial world COM [0.9366603077856251, 0.055828222965153544,
  1.3489299522131155] matches Pinocchio within about 6e-15 m.
- Native q/qd satisfy the Pinocchio grip pose/velocity constraint to numerical
  precision. J times the acceleration difference is around 1e-9 or smaller:
  the discrepancy is predominantly within the allowed motion, not a gross
  closure violation. Do not infer that all constraint-force conventions pass.
- Native finite-difference checks broadly support the logged derivatives
  before transition (sampled qdd discrepancies around 0.02-0.03 rad/s^2 versus
  the much larger Pinocchio mismatch). Rapid-transition finite differences
  are less accurate and are diagnostic only.
- The inertia-signature diagnostic matches 10 of 12 attached sensor groups
  to native COM/inertia signatures. The two hand sensors do not match the
  fully weld-collapsed groups used in this diagnostic. Sensor extent can stop
  at joints while the Pinocchio tree aggregates welded bodies; this is not
  yet proof of an inertia error. Resolve sensor routing and extent before
  drawing a conclusion. Symmetric groups have duplicate numeric signatures.

Reproduction files are `export_native_pose_samples.m` (optional fourth
argument: geometry specification), `check_native_pose_samples.py` with
`--check-accelerations`, and `check_native_inertia_samples.py`. All under
`native_evidence/reproduction`. Native R2025b exports 01, 02 and 03 completed;
03 fixes initial-sample extraction for timeseries whose time axis is last.
The tracked dynamics fixture comes from native export 03. Raw executed files
remain in local `simscape-tour-checkpoints`; the acceleration receipt hashes
refer to raw export 01 and the then-executed module, not reformatted JSON.
Both exports represent the same saved candidate states; 03 adds diagnostics.
Native log: `DeskComputer:C:/Users/diete/SimscapeTour9921/native-dynamics-samples-9967-03.log`.
Pinocchio diagnostic intentionally exits nonzero because parity fails.

Next decisive experiment: native zero-effort and 27 single-channel force/torque
pulses at the same initial pose with zero rates. Export the actual assembled
q/qd and qdd for every pulse. Compare baseline-subtracted acceleration response
columns with Pinocchio to isolate inertia/actuation from gravity and velocity
bias. Reuse R2025b `simulate_with_coefficients`, native initial-state overrides,
the actuator audit and explicit SI extraction. Keep polynomial and primitive
effort frames distinct. If the response matrix differs, inspect actuator
virtual work and body-group inertias before time integration. If it matches,
isolate gravity and velocity-dependent bias next. No inverse dynamics is
required for this experiment. Do not change physics to fit the observed
accelerations without locating and testing the discrepancy.

## Six Native Poses Verified Through Transition

The new `export_native_pose_samples.m` diagnostic ran successfully in a fresh
DeskComputer MATLAB R2025b process. It loaded the saved 0.80 s native replay,
selected actual raw states near 0, 0.4, 0.6, 0.7, 0.75 and 0.8 seconds, and
queried all schema frame poses using the native KinematicsSolver. It did not
interpolate coordinates or save the source model.

`check_native_pose_samples.py` then executed the actual production Pinocchio
module on ControlTower. All six comparisons passed: maximum position component
error 1.333e-15 m and maximum rotation-matrix component error 3.442e-15.
The fixture and receipt are `native_evidence/native_pose_samples_9967.json`
and `native_evidence/native_multipose_check_9967.json`. The receipt hashes refer
to the raw executed files, preserved under local `simscape-tour-checkpoints`
as `native-pose-samples-9967-01.json`, `native-multipose-check-9967-01.json`,
and `native_geometry_spec_9967.json`; tracked JSON may differ in formatting.

Native log: `DeskComputer:C:/Users/diete/SimscapeTour9921/native-pose-samples-9967-01.log`.
Both MATLAB and Pinocchio diagnostic exit codes were zero. The native query
uses the saved replay's arm geometry and all exact measured joint coordinates.
This supports multi-pose geometry/coordinate parity, not effort, inertia or
continuous dynamics equivalence. Next: inspect native upstream effort routing
and active passive-joint parameters, then compare same-state qdd under force
pulses before integrating complete trajectories.

## Current Resume Point: 22:02 PDT Review

Read `REVIEW_AND_EXECUTION_HANDOFF_20260911_2200.md` first. It supersedes the
historical next-step lists below and reviews Gemini commit 58df3af67 plus its
actual DeskComputer runner. The purported multiple-shooting run was single
shooting and changed the warm-start polynomial time basis; it passed 0/5 gates.
Repair identity and acceptance before further long optimization runs.

The full native geometry now assembles in Pinocchio: all 27 scalar coordinates,
31 uncommented solids, native fixed transforms and a 6D right-hand closure.
Actual ControlTower execution matched the initial 25-marker native fixture
with RMS 2.895e-13 m; zero-effort/zero-velocity free fall passed with maximum
acceleration discrepancy 8.786e-13. Source, portable geometry and the historical
receipt are preserved under `native_evidence`. These results qualify initial
FK and an invariant only. Multi-pose FK, native effort mapping, damping/limits,
acceleration parity and constraint-consistent continuous integration remain.

Current targeted native conversion suite: 16 tests passed. Ruff check and
format check passed for the seven new/modified Python implementation and test
files. The full matching goal remains incomplete.

## Complete Native Solid Port Coverage and Fixed Transforms

All 31 uncommented solids now have connector mappings measured in R2025b on
ControlTower: 51 physical ports, with exact coverage checked against the source
inventory. The final batch used one frozen probe source, hashed inventory and
hashed block list; its terminal exit code is zero. Results and per-file hashes
are recorded in `native_solid_port_bindings_20260911.json`. Raw probes, frozen
source, native log and start/finish receipts are archived at
`C:/Users/diete/Repositories/simscape-tour-checkpoints/native-solid-ports-20260911-02`
and `ControlTower:C:/Users/diete/native-solid-ports-20260911-02`.

`bind_solid_ports` compares separately measured named-frame and physical-port
poses, retaining aliases if frames coincide. It rejects unmatched ports and
requires original-layout R2025b evidence. The receipt additionally checks that
every expected solid and every physical connector was covered. It does not
certify inertia or dynamics. A native upper-arm connector fixture is committed.

The native probe now supports cylinders/spheres with or without custom frames.
The first exploratory batch stopped at cylinder 14 (no custom frame); that
terminal failure was repaired, then all 31 solids were rerun in the final frozen
batch. Do not resume either batch: both are terminal and complete for their
documented scope. The final batch is the authoritative connector evidence.

`rigid_transform` converts all 14 exported rigid-transform records (13 without
commented ancestry). It preserves intrinsic follower-axis versus extrinsic
base-axis sequence order, as specified by the MathWorks Rigid Transform block.
The targeted graph/solid/transform suite has 12 passing tests; Ruff passes.
Fixed-transform conversion has unit/source-documentation evidence; it still
needs full-model native FK comparison.

Next assembly decision: test cutting the native
`GolfSwing3D_Kinetic/Grip/RightHandOnClubForce` weld to form the Pinocchio tree,
then reimpose that exact six-dimensional closure. Its hand-side standoff has
0.01 kg native mass, avoiding an invented massless leaf for the actuated wrist.
Prove that the cut removes the loop without losing any of the 27 joint
primitives. The native hip is Bushing (Px, Py, Pz, Rx, Ry, Rz), not quaternion
Six-DOF; preserve its translation/rotation and virtual-work conventions.
Assemble fixed frames/inertias first, check loop consistency and native FK,
then implement constraint-consistent forward integration and pulse parity.

## Solid Properties and First Native Frame Parity

`native_solids.py` now converts native CalculateFromGeometry cylinders and
spheres, using the existing shared primitive-inertia API. It explicitly handles
mass versus density, SI conversion, nonnegative mass (including zero-mass visual
solids), and reference-axis custom frames. Unsupported modes, unresolved active
values and degenerate axes fail instead of receiving surrogate defaults.

All 42 solid records, including commented descendants, parse. The 31 solids
without commented ancestors sum to 77.60581783574676 kg and define 41 custom
frames. This sum has not yet been verified against a native assembled-system
inertia sensor, and is not a claim about independently moving body count.

An independent ControlTower R2025b fixture reconstructed the source LUpperArm
cylinder and queried its first custom frame through KinematicsSolver at a zero
revolute angle. Native translation matches Python exactly; rotation-matrix max
difference is 2.220446049250313e-16. The fixture and its input-inventory/output
hashes are committed at
`tests/fixtures/motion_matching/native_left_upper_arm_reference.json`.
The reproducible native probe is
`motion_matching/tests/export_native_solid_probe.m` under the MATLAB tree.
Native log: `ControlTower:C:/Users/diete/native-solid-probe-green3-9967.log`.
Raw output: `ControlTower:C:/Users/diete/native-solid-probe-9967.json`, also copied
to the local `simscape-tour-checkpoints` directory. Native exit code was zero.
The combined graph/solid suite now has eight passing tests; Ruff passes.

Native API details established by execution: custom-frame lookup uses its display
name (e.g. `Top of Left Arm`), not serialized ID `Frame1`; KinematicsSolver refuses
queries between rigidly connected frames, so the isolated fixture uses a zero-angle
revolute joint. The fixture validates the first custom frame only, not all 41
frames or native inertia values. Do not broaden this claim in the handoff.

Next: convert the remaining rigid-transform rotation sequences and native joint
primitives; qualify the mapping between solid custom-frame names and physical
wire endpoints; assemble the complete tree plus grip constraint. Preserve native
Bushing Joint force/rotation conventions and both forearm rotation joints. Then
compare full native poses before acceleration/rollout parity.

## Completed Inventory and Connection Graph

Update after the initial checkpoint: both native export jobs completed with
exit code 0. Version 2 adds library references and stable physical endpoints,
tested with an actual R2025b rigid-transform/revolute-joint connection.
Both native tests pass. The Python reader has three passing contract tests
covering subsystem traversal, joint-side separation, commented ancestors,
dangling endpoints and duplicate block paths. Ruff checks pass.

The real v2 export contains 3,679 blocks. The reader reconstructed 436 physical
wire nets containing 1,473 endpoints, including correct traversal from the
hip joint base through the nested subsystem to the upstream rigid transform.
These counts include physical signal nets, not just mechanical frames.
See `native_inventory_v2_receipt_20260911.json` for hashes and artifact locations.
The local inventory and wire-net graph are in `simscape-tour-checkpoints` outside
Git; the code and receipt are committed. There is no remaining inventory job
to wait for from this checkpoint.

The native library reference identifies the hip as a Bushing Joint. Preserve
its native primitive/effort conventions when mapping to a Pinocchio tree;
do not assume that a quaternion free joint has the same effort coordinates.
All 138 multibody blocks, including commented descendants, have native library
references. The earlier exporter field `source_block` was empty at runtime;
use `library_reference` and the preserved `BlockFunction` parameter instead.

Numeric expressions for the sampled physical properties resolved, including
dimensions, mass, density, COM and inertia parameters. This is not proof those
fields are the active inertia configuration: CalculateFromGeometry requires
deriving inertia using the original BasedOnType enum, geometry and units.
For example, the string `Mass` itself resolves to a workspace numeric value.
Always interpret enums from `expression`, never from `numeric_value`.

Next: reconstruct solid custom-frame transforms, explicit SI inertia and joint
primitive transforms from this native inventory. Compare the resulting FK with
native body-frame exports before implementing full constrained rollouts.

## Scope and Verified Results

The preceding review was progress: it corrected the completed 0.80 s result and
identified concrete engine discrepancies. The full matching goal remains active.
This checkpoint adds native infrastructure, not a completed golfer port.

- ControlTower: isolated Python 3.12.3 environment at
  `/home/dieterolson/simscape-pinocchio-9967/.venv` in WSL `ControlTower-Runner`.
  Pinocchio 4.1.0 imports successfully. Dependency versions are recorded in
  `pinocchio_controltower_20260911_requirements.txt`.
- Native Pinocchio test: `test_native_constraint_dynamics.py` passed (1 test,
  0.79 s). It applies six independent forces/moments to a free body and verifies
  zero acceleration under a rigid world closure versus nonzero unconstrained
  acceleration. This is a capability test, not golfer topology or rollout parity.
- Pinocchio 4.1 requires `initConstraintDynamics(model, data, models, datas)`.
  The three-argument invocation failed in the installed native library; retain
  this evidence when adapting earlier examples or supporting other releases.
- R2025b native exporter test passed on DeskComputer. TDD began with an actual
  missing-function failure, then found serialization errors during development.
  The exporter records original expressions, resolved numeric values, units as
  separate parameter entries, block paths, source libraries, comment state and
  port connectivity. No unresolved parameter is substituted with a physical
  default. It does not compile variants or certify active topology.

## New Files and Commands

MATLAB source is under
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/`:
`shared/export_native_inventory.m` and `tests/test_export_native_inventory.m`.
These deliberate source files were explicitly staged because an existing broad
`motion_matching/` gitignore pattern otherwise hides them.

Run native tests in R2025b after adding those directories:

```matlab
results = runtests('test_export_native_inventory');
assertSuccess(results);
```

Run the Pinocchio capability test in the pinned environment, using the repo's
native-test lane (`-m live_simulation`). On ControlTower the standalone transferred
test was executed with:

```text
/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python -m pytest /mnt/c/Users/diete/test_native_constraint_dynamics_9967.py --confcutdir=/mnt/c/Users/diete -q
```

## Full Golf Inventory Job

Launched on DeskComputer at 21:00:29 PDT with explicit
`C:/Program Files/MATLAB/R2025b/bin/matlab.exe`. Process IDs observed:
94704 launcher and 82556 worker. Revalidate command line and creation time;
do not restart based on this document alone.

Log: `C:/Users/diete/SimscapeTour9921/native-inventory-green4.log`.
Expected result: `C:/Users/diete/SimscapeTour9921/native-inventory-20260911.json`.
It loads the existing runtime checkout model, assigns UpperArmLength and
LowerArmLength from `initial_velocity_seed_qualified_r2025b.json`, exports,
and closes without saving. The job is an uncompiled parameter inventory;
it does not replay the selected candidate's torques or dynamic state.
At this checkpoint the native test passed and the golf export remained running.

## Findings That Change the Port

1. The static SLX contains a Flexible Cylindrical Beam, but its parent Flexible
   Beam Model subsystem has `Commented=on`. Verify ancestry in the native
   inventory and compiled variant selection before treating the shaft as rigid.
2. Native `golf_kinematic_schema.json` has separate left and right forearm Rz
   coordinates, LFInput and RFInput, in addition to LEInput and REInput elbows.
   The proposed MuJoCo 19-joint internal chain omits those forearm rotations.
   Do not derive the Pinocchio port from that reduced topology without a proven
   reduction. There are 27 native effort channels, not necessarily 27 independent
   unconstrained degrees of freedom.
3. Existing shared Simscape-to-URDF conversion admits unevaluated expressions,
   skipped constraints and approximations. Its helpers also have Tools ownership
   notices. Reuse compatible infrastructure but do not edit mirrored files or
   silently accept approximations as a native physical export.

## Next Execution Steps

1. Poll the exact inventory job, collect its terminal exit and hash the export.
   Inspect numeric resolution failures for physical parameters and resolve them
   from the actual candidate workspace; distinguish enum strings from failures.
2. Reconstruct physical frame connectivity through subsystem ports and fixed
   transforms. Exclude commented ancestors and qualify variant activity natively.
   Export masses/COM/inertias and joint transforms in explicit SI units.
3. Preserve the two forearm joints, both grip frames and all base efforts in the
   canonical representation. Use a tree plus engine-level closure constraints,
   preserving virtual work under coordinate changes.
4. Build the Pinocchio model and perform same-pose FK against the existing native
   body-frame exports before force-pulse or full-rollout parity tests.
5. Keep all-source and candidate identities separate from this capability probe.
   Only a full continuous native rollout can qualify the final swing.
