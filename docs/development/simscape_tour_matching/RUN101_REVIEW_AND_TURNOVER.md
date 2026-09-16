# Run 101 Review and Completion Turnover

## Authoritative Status

Reviewed 2026-09-15 at source checkpoint `ca750f7d7`. This review supersedes
the claim that all fitting gates are certified. The goal remains incomplete.
Run 101 is a useful, rejected 0–0.85 s prefix, not a full-swing solution.
Its returned report says `accepted=false`, `optimizer_converged=false`,
`improved_uninterrupted=false`, and `Iteration limit reached`.
Yaw improved substantially, but terminal RMS increased relative to run 100.
Do not discard that tradeoff or rename a failed gate as a plateau/pass.

## Independently Checked Evidence

All paths below are relative to this directory. Start from
`native_evidence/two_window_fit_9967_101/`:

- `returned-candidate.json`, recorded canonical identity
  `b505d8b6bd93bd84fb7aad5d5a0e3df257ebb37827a6ce7c7a93333f367bd7db`.
- `returned-replay.npz`: actual time, native q/qd, markers, target and valid mask.
- `qualified_candidate_replay.mat`: actual MATLAB prediction, q/qd/qdd,
  tau and omega; independently readable with SciPy.
- `qualified_candidate_replay.json`, `returned.json`, `receipt.json` and launchers.

The reviewer recomputed these values directly from the archived NPZ and MAT,
not just the prose. The clocks match exactly, with 307 samples:

| Quantity                                          |      MATLAB R2025b Result | Status                    |
| ------------------------------------------------- | ------------------------: | ------------------------- |
| Overall RMS                                       |               20.26494 mm | Pass, <=25 mm             |
| Early RMS                                         |                9.99517 mm | Pass, <=12 mm             |
| Terminal RMS                                      |               40.31153 mm | **Fail, >35 mm**          |
| Club cluster RMS                                  |                8.42248 mm | Pass, <=60 mm             |
| Pelvis yaw error                                  | 0.54345%, 0.29678 degrees | Pass, <5%                 |
| Maximum Euclidean marker discrepancy vs Pinocchio |              0.0604935 mm | Measured prefix agreement |

Settings: MATLAB R2025b Update 5, ode15s, RelTol 1e-6, AbsTol 1e-9,
MaxStep 1/1440 s. Keep these explicit; do not fall back to model defaults.
The earlier geometry-seed lookup and duplicated force-rotation defects were
corrected. Preserve required calibrated geometry and world-force input convention.

Measured agreement supports using this prefix for the next fitting experiment.
It does not prove that all disagreement was exclusively truncation, nor establish
accuracy for future candidates or the full horizon. The nine-case run100 audit
uses its best Simscape result as its own reference; its zero self-difference is
not an independent refinement check. Its acceleration-agreement summary actually
lists state/rate differences. Do not describe that as an acceleration comparison.

Run101 shooting defect is 2.06997e-4 in scaled mixed state coordinates;
terminal segmented/replay RMS gap is 0.148264 mm. An uninterrupted replay has
no shooting resets, but that does not make its physical closure identically zero.
Evaluate closure separately and never label the scaled shooting norm in meters.

## What Dominates the Remaining Error

Computed from run101's saved Pinocchio terminal frame and valid mask:

| Marker    | Attached Body |     Error | Share of Terminal Squared Error |
| --------- | ------------- | --------: | ------------------------------: |
| HeadSide  | Hub           | 81.491 mm |                          16.34% |
| LUArmHigh | LS            | 79.105 mm |                          15.40% |
| HeadFront | Hub           | 78.320 mm |                          15.10% |
| HeadTop   | Hub           | 53.976 mm |                           7.17% |
| WaistLeft | Hip           | 52.271 mm |                           6.72% |

The three head markers contribute 38.61%; they share Hub with torso markers.
The club already fits well. These data favor a body/attachment feasibility audit
over another indiscriminate club or terminal-weight increase. They do not prove
the 35 mm gate is infeasible. Previous full-capture rigidity lower bounds are not
a substitute for a bound evaluated at this exact terminal frame and mask.

## Ordered Work Packages

1. **Freeze and verify the restart.** Preserve run101; check leases, local changes,
   live remote processes and source hashes before launch. Recompute metrics and
   verify coefficient order, force frame, geometry, original state and absolute
   clock. Use a fresh runtime/output directory. Run101's launcher still names
   runtime78: verify the executed source manifest rather than inferring source
   identity from that directory name. Never rerun its launcher verbatim: it
   targets the historical output and may remove it.
2. **Finish bounded numerical assurance.** At fixed run101 inputs, perform one
   tighter/half-step MATLAB comparison and an independently refined Pinocchio
   replay. Compare physical poses/rates, closure and markers, not Euler values
   alone. Retain existing applicable tolerances and clearly distinguish the
   prefix's measured 0.0605 mm agreement from stricter representation gates.
   If no material change occurs, stop auditing and proceed. If it does, locate
   first divergence before fitting. Save exact configuration and raw arrays.
3. **Diagnose terminal feasibility without simulation searches.** Reuse shared
   rigidity and constrained-pose providers. Compute per-body residual vectors,
   rigid-cluster lower bounds and a closure-constrained terminal pose fit using
   all valid markers, fixed attachments and original geometry. Report physical
   closure, yaw and residuals. This is only a kinematic lower-bound diagnostic,
   never an accepted forward-dynamics result. Investigate head/Hub and left-arm
   errors first. Do not remove markers or silently introduce a neck joint.
4. **Choose one controlled numerical intervention.** If the target is locally
   feasible, inspect the existing terminal/yaw Jacobian and feasible directions
   under closure and bounds. Run101 has 22 active control bounds and 11 active
   node bounds; recentering or selectively changing a justified bound is more
   informative than another large weight escalation. Keep yaw and early gates.
   If B4/B5/B6-only controls cannot produce useful directions, qualify selected
   lower-order sextic coefficients while retaining early-motion constraints.
   Change one factor per trial, predict the improvement, then compare actual
   original-state replay. Use a small trial budget before another 40-iteration
   job. Reject apparent progress that depends on node resets or loses other gates.
5. **Extend coverage deliberately.** After useful prefix progress, evaluate the
   unchanged global sextic over the next 0.05–0.10 s. Do not reset time or state;
   polynomial extrapolation beyond the old basis interval needs explicit checks.
   Add integrated shooting nodes only where conditioning requires them. Reuse
   the candidate as a warm start while optimizing globally, preserve early
   retention and independently replay in R2025b at each promoted checkpoint.
   If exact 35 mm prefix feasibility is demonstrably blocked by fixed anatomy,
   report that evidence and request a physical-model decision rather than spend
   indefinitely polishing a prefix or silently weakening acceptance.
6. **Complete acceptance.** All 654 samples through 1.813888889 s, original-state
   uninterrupted forward dynamics, continuous global degree-six inputs, valid
   masks, unchanged matching/effort/closure gates and independent R2025b replay.
   Save coefficients, geometry, hashes, raw states, per-marker error plots and a
   synchronized 3D C3D/model overlay. Native MuJoCo/Drake and alternate-coordinate
   equivalence remain separate gates; OpenSim follows its own epic. Do not claim
   those engines are qualified by this Simscape–Pinocchio comparison.

## Code and Test Review Limits

The shared `pelvis_yaw.py` now uses a two-component unit-direction residual and
actual directional derivative tests. This removes the false zero at reversal;
the squared objective can still have zero angular gradient at exact opposition,
so do not claim globally guaranteed convergence. The reviewer ran
`python -m pytest tests/unit/motion_matching/test_pelvis_yaw.py tests/unit/motion_matching/test_replay_regression.py -q --no-cov`:
7 passed, with existing deprecation warnings.

The missing-seed regression defines a loader inside the test rather than invoking
the production MATLAB path. Strengthen that integration test before claiming
the pipeline defect is protected against recurrence. For future yaw/fitter edits,
also run full-residual directional tests covering node columns, cache ordering,
invalid observations and SLSQP. Seven tests are not whole-engine qualification.

Use TDD, explicit finite/shape/unit/frame contracts, shared providers and narrow
dependencies. Preserve historical evidence. Update HANDOFF and DEVELOPMENT_LOG
at each checkpoint; commit incrementally. No simulation or optimizer was launched
by this review. Remote live-process status was not rechecked by the reviewer.

## Copy-Ready Prompt

Resume the tour-average C3D matching goal from
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native`, branch
`feat/9967-native-simscape-pinocchio`. Read AGENTS.md, CLAUDE.md,
docs/development/HANDOFF.md and this RUN101_REVIEW_AND_TURNOVER.md first.
Acquire the issue9967 lease and check for newer work before edits or execution.

Use returned101 as the yaw-qualified exploratory restart, not an accepted fit.
Simscape/Pinocchio prefix agreement is now measured at 0.0605 mm maximum, but
terminal RMS remains 40.31 mm and only 0.85 s is covered. Execute the ordered
work packages above: bounded refinement verification, per-body terminal feasibility,
one justified fitting intervention, then progressive horizon extension and full
R2025b acceptance. Preserve global sextic torques, original clock/state, all valid
markers, physical constraints and existing gates. Do not repeat weight escalation
or declare success from prefix RMS, an optimizer exit, or representation roundtrips.
Keep exact runtimes, artifacts, checks and next commands in turnover documents.
