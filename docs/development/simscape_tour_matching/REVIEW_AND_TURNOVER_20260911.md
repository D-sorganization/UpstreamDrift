# Model Matching Review and Turnover

Review snapshot: 2026-09-11 20:50 PDT. Inspected collaborator checkout at
`9bf2a4f8d`, source implementations, and live DeskComputer files/processes over SSH.
This supersedes the active-run status in AGENT_HANDOFF.md at that revision.
The overall matching goal remains incomplete. No simulation was launched or
collaborator source changed during this review.

## Verified Progress

| Candidate              | Horizon | Whole RMS | Terminal RMS | Club Terminal RMS | Status                    |
| ---------------------- | ------: | --------: | -----------: | ----------------: | ------------------------- |
| Run 06 Eval 79         |  0.75 s | 23.859 mm |    94.690 mm |        115.979 mm | Handoff reports 3/5 gates |
| Latest 0.80 s final    |  0.80 s | 44.042 mm |   200.541 mm |        274.539 mm | Native output: 2/5 gates  |
| Latest 0.80 s Eval 559 |  0.80 s | 43.228 mm |  Not audited |       Not audited | Best logged average only  |

The 0.75 s numbers above are from the committed handoff, not freshly recomputed
in this review. The 0.80 s numbers were read from the remote result itself;
see [remote receipt](review_800ms_remote_receipt_20260911.json).
Different horizons are not directly comparable as optimization progress.
At the same 0.80 s horizon, the documented seed baseline was 45.109 mm versus
44.042 mm final: approximately 2.4% average improvement. Yaw improved strongly,
but the full marker trajectory remains poor near the endpoint.

The 0.80 s job finished at approximately 17:54 PDT: 775 logged evaluations,
773 stage evaluations, `xtol` termination. There was no MATLAB fitting process
at inspection; only MATLABConnector.exe remained. Final early-retention and yaw
gates pass; whole-window, terminal and club gates fail. Final yaw difference is
-0.249 degrees; p95 marker error 93.801 mm; maximum 380.553 mm.
Optimizer convergence therefore does not mean motion acceptance.

## Assessment of the Approach

There is useful engineering progress: checkpointing, explicit degree-six inputs,
R2025b execution, candidate preservation and honest per-gate result fields.
There is limited motion progress through transition. The evidence does not
justify continuing the same weight-tuning/horizon-extension loop indefinitely.
Correct pelvis yaw alone does not determine arm, wrist and club positions.
The present result cannot identify a unique cause: sensitivity conditioning,
regularization, geometry/attachment feasibility, velocity error, and search
restrictions need controlled experiments before assigning blame.

Two distinct problems must be resolved: matching the capture in native Simscape,
and making other engines represent that same physical system. Faster simulation
of a different mechanism will not reliably accelerate the first problem.

## Concrete Code Findings

1. `src/engines/physics_engines/pinocchio/python/motion_matching/simulate.py`
   calls ordinary `pin.aba` at every RK4 stage. This path does not enforce the
   dual-arm closed loop. Its sub-100-ms performance statement is a target,
   not a benchmark of a qualified equivalent model.
2. `src/engines/physics_engines/drake/python/motion_matching/humanoid_urdf.py`
   explicitly resolves URDF's loop limitation by attaching the club to the
   right hand only. An engine-level closure constraint is still required;
   inspect the actual wrapper's routing before claiming it exists.
3. `src/engines/physics_engines/mujoco/_golf_swing_canonical_xml.py` claims an
   exact analogue but contains manually specified masses, damping, limits,
   dummy-body inertias and 19 internal motor channels. A weld declaration and
   matching names do not verify physical equivalence to native Simscape's
   27 effort channels. Audit free-base effort routing, evaluated inertias,
   constraint frames and contact behavior; prove any coordinate reduction.
4. `tests/cross_engine/test_four_engine_parity.py` uses a 20-ms benchmark,
   optional native availability and a baseline loader without enforced native
   provenance. That is insufficient full-swing certification. Require exact
   benchmark input/state identity and all four actual runtimes in an acceptance
   job; keep optional unit-test skips separate.
5. AGENT_HANDOFF.md still calls the completed 0.80 s job active. Historical
   sections also use stronger certification/record language than their gates
   support. Correct the active ledger without erasing historical evidence.

These are source findings, not results of fresh four-engine physics tests.

## Recommended Execution Order

1. Cold-replay and independently score Eval 79 and the 0.80 s final/559
   candidates. Keep accepted and exploratory lists separate. Do not advance
   the accepted horizon while transition/terminal gates fail.
2. On DeskComputer, diagnose the 0.70–0.80 s residual: per-body and per-marker
   errors, feasible native kinematic floor, velocity error, repeatability,
   scaled finite-difference stability and Jacobian singular values. Compare
   equal-budget objectives instead of escalating arbitrary weights.
3. On ControlTower, qualify a native physical export and Pinocchio constrained
   forward model. Reuse the existing shared Simscape converters after checking
   actual supported blocks; do not silently approximate unsupported physics.
   URDF is a tree artifact plus a versioned constraint/actuator/marker manifest,
   not the entire model. Transfer the same manifest to Drake and MuJoCo.
4. Keep one global sextic on a fixed physical clock. Use prefix continuation
   as an optimization aid. If single shooting stalls with reliable derivatives,
   use constrained multiple shooting with the same coefficients, feasible
   boundary states and enforced state continuity. Final evidence must be a
   continuous forward run from t0 with no target-state resets.
5. Benchmark the qualified fast engine, then use bounded multistart searches
   and independent perturbation batches. Revalidate candidate improvements and
   local sensitivities in R2025b before extending the surrogate trust region.

Stitched cubics may supply a discovery seed, but fitting their torque values
to a sextic is an approximation. Always refine the sextic against motion after
conversion. Do not require an inverse-dynamics solution for this workflow.

Pinocchio supports constrained forward dynamics; verify the installed version's
API and integration behavior against the [official project](https://github.com/stack-of-tasks/pinocchio)
and its [closed-loop example](https://github.com/stack-of-tasks/pinocchio/blob/devel/examples/simulation-lcaba.py).
MathWorks documents [URDF import and joint/frame mapping](https://www.mathworks.com/help/sm/ug/urdf-import.html);
that documentation does not certify this repository's reverse conversion.

## Locations and Ownership

- Collaborator source: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour`.
- Isolated review/Pinocchio branch: `feat/9967-native-simscape-pinocchio` in
  `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native`.
  At this checkpoint it contains review documents, not a completed native port.
- DeskComputer run: `C:/Users/diete/SimscapeTour9921/prefix-800ms-sextic-01`.
  Result: `first_prefix_fit.json`; native replay: `final_native_replay.mat`.
- Selected short candidate: DeskComputer
  `C:/Users/diete/SimscapeTour9921/candidates/candidate-run06-eval79-pkg/candidate_eval79_package.json`.
- Local visualization: `C:/Users/diete/.gemini/antigravity/brain/9efc50d9-1e1a-4d78-a883-cf8c5e9f7a59/simscape_matlab_matching_eval79.gif`.
  This illustrates the short candidate; it is not a full-swing or parity certificate.
- ControlTower SSH works; explicit R2025b executable exists. Default MATLAB on
  PATH is R2026a: do not use it. Linux distributions include ControlTower-Runner
  and OllamaServer; Pinocchio installation/availability is not yet qualified.
- Epics #9921 and #9964; existing #9967 lease must be rechecked before edits.

Use [the execution prompt](NEXT_AGENT_EXECUTION_PROMPT_20260911.md) to resume.
Preserve source/runtime ownership, candidate hashes and receipts at each step.
