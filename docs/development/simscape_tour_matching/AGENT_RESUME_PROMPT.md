# Simscape Matching Agent Resume Prompt

Continue the tour-average forward-dynamics matching epic #9921. Read AGENTS.md,
CLAUDE.md, docs/development/HANDOFF.md and the latest PROGRESS_REVIEW_20260917.md.
Use MATLAB R2025b explicitly. Recheck issue leases, current commits and live jobs
before execution. Preserve other agents' work. The goal is incomplete.

## Verified Starting Point

Run102 is a rejected 0–0.85 s / 307-sample prefix, not the full 654-frame /
1.813888889 s swing. Actual R2025b metrics: whole-motion RMS 20.26736 mm,
early RMS 9.99539 mm, terminal RMS 40.30135 mm (35 mm gate), club RMS 8.38853 mm,
yaw error 0.60998%. Pinocchio–Simscape maximum marker difference is 0.060521 mm.
Candidate SHA: ee4a908737f4cf3065bb6e64a056dfab71926c43ac40c0e85574fc8531576c82.

Replay source restoration reproduces markers exactly; fitter startup parity
passes. Do not repeat the source hunt. The shared constrained_marker_pose solver
is restored and tested. Twenty-four local terminal solves reach 39.76310 mm best
within the yaw allowance; unrestricted yaw reaches 39.20167 mm but fails yaw.
All fail 35 mm. These are local results, not proof of global infeasibility.
Head and left upper-arm residuals dominate. Sampled native chart/retraction checks
pass with the corrected finite-weld provider. No new dynamic fit was produced.

## Sequential Execution and Decision Gates

1. **Reproduce the checkpoint.** Verify evidence archive hashes and exact inputs
   using native_evidence/terminal_feasibility_20260917/README.md. Pin and consume
   the owner's #10263 finite-weld correction (audited commit 0db3c295a7f668789afbb4666f60912294f81cce).
   Coordinate #10260; do not independently rewrite that provider. Keep the pose-log
   derivative separate from the velocity Jacobian used in acceleration constraints.
   Run affected tests and baseline replay before trusting a changed provider.
2. **Diagnose representability before a large torque search.** Use the restored
   shared pose solver to save a fixed-geometry residual curve at 0, 0.3, 0.6, 0.7,
   0.8, 0.85, 0.9 s and selected later downswing/impact/follow-through frames.
   Use actual reference timestamps and marker masks, several starts, original weld,
   declared numerical bounds and yaw gates. Save coordinates, per-marker vectors,
   closure, yaw, convergence and active bounds. Overlay targets and predictions.
   A nonconverged solve is inconclusive; a passing static pose is not dynamic proof.
3. **Investigate the persistent residuals.** Check C3D units, labels, rigid cluster
   assignment, frame transforms, time alignment and attachment coordinates against
   authoritative model/capture exports. Separate constant attachment bias from
   missing articulation and soft-tissue motion. Evaluate segment-length and fixed
   attachment sensitivities across the capture. Fit only physically justified,
   bounded parameters shared across frames; validate on held-out frames. Do not
   give each frame its own marker offsets or drop difficult markers.
4. **Make a controlled model decision.** If calibration/length changes materially
   improve held-out errors, create a separately named model variant with explicit
   parameter deltas, bounds, provenance and consistent body frames, inertia and
   loop anchors. Export from the authoritative Simscape model, then verify FK,
   closure and forward dynamics in the other engines. Preserve original baseline.
   If geometry is not the explanation, report the evidence and investigate model
   constraints/articulation explicitly; never silently add anatomy or loosen gates.
   Anatomical limit acceptance is currently unqualified: numerical boxes are not
   physical joint limits. Obtain authoritative limits before claiming it.
5. **Qualify the control search.** At a fixed chosen model, verify trajectory
   directional sensitivities and equality-projected objective gradients. Compare
   predicted and actual objective changes under small coefficient perturbations by
   uninterrupted replay. Record scaled gradient, constraint rank, accepted steps,
   rejection reasons, evaluation budgets and wall time. Budget exhaustion is not
   convergence. Release selected lower-order coefficients if evidence shows the
   high-order-only search is too restrictive, while retaining early-motion gates.
6. **Extend progressively with one global sextic per actuator.** Start with a bounded
   0.90 s trial from the original initial state and absolute clock. Keep polynomial
   basis duration independent of replay horizon. Use normalized time and a
   well-conditioned polynomial basis with tested conversion to MATLAB coefficients.
   A stitched cubic profile may initialize coefficients, but regression of torques
   alone does not qualify the resulting motion. Reoptimize the sextic through
   forward replay. Multiple-shooting states are solver variables only: acceptance
   always uses uninterrupted dynamics with no state resets or tracking controller.
   Decide each subsequent horizon from recorded results until all 654 frames pass.
7. **Validate and package each candidate.** Run independent MATLAB R2025b replay
   with explicit executable path; preserve all existing marker, club, yaw, closure,
   early-motion and engine-agreement gates. Export actual applied actuator efforts
   and verify their channel mapping/units; run102 stored tau is nonfinite and must
   remain visibly unavailable. Include per-frame/per-marker errors and multi-angle
   cylinder animation from actual saved states, with rejected/accepted status.

## Implementation and Handoff Contract

Use TDD for behavior, DbC for shapes/units/time/masks/finite inputs, LoD through
shared public providers, and DRY configuration-driven drivers. Do not copy another
numbered fitting script. Test failure paths as well as successful fixtures. Every
run gets a new output directory, config, source/input hashes, host/environment,
command, stdout/stderr, exit code, timing, metrics and decision. Checkpoint during
long runs; restart from immutable artifacts. Never run historical destructive
run_full_102.sh against existing output. Commit small tested increments and update
HANDOFF.md, DEVELOPMENT_LOG.md and the latest review with exact resume commands.

## Product Delivery Alongside Matching

Epic #10285 already has verified saved-run manifest loading and source-time replay.
Use the existing tour_matching_viewer and SimulationDataStore; do not build another
viewer. Deliver a discoverable saved-run catalog, accepted/rejected and missing-data
badges, persistent multi-angle cylinder views, marker residual overlays, qualified
effort plots and explicit inspect/rerun actions in MATLAB R2025b. Test corrupt or
mismatched manifests, missing efforts, end/pause/seek/restart and retained camera.
Visually inspect with normal fonts; the offscreen font issue is not polished UX.

Epic #10286 is separate: qualify native constrained ZTCF/ZVCF and reaction wrenches
against the AffineDrift definitions and WSCG assets. Never use unconstrained ABA or
fabricated qdd-times-length arrows for a welded model. Recompute constraints and
reactions for each declared intervention. Do not claim all-engine equivalence from
one engine or saved playback. Keep this work from blocking the geometry diagnosis.

Stop at a tested, committed checkpoint if handed off. State remaining failures and
exact next command. Full completion requires accepted full-horizon global-sextic
forward dynamics in R2025b, reproducible cross-engine evidence and usable saved-run
inspection; a static witness, attractive animation or closed issue is insufficient.
