# Simscape Matching Agent Resume Prompt

Read AGENTS.md, CLAUDE.md, docs/development/HANDOFF.md and the latest section of
PROGRESS_REVIEW_20260917.md in this directory. The current goal is incomplete.

Do not repeat the resolved source hunt: native replay reproduces run102 markers
exactly, and the restored fitter passes its startup parity audit. All original
inputs and missing historical sources are hash-matched and archived under
native_evidence. The qualified candidate still covers only 0–0.85 s and is
rejected (terminal RMS 40.30 mm versus 35 mm). Zero active bounds or a closed issue
is not convergence.

1. Recheck leases, current commits and remote jobs. The two audits recorded in
   restoration_audit_20260917 are terminal. Preserve their outputs and runtimes.
2. Coordinate and pin the finite-weld correction from #10260/#10263. Head
   0db3c295a passes the native pose directional audit; current local native_model
   still has the old provider. Verify downstream trajectory derivatives,
   node-chart retraction and acceleration constraints separately. Do not confuse
   the pose log derivative with the velocity constraint Jacobian.
3. Restore only needed constrained-pose helpers and tests from the verified
   source archive. Fit an articulated terminal pose with the original geometry,
   weld, joint limits and all valid markers. Save the witness/residuals. Failed
   local starts do not prove infeasibility; relaxed cluster bounds are not witnesses.
4. Inspect equality-projected objective gradients and test predicted versus actual
   changes by uninterrupted replay. If restricted high-order controls cannot improve
   transition, test a small justified release of lower sextic coefficients with
   early-motion retention. Record budgets, accepted steps and failed directions.
5. Execute a bounded 0.90 s trial, using one global sextic per actuator, the original
   initial state and absolute clock. Keep polynomial basis duration independent of
   replay duration. Validate without node resets and independently in MATLAB R2025b.
   Decide the next horizon from results; do not launch a blind optimization ladder.

Use TDD, DbC, LoD and DRY. Reuse shared providers; make the fitting command
configuration-driven rather than copying another numbered script. Never drop
markers, add anatomy, soften the weld or relax gates silently. Preserve rejected
results and exact hashes/commands; commit incremental tested checkpoints.

The product replay path already loads the hash-verified Simscape manifest and
plays at source time. Epic #10285 still needs a shared catalog, cylinder/multi-angle
GUI, qualified effort display and explicit MATLAB rerun actions. Epic #10286 needs
native constrained counterfactual/reaction qualification. Keep these separate
from matching progress. Full completion requires all 654 frames / 1.813888889 s,
global sextic forward dynamics and independent R2025b acceptance.
