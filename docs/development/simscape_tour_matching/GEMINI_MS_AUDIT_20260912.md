# Gemini Multiple-Shooting Evidence Review

Updated 2026-09-12 UTC. Read-only review of Gemini branch at 9dd45e476 and
DeskComputer runners copied into local simscape-tour-checkpoints/gemini-ms-audit-20260912.
manifest.json preserves byte identities. No Gemini source or running job changed.
The previous goal turn made progress by independently replaying the expanded fit;
this turn establishes why the alternative seed should not replace it.

## Findings

1. **Acceptance Is Inconsistent.** candidate_ms_080s_seeded_package.json says
   accepted=true, optimizer_converged=true and gates_passed="1/5". Whole,
   terminal, club and pelvis-yaw gates are false. multi_shooting_fit.py computes
   accepted from optimizer success, finite predictions and defect tolerance;
   it never requires marker-fit thresholds. This flag is solver feasibility,
   not swing acceptance. Keep these concepts separate in the API and package.
2. **No New Torque Solution.** Convert the seeded package's 27-by-7 Bernstein
   efforts with the existing bernstein_to_simscape and basis_duration_s=1.813889.
   Coordinate order matches native run03 exactly. Maximum absolute coefficient
   difference from run03 returned-candidate.json is 1.1368683772161603e-12.
   Its reported 31.7637 mm RMS is a replay of the existing seed to numerical
   conversion precision, not evidence of successful optimization improvement.
3. **Boundary State Is Misdated.** Both downloaded seeded and targeted runners
   simulate win_duration+0.001 and return sim_q[-1], sim_qd[-1]. Thus the node
   requested at 0.600 s receives the terminal state at 0.601 s, assuming normal
   completion to configured stop time. Return/interpolate q and qd at the exact
   requested boundary and assert the returned clock. Merely extending simulation
   to support interpolation is permissible; using its later final state is not.
4. **Candidate Identity Is Incomplete.** Packages omit model/capture hashes,
   initial q/qd, attachments and force-frame semantics. Do not import them as
   standalone native candidates by filling missing identity from guesswork.
   Seeded runner explicitly loads candidate_ct_9967_03.json coefficients but
   loads q/qd/geometry/attachments separately from initial_velocity_seed.
5. **Parity Claim Uses the Wrong Comparison.** Gemini walkthrough calls the
   difference between two target RMS values a cross-engine trajectory delta.
   That scalar difference does not measure pointwise engine agreement. Compare
   actual marker arrays on identical clock/masks and same candidate/model, then
   run a tighter-tolerance replay. Existing native baseline parity receipts remain
   valid for their declared scope; this walkthrough adds no such proof.

DeskComputer process query found no python.exe command matching run_ms_080s or
run_real_ms at inspection. This is a bounded process observation, not proof that
all remote work has stopped. Do not restart any job without checking its exact
command, output ownership and current process state.

## Bounded Repair Prompt for the Owning Agent

Preserve current outputs. First write red tests showing: finite but poor marker
fit cannot be accepted; a requested .600 boundary is not .601; zero-iteration
seed replay is not reported as improved; incomplete candidate provenance fails
import. Fix these through shared public contracts before another expensive solve.
Use one authoritative acceptance evaluator on the final unsegmented replay.
Report solver convergence separately from physical feasibility and fit acceptance.

Validate shooting node closure in position and velocity, and verify the assembled
state actually used by Simscape equals the requested node. Scale defects using
explicit coordinate/speed units before choosing tolerances. Avoid duplicated
conversion helpers; retain the existing conversion round-trip tests.

The targeted runner still uses ordinary finite differences and up to 135
variables (81 efforts plus 54 node states). Qualify derivative steps against
simulation noise before spending that budget. Native Pinocchio has tested
trajectory sensitivities; use that lane for an accuracy-matched pilot after
window/closure contracts pass. Cache keys must include every physical input.

Do not change geometry or weights simultaneously with fixing the solver. Preserve
an identical-objective seed replay, then one bounded experiment with independent
continuous validation. Current best native return remains expanded run01:
whole 23.783870 mm, terminal 62.819025 mm through .80 s; still unaccepted.
Full capture and final R2025b sextic verification remain required.
