# Native Golf Model Equivalence and Matching Program

Updated 2026-09-12 UTC. This document adds the user's explicitly requested
MuJoCo and Drake implementation lanes to the active Pinocchio/Simscape goal.
It preserves the full goal: a repeatable full-capture tour swing driven by one
global degree-six effort polynomial per channel, with geometry calibration where
justified and independently verified native behavior. MATLAB acceptance requires
R2025b. OpenSim is an additional staged implementation lane under epic10003.

## Latest Terminal Result

Run20 is finished and independently rejected: whole29.941228 mm,
terminal98.417488 mm, continuity1.202905e-4, iteration limit. No optimizer is
live. Read the top native checkpoint before older live-run descriptions below.
The optional executor is implemented and qualified in isolation; solver/cache
integration is the next bounded implementation, followed by the native smooth
path prerequisite for better initialization. No new fit has been launched.

## Ownership and Resume Pointers

| Lane                        | Issue       | Branch and Worktree                                                   | Owner                                          |
| --------------------------- | ----------- | --------------------------------------------------------------------- | ---------------------------------------------- |
| Shared Solver and Pinocchio | 9967        | feat/9967-native-simscape-pinocchio; UpstreamDrift-pinocchio-native   | Root Codex                                     |
| MuJoCo Native               | 10021       | feat/10021-native-mujoco; UpstreamDrift-mujoco-native                 | Parallel mujoco_native agent                   |
| Drake Native                | 10022       | feat/10022-native-drake-equivalence; UpstreamDrift-drake-native-10022 | Parallel drake_native agent                    |
| OpenSim                     | 10003       | docs/10003-opensim-matching-epic; UpstreamDrift-opensim-10003         | Planning complete; OS-0 implementation pending |
| MATLAB and Coordination     | 9921 / 9964 | UpstreamDrift-simscape-tour                                           | Existing Gemini lane                           |

Worktrees live under C:/Users/diete/Repositories/Worktrees. Read each lane's
AGENTS.md and CLAUDE.md and renew its issue lease before editing. Engine agents
own engine-specific source/tests and docs/development/mujoco_native_matching or
drake_native_matching, including HANDOFF.md. Shared changes require coordination
with root; no parallel optimizer implementations. Never modify another agent's
runtime, environment or active jobs. Issue9964 is the cross-engine coordination
record. The top of NATIVE_PORT_CHECKPOINT_20260911.md owns current Pinocchio state.

## Required Native Equivalence Gates

1. Inventory the real engine environment, source and model. Keep native runtime
   dependencies isolated. Reuse native URDF plus sidecar validation and shared
   candidate, marker, effort and forward-replay contracts. Plain URDF is not
   sufficient to preserve loop closure or actuation semantics.
2. Preserve27 scalar primitives,31 solids, original inertia/transforms/gravity,
   six-dimensional right-grip closure, declared effort force-frame mapping,
   marker frames, original q0/qd0 and zero native damping/stiffness. Numerical
   parser limits must not become physical constraints. Check exact identities.
3. Write failing tests for schema/contracts, topology and dynamics first. Check
   malformed/nonfinite inputs and incompatible hashes. Keep functions small and
   dependencies explicit; reuse shared code (TDD, DbC, LoD, DRY).
4. Compare mass/COM/frames, zero-input and27 independent effort pulses, and
   moving-state accelerations at the SAME physical states against native
   Pinocchio/R2025b evidence. Quantify errors and record units and tolerances.
5. Independently replay identical candidate controls from original initial
   state. Compare pointwise states/markers and closure, not differences between
   two aggregate marker RMS values. Qualify the original0.8 s baseline first;
   extend qualification to final fitted controls and full capture later.
6. Integrate each qualified engine through existing fitting/replay contracts.
   Preserve one global sextic and actual missing-marker masks. A segmented
   optimization result is never accepted without continuous forward replay.

## Engine Constraint Semantics

MuJoCo's stock compliant weld and Drake's discrete SAP weld are not automatically
the same dynamics/integration contract as the verified continuous native model.
Both agents have qualified clearly named rigid constrained adapters built
from their engine's actual mass, bias and closure Jacobian/Jdot quantities.
Such an adapter must be independently qualified. It must not be presented as
proof that unmodified mj_step or a stock Drake simulator is equivalent. Preserve
and quantify stock-mode limitations separately. A visually similar model is
not behavioral parity, and an independently computed equivalent adapter is not
proof of every engine application entry point.

## Current Evidence and Next Work

Pinocchio native initial/moving and continuous 0.8 s baseline parity against
R2025b is qualified; full-swing matching is not. Run19 ended unaccepted at its
iteration limit: whole 30.791102 mm, terminal 99.988650 mm, continuity defect
1.091819e-5. Run20 is live, session43223 / ControlTower-Runner PID2348439,
with 71 selected correction intervals widened from +/-2 to +/-4. Its distinct
reconstructed seed passed derivative checks but failed strict source-state
rate parity, preserved explicitly. The top native checkpoint records exact
source identities, immutable runtimes, archives and ordered continuation steps.
The two-worker sensitivity study measured 1.83572x speedup with identical full
outputs; production batching is not yet integrated. The loop-reaction-eliminated
sextic study supports an initializer pathway but still requires a smooth native
target trajectory and independent forward acceptance. Both reports are linked
from the checkpoint. No second optimizer or runtime change is active.
MuJoCo and Drake custom rigid baseline adapters are now integrated into the
root branch as06576fa4a and9ff486f11. Their source-hashed R2025b and Pinocchio
qualification evidence covers the specified0.8 s baseline only; read the latest
Pinocchio checkpoint and engine HANDOFF.md files for exact measured quantities.
Stock simulator modes, full-swing fitting and engine sensitivities remain
unqualified. MuJoCo's validated bundle conversion factory is integrated as
23f2d0235, with another passing 168-case/0.8 s qualification. The native entry
import fix is integrated as 8ac65c486; 23 root regression tests pass. Drake's
run 18 extension at 0.85 s failed the rate gate at two matched integration
tolerances; the discrepancy is archived and remains unresolved. OpenSim implementation
has not started.

Each lane must commit incrementally and keep its HANDOFF.md current with source,
model, capture and candidate hashes; exact environment/commands; live versus
terminal process identities; test evidence; measured parity scope; current
limitations; and the next executable assignment. Preserve raw hashed evidence
in immutable archives when formatters change adjacent JSON. Update issue9964
on material results or ownership changes. Do not mark the overall goal complete
until all requested engine and final full-swing requirements are verified.
