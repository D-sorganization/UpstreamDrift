# F09i Project MyoSuite Task Turnover

Issue: https://github.com/D-sorganization/UpstreamDrift/issues/11983.
Branch: `feat/f09i-myosuite-task-producer-11983`, based on published F01f
`73319871c74dbafeefc6548081e606cadc61de54`. The stack remains dependent on
actual Tools and UpstreamDrift parent merges. Feature pins are not main authority.

## Implementation and Canonical Design

Read canonical chapter 38, `38-project-myosuite-task-replay.qmd` before extending
the pathway. The project task invokes actual SDK reset/step/action/observation
bookkeeping. Its frozen mixed-command history is exported through existing T01
and compiled-profile contracts, then verified by the existing independent native
executor. The fresh native factory has no Gym lifecycle or stepping loop.

Initialization policy 1.1 preserves the absolute native epoch inside the full
integration state while returning measured zero-based elapsed times. The shared
interval precision predicate rejects stalled and distorted clocks. Old policy
1.0 keeps its zero-native-origin restriction. All resource, callback, ordered
compiled-law, full-state and external-load gates remain enforced.

## Actual Evidence and Failed Attempts

Actual MyoSuite 3.0/MuJoCo 3.6 campaign: 70 passes, zero skips. Both unchanged
driver/iron models pass complete resource admission, byte-round-tripped T01,
native-profile matching and exact full-state uninterrupted, suffix and changed
future replay. Independent replay works with SDK lifecycle hooks poisoned.
Separate native MuJoCo 3.8 regression: 44 passes, three explicit skips.

Retain failed native attempts: four RED cases demonstrated the old 3.8-only
resource/law assumptions and large-epoch clock failures; two further RED cases
demonstrated nonzero-origin/suffix rejection. Adversarial export originally
accepted altered intermediate state/time and out-of-bounds post-mapping
commands (10 passes/three RED). These now fail through bounds/state-clock
admission and complete history verification using the canonical executor.
One intermediate 64-pass run failed only its expected error-message regex;
the rejection was already correct, and the diagnostic now identifies state.
Two actual source-origin RED cases accepted rehashed XML comments/whitespace
with identical compiled MJB. Preconstruction source/resource freezing now
rejects them. The actual SDK execution profile found CPU accessors in the base
source and an executed `muscle_stages.py` helper; that helper and distribution
METADATA now have separate declared digest checks. Two additional RED cases
first demonstrated the missing helper/metadata binding interface.

The actual campaign used a preserved fleet SDK environment and an owned public
source staging directory, not another model/environment download or changes to
a foreign checkout. Public Python Tools staging is bound to exact source
`e775bce870690bba1b56f3d6297513003fb7dbac`, still a feature pin. Model files and
capture inputs were not changed. Reproduction commands and numerical limits
are in the canonical chapter. Source-hashed JUnit/attempt receipts are retained
in the workspace's feedback-controls planning evidence directory.

## Remaining Acceptance

An additional original-source nominal viability experiment completed all 1,814
driver and 1,828 iron native steps with zero motor commands and constant 0.05
muscle commands. Both serialized T01 replays matched complete states, commands
and clocks exactly, with no native warnings. This uses capture duration only.
Contact penetration near 0.032 m and large mixed-unit native constraint
residuals remain unqualified. Retained receipts are in planning's
`f09i_native_horizon_viability` folder; the initial three-step test campaign and
these full-duration nominal experiments remain distinct.

The driver exposed substantial fingerprint cost: its native MJB is 29,487,449
bytes and is checked before and after every step. Invocation-local zero-filled
scratch and immediate memory-view hashing preserve every check while avoiding
allocation and bytes-copy overhead. An actual native serialization-only
microbenchmark measured 0.9078 s versus 0.5152 s for twenty operations with
identical hashes; no full-run speedup is inferred. Six native 3.8 regression
cases followed API RED to GREEN for external-byte equivalence, mutation
detection and invalid storage. Existing full-duration receipts used the original
producer, before this optimization. Current actual MyoSuite 3.0/MuJoCo 3.6
revalidation after the optimization and exporter refactor passed 76 tests with
zero failures, errors or skips.

PR #11996 CI exposed inherited Python and Rust install pins that disagreed with
the vendored Tools source and two export architecture-budget violations. Both
pins now match the existing feature gitlink; this does not claim main authority.
Resource admission is extracted into a single helper and registration imports
the reviewed native module directly. The initial local architecture check ran
before commit and therefore did not inspect the uncommitted exporter. The exact
CI failures and locally reproduced dependency RED are retained in planning.

The integration test campaign covers three production steps; the separate
full-duration nominal experiments do not fit the captured swing motion.
Close neither F09 nor the epic from them. All six ecosystems and all 17 required
rows remain in scope. Complete capture horizon, trustworthy anatomical source,
contact/grip calibration, measured motion fit, muscle-only assistance and the
ultimate fully muscular OpenSim solve plus independent excitation replay remain
required. The 520-muscle OpenSim source's passive-force readiness remains a
separate scientific blocker; a numerical seed does not resolve it.

## Initial Assembly Findings and Next Implementation

The saved reset is exactly source `qpos0` in both variants. Read-only native
forward diagnosis places the largest mixed constraint residual at reset in an
active grip weld, with right/left grip-site separations about 1.1703/0.9940 m.
MuJoCo permits separated site welds to align during simulation; this finding
does not prove source invalidity or assembly infeasibility. Native foot contacts
are absent at reset. Later skull/platform penetration and right-ankle limit
excess are separate events observed in the uncontrolled rollouts.

Before fitting, qualify bounded native initial assembly with explicit provenance,
named grip/body-weld residuals, joint limits, actual collision support and full
integration-state independent reload. Preserve the original default. An
optimizer-assembled pose cannot inherit a captured-reference label without a
qualified marker/frame correspondence. Kinematic support is not dynamic force
balance. Planning receipts in `constraint_diagnostics` bind XML/MJB/history;
they do not independently recheck the full resource closure, and the included
Desk scene differs from local scene bytes. See chapter 38 for limitations.

Current protected CI passes architecture, source contracts, Rust quickstart and
Python 3.11/3.12 tests. Its completed agent-context failure is now the intentional
Tools-main ancestry gate: the feature gitlink is still unmerged. Retain the
draft/unarmed dependency hold.
