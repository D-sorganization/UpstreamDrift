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

These native probes cover three production steps, not full swing motion.
Close neither F09 nor the epic from them. All six ecosystems and all 17 required
rows remain in scope. Complete capture horizon, trustworthy anatomical source,
contact/grip calibration, measured motion fit, muscle-only assistance and the
ultimate fully muscular OpenSim solve plus independent excitation replay remain
required. The 520-muscle OpenSim source's passive-force readiness remains a
separate scientific blocker; a numerical seed does not resolve it.
