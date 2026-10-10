# Native Tools Integration Turnover — F09o #12044

## Scope and Source Lineage

This branch integrates F09n `df9f3b95386433fe4c3a45559498b3e3f501cfdf` with
the admitted F01b/Tools/private-consumer head
`d232efcc6981b004d41698c565e3f7cbef058a71`. Tools is consumed at merged main
`86d0f28b1cc5acf61185e07e320c816c2d005512`. Requirements, Rust and gitlink
pins agree. The private loader dependency is now also merged on UpstreamDrift
main in PR #12002; its original implementation is reused.

`native_replay_contract_types()` uses `load_pinned_tools_package` and preserves
one private replay class identity. No new optimizer, native stepping loop,
replay format, timing cache or integrity-check relaxation is introduced.
Both branches' substantive SPEC/handoff additions are retained. Generated
context, divergence and monolith views are regenerated from integrated source.

## Actual Native Evidence

The existing DeskComputer MyoSuite 3.0.0/MuJoCo 3.6.0 environment ran the nine
F09i–F09n/shared-kernel modules listed in the F09n reproduction section:
**170 passed, zero failures/errors/skips, 208.34 seconds** (JUnit 208.176 s).
The suite retains actual original driver/iron probes, complete native-state
forecast/replay, mutation rejection and guarded fallback/optimization cases.

The synthetic three-step solver again improves objective `16.361447408944105`
to `0.0002999297924770022` in 189 numerical evaluations. This run's solve time
is `37.9167507999955 s`. This is another correctness observation, not a
controlled speed comparison, real-time result or private-reference match.

Before and after execution, all 50 declared integration source/test/input
hashes and all 8,122 tracked Tools archive files matched exactly. The manifest
also includes adjacent integration inputs; it does not imply that all those
modules or all model rows were executed by this nine-module campaign. The
native stage uses a verified Git archive without Git metadata. Its old vendor
tree is retained separately; no foreign primary checkout or SDK was modified.

Local evidence in the workspace planning folder:

- `f09o-native-full-campaign.xml`, SHA-256
  `b79074667c3467b8830b0bcacb1ecaf1f198b5b872521f7cfe18c68ae7bd77e5`.
- `f09o-native-source-hashes-before.json` and matching `after.json`; after SHA-256
  `6f7ddcaa38ba74e040fb9385bead819c4c2a35693fe65afdff5cc2d39d2932f9`.
- `f09o-staged-source-receipt.json`: Tools archive SHA-256
  `1990d6afeed692efa2c3e57d809863371a004527c71d6e5627e377e632cf1e43`.

## Consumer and Capability Validation

The existing local Python 3.12/MuJoCo 3.8 environment ran the six private
consumer modules: **81 passed, three skipped, 6.06 seconds**. Skips are the
unconfigured public MyoSim artifacts, unconfigured arm-model root and optional
Drake runtime. They are not native availability proof for those cases.
The global Python 3.13 portable run passed 58 with 26 skips; the central
affected seven-module run passed 61 with 28 skips. These portable results
remain separate from both actual-native lanes.

The impact pin check first failed genuinely against the old reconciliation.
All 26 applicable capability probes then passed, followed by actual served
verification at the new pin: 76 assets, five JavaScript assets, 2,801,426 bytes,
exact mount/revision/digests and missing-artifact 404. Only then was acceptance
metadata advanced; all 27 matrix tests passed. Preserve the previous matrix
and T01 receipt. The new public receipt is `F09O-IMPACT-SERVED-RECEIPT.json`.
See the impact acceptance matrix for actual Node/npm/Python/FastAPI versions;
these diagnostic toolchains do not claim equivalence to the CI versions.

Retained setup failures include the initial remote extraction quoting error,
the missing FastAPI provider in the native SDK environment, and an incorrect
test-file path that ran no tests. No environment installation, test bypass or
metadata-only acceptance was used to repair them.

## Reproduction and Handoff

Initialize the exact admitted Tools gitlink. Use the F09n reproduction commands
with that new pin and a fresh receipt filename; preserve the original F09n
campaign at its historical `e775bce` pin. Native environment/resource variables
retain the same meanings. Run the consumer suite separately in the reviewed
MuJoCo 3.8 environment, and the capability/served commands in the impact matrix.
Before publication, verify runtime source correspondence against committed HEAD,
record any formatter-only metadata delta separately, and run normal hooks.

Local checks currently pass: lint/format on all 82 integrated Python files,
central five pre-PR gates on the replay/loader integration scope (including
mypy on five sources), 15 pin/build-boundary/register tests, LoD no-growth,
document titles, file-size budget and design-manual governance. Architecture and
DRY pass without new exceptions or duplicate growth. Context review/render/check
passes at the integrated source. The manual inventory remains release-blocked.
Normal hook, current-main integration and GitHub results must be recorded
truthfully before publication.

## Remaining Full Epic Work

The prior profiling evidence identifies repeated provider/source verification
as a major cost. Any optimization needs its own TDD and complete mutation,
state, callback, replay and hard-criterion preservation; no cache is added here.
Continue model-specific hard margins, reference mapping and receding-horizon
execution, then measured complete time-to-accepted-match and independent replay.
All seventeen production model/variant rows, six ecosystems, full private
capture fitting and the muscular OpenSim endpoint remain open. Preserve Astra's
prepared-state/domain/conditioning and source-coordinate/branch/anatomy gates.
