# Budgeted Fixed-Candidate Sensitivity Diagnostic 65

**Passed with 82,931 sensitivity evaluations**, below the explicit 100,000
limit. Maximum primal marker component discrepancy was 3.88458606e-8 m,
below the unchanged 1e-7 m gate. Sensitivity integration took 27.0993 s;
call/save took 35.1773 s and launch-to-terminal took 39.3157 s (exit 0).
No optimization, interruption, restart or threshold modification occurred.

This checks exactly the run63 candidate
`e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d`,
verified from its checkpoint before execution, against the original model,
initial state and 307 capture times over 0–0.85 s. Arguments are first_control
4, rtol 1e-10, atol 1e-12, max_step 0.000125 and
max_sensitivity_evaluations 100000. Supplied tolerances govern the augmented
solve; the independent primal provider retains rtol 1e-11 / atol 1e-13.
The smaller max_step applies to both. The budget counts augmented
linearization calls, not independent primal or output-audit work.

Run63 failed at max_step 0.00025 with these augmented tolerances. Run64 passed
with tighter augmented tolerances at 1,222,535 evaluations / 409.086 s. Run65
shows that the smaller max_step alone, with the new nonbinding cap, is enough
to pass this one candidate's primal agreement check at substantially lower
cost. It does not validate every derivative column, establish robust convergence
across candidates or accept this exploratory swing. No additional fit was run.

## Qualified Runtime and Trace

Immutable runtime `/home/dieterolson/native-sensitivity-9967-65` clones61 and
overlays exact c52f2029f continuous_forward, forward_sensitivity and
native_sensitivity providers plus their tests. Final qualification ran 36
tests in 0.88 s, including budget contracts and nine real Pinocchio manifold
tests. Six unknown pytest-unit-mark warnings reflect omitted repository pytest
configuration. The inherited outer namespace scaffolding remains documented
in runtime61's handoff; no numeric providers were shimmed or replaced.

`runtime-receipt.json`, `overlay-hashes.json`, `source-overlay.zip` and
`runtime-source.zip` preserve test results and exact runtime files. Runtime
archive SHA256:
`c416d5d6b986837205501ed8227c28b91a286b86d83d55003df2c4aa71966b97`.

`diagnose_sensitivity.py`, `receipt.json`, `summary.json` and `raw-run.zip`
preserve exact driver, arguments, launch/environment, source/input hashes,
terminal outcome and full marker/state sensitivity arrays. Raw archive SHA256:
`4f7fa72ed3b175316542d2b72262dc18dfe80ae3ab3d9c7324d47380973a6966`.
All overlay hashes match the runtime, qualification and execution source hashes
match, and raw input/output/archive hashes were verified after download.
Force-add ignored ZIP archives when committing.

Remote output: `/mnt/c/Users/diete/native-sensitivity-9967-65`. Preserve it and
the runtime unchanged. Root owns derivative reliability review and the next
fitting decision. No production edits or commits were made by this execution
agent; all its processes are terminal.
