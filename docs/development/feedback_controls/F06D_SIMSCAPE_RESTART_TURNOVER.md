# Native Simscape Restart Turnover

## Scope and Ownership

Bounded fixture issue11933; parent integration issue11921 remains open.
Branch `feat/feedback-simscape-replay-11921`; reused checkout
`Worktrees/feedback-opensim-bundle-11908`. Its parent now includes actual
OpenSim bundle seam fix5343eb8f7d. The earlier Pinocchio checkout is separately
reused for that published bundle branch. No new clone or MATLAB installation.
Chapter30 is canonical design authority; code lives in scripts/matlab.

## Actual Native Validation

DeskComputer, explicit MATLAB R2025b Update5, batch exit0. A physical mass-spring-
damper network and Unit Delay run uninterrupted and from a saved/reloaded native
ModelOperatingPoint at0.2s to0.4s. Position error0m over366 post-split samples;
discrete error0 over21; actual executed injection error0N. Native clock remains
0.2s. Policy: ode23t, MaxStep0.01s, RelTol1e-8, AbsTol1e-10, block reductionoff,
converterN/zero filtering and derivatives; ZOH with holding final value.
Missing/model/time/runtime/blob/class negatives passed. Missing-helper and
nontrivial-motion/interface failures were recorded before fixes.

The fixture loader is limited to task-owned producer artifacts and binds hashes,
class, runtime release, model, input, solver and clock. It is not an untrusted MAT
loader or the complete Tools1.1 frozen-byte consumer. Exact native production
provider semantics and all-model replay remain open.

## Reproduction, Evidence and Storage

Run `test_native_simscape_restart(owned_output_directory)` with explicit
`C:/Program Files/MATLAB/R2025b/bin/matlab.exe`. Native MAT/SLX artifacts remain
in the owned Desk run directory, not in Git. Six diagnostic CSV hashes were
verified against `native-restart-receipt.json`. Fleet staging retains receipts,
stdout/stderr and source test failures. Desktop preview output is
`C:/Users/diete/Desktop/Motion_Matching_Previews/simscape_native_restart_11921`;
its producer verifies every CSV digest and records media hashes. This is a
synthetic restart preview, never a golfer/capture-match video.

Model SHA7297fc367b1dd6cd8a148d3a15a718db5f4371d4b5f649a06a4ea7c48b55134c;
snapshot SHAe7ce49850b9fc818487bc868cdbb26061264c014a0ace1b95e56eaf54f104be3;
input-file SHA6f7c79ae8a0623ef8fdffab137da5029c6b37af29dc402c75ddd41fdf8264fdc.
MAT byte identity is not canonical semantic state encoding. Root owns publication
and normal repository checks. Centralized five-gate checks passed for the four
MATLAB sources; their Python test mapping is inapplicable, so the actual R2025b
native tests above supply execution evidence. Document title and design-manual
governance checks passed; scientific release remains blocked. The ten preview
artifacts total 371416 bytes before the separate preview receipt. Their source
hash manifest matches all four executed MATLAB files byte for byte.

## Next Required Work

Consume exact published Tools5471 native envelope dependency; implement immutable
verified byte ownership through native decoding; bind effective configuration and
actual production source/model/runtime/input mapping; add F09 native execution
and retained required driver/iron rows. Verify nonzero-time native restart then
full-horizon independent replay and refinement. Source anatomy, contact support,
private placement/registration, physiology and six-engine parity remain blockers.
