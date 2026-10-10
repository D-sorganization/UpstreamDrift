# Exact 557-Muscle Native Replay Turnover

Issue #12182 owns the opt-in diagnostic profile on branch
`feat/f07-exact-557-muscle-replay-12182`, stacked on #12181 and normally merged
with root #12173 commit `9c4870c1f6368e84ebe57fcae4d989c819969ad2`.
Keep the PR unarmed until those prerequisites land. Version two of
`native_constrained_muscle.py` still rejects the complex source. Version
three requires the exact derived source SHA-256
`453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d`.
The input XML is a locally retained derived artifact, not committed model
data. The new profile's component-path manifest, 106 attached visual Mesh
references, four explicitly audited rotational `Coordinate.Coupled` units,
all 40 source-clamped ranges, and native source/runtime hashes are checked.
There are no native contact geometries or non-muscle forces in this artifact.
The missing visual VTP assets remain an explicit rendering limitation; they
cannot supply an unreviewed dynamics resource.

The initial values come from the one-equilibrium v2 receipt SHA-256
`21f1095267f1b0d476c39df655118f134da2e7292aada7a4b9d0c45e9b9eca9b`.
The 12 removed coordinates' value/speed states are omitted from the derived
model, leaving 1,226 native continuous states. The first untouched-source
seed and a subsequent interior-seed 2 ms attempt both failed when native
`SC_z` crossed its unchanged zero lower range at 1 ms. Their exact script
and log hashes are in [the receipt](F07_EXACT_557_REPLAY_RECEIPT.json).
The retained interior diagnostic moves only six source-bound-sitting
coordinate values by 0.1% of each native range. Every speed, activation,
fiber state, model parameter, constraint flag and XML byte is unchanged.
There is no post-restore assembly, fiber equilibrium, reset or measured
capture target.

At 0, 0.1 and 0.2 ms with 557 constant 0.05 muscle excitations, two fresh
saved-input T01 replays produce identical 1,226-state and 557-force arrays.
All 17 couplers are observed: maximum QErr `6.27230114285851e-17`, UErr
`1.5987211554602254e-14` in native units. A different final excitation
knot keeps the preceding native state identical and changes the selected
muscle activation by `0.0005770428853785967`. Exact input digests differ.
The run binds the native OpenSim 4.6 binary, loaded model, provider, policy,
observer, source and script hashes; the retained producer log is in owned
fleet-local planning `exact_557_12182/probe_bundle_v4.log`. The public receipt
contains numeric outputs and hash identities without bundling the external
model. It is a short diagnostic only. The original source additionally has
four independently identified physically invalid body inertia tensors, so
source qualification remains blocked even if longer replay becomes possible.

Reproduce on the retained exact source and preparation receipt:

```powershell
$env:CASADIPATH='C:/Users/diete/Repositories/.venv-feedback-opensim/Lib/site-packages/opensim'
$env:PYTHONPATH=(Get-Location).Path
$env:OPEN_SIM_EXACT_557_SOURCE='<retained-03-custom-mtp.osim>'
$env:OPEN_SIM_EXACT_557_PREPARATION='<retained-single-equilibrium-v2-receipt.json>'
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/opensim/test_native_exact_557_profile.py tests/opensim/test_native_constrained_muscle.py tests/opensim/test_native_muscle_bundle.py -q
```

The focused TDD suite was RED on a missing exact profile, then 44 native
tests passed, with no skip, on the executed files. Negatives include a
rehash-updated XML mutation, a dynamic external-file reference, a missing
clamped coordinate declaration, and default version-two rejection. The
changed-future test checks independent fresh native integration, not just
wire integrity. Keep source/model assets, raw logs and speculative anatomy
outside the public repo. The exact profile does not qualify whole-model
inertias, source physiology, contact/grip, full SimTK restart, native Moco
solve, full-horizon capture fit or the 17-variant/six-engine denominator.
