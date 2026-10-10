# Native Zero-MTP Reduction Turnover

Issue #12150 adds `native_mtp_reduction.py`, an exact-source, zero-target-only
OpenSim ModelFactory reduction. The public entrypoint accepts a source XML path
and SHA-256, an absent derived path, and the ordered right/left MTP zero-target
declaration. It rejects nonzero, unlocked, prescribed, non-PinJoint, boundary,
speed, and dangling-coordinate cases before publishing a derived artifact.
The source bytes are never modified. A fresh native reload compares the source
and derived models at the same named continuous state, including two varied
ankle/pelvis samples. Failed native comparisons remove the output.

The retained [native receipt](F07_MTP_REDUCTION_NATIVE_RECEIPT.json) binds the
original Rajagopal source SHA-256
`8708ed0d6a212080a72b9c277071ddd12939c8306efceb3ee4a9592735f43f80`,
scaled club factory SHA-256, reducer and emitter source bytes, derived XML,
OpenSim 4.6 build, and its simulation, Simbody, and actuator extension binaries.
Both exact native sources pass three sampled poses with full mobility lift rank
37, condition 1, and zero measured transform, spatial-velocity, muscle-length,
mass and total-applied-force differences. The factory maximum QErr/UErr is
`5.551115123125783e-16`. The source-backed factory remains a mixed-assistance
hypothesis, not an 80-muscle golfer or a pure-muscle model. Its serialized
derived XML is separately reloaded; output hashes are in the receipt.

The native force comparison projects rigid-body forces through the native
system Jacobian transpose and adds mobility forces. A 2 kg offset Pin fixture
shows why mobility forces alone are insufficient: they are zero while body
projection is approximately `-19.6133` N·m. Inertial bias and constraint
reactions are distinct quantities. The checker rejects nonfinite native
observations; the test injects NaN into that boundary. The nonzero 0.3 rad
lock test preserves the independent counterexample to a stock weld at a
nonzero target.

Reproduce with the qualified OpenSim 4.6 interpreter and explicit read-only
donor path:

```powershell
$env:CASADIPATH='C:/Users/diete/Repositories/.venv-feedback-opensim/Lib/site-packages/opensim'
$env:PYTHONPATH=(Get-Location).Path
$env:UD_RAJAGOPAL_SOURCE='C:/Users/diete/Repositories/UpstreamDrift/shared/models/opensim/opensim-models/Models/Rajagopal/RajagopalLaiUhlrich2023.osim'
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/opensim/test_native_mtp_reduction.py -q
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' scripts/opensim/emit_zero_mtp_receipt.py --donor $env:UD_RAJAGOPAL_SOURCE --golf-model src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim --output docs/development/feedback_controls/F07_MTP_REDUCTION_NATIVE_RECEIPT.json
```

The emitter requires an absent output path; use a fresh copy/path for a rerun.
Its XML hash binds only the entrypoint model; referenced visual-mesh closure
remains unverified. Other locked joints, knee couplers, passive muscle loads,
assistance, foot contact, full state/numerical restart, active wrap-route
changes, force/power and cut-wrench equivalence, Moco feasibility and full
horizon excitation replay remain open. Do not replace these gates with the
sampled reduction receipt. All 17 variants and six engine families remain in
the parent epic denominator.

TDD evidence: the new module first failed import in the native fixture; the
qualified runtime then passed 11 native cases, including exact donor/factory,
nonzero target, dependency and source mutation, body-force projection, and
nonfinite observation negatives. Canonical calculation authority is manual
chapter 43 and its calculation-registry entry.
