# Native Zero-CustomJoint MTP Reduction Turnover

Issue #12176 is a narrowly scoped F07 source-preparation child of #11791. Its
branch `feat/f07-buet-zero-custom-mtp-12176` is stacked on zero-subtalar
#12172 and BUET–Hamner assembly #12165. Keep the PR unarmed until those
prerequisites land. The separate `native_mtp_reduction.py` PinJoint API and
source admission remain unchanged.

The assembled v4 source XML SHA-256 is
`8d4d349747060efbac31c707c4e01f452746a8fba32be77a425b5ac5be1d34a3`.
Its MTP joints are one-coordinate **CustomJoints**, contrary to the initial
PinJoint hypothesis. A stale PinJoint-only reduction correctly rejected this
source. The new exact profile admits the ordered zero targets, source
rotation/constant laws, native default and achieved lock state, zero frame
transform and no external coordinate consumers before using the native weld
on a derived copy. The zero-subtalar and CustomJoint MTP wrappers use one
profile-parameterized mechanical comparison; no generic CustomJoint class
admission is inferred. MTP wrapper source SHA-256 is
`05837ea5ce934d676c44cd6effb8e5062058e7a3b355a7b5083a1d6bde60cb29`;
shared reducer SHA-256 is
`6c05fdd87875207a35172366d7e4a0363c77ecaffbd97abdfee8e143ffdbf482`.
The prior subtalar public receipt binds its earlier reducer bytes and remains
historical evidence. Its exact native test was rerun under the shared source
revision and passed; this child does not silently relabel the old receipt.

The [public receipt](F07_CUSTOM_MTP_REDUCTION_RECEIPT.json) has SHA-256
`86cd7495e47b3358b716a1270a2c0cd0c4ba6115120f4f43de94b0abc76978d8`.
The retained native producer receipt in owned fleet-local planning has the
same JSON values with original CRLF line endings and expanded-array formatting (SHA-256
`55483ffe29e79b41a13084f73248346f9fd10da4d4e4b61f5f56b7b5a877af5e`);
the public copy uses governed LF/Prettier formatting. Its input is the
root-retained abdomen→subtalar artifact SHA-256
`8d8c3cf1dbe5e149df07b0309d508d78d2524e3c7beeaee327e24c4c43c8b81b`.
The fresh MTP-derived XML has SHA-256
`453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d`
and 2,247,450 bytes. OpenSim 4.6 reloaded it and observed four native
common-state poses with rank-56, condition-one mobility lift. Maximum body
transform/velocity, all 557 muscle length/speed, projected/full mass and
actual body-Jacobian-plus-mobility applied-force, QErr/UErr differences were
zero at these poses. The receipt binds the native binaries and source wrapper,
shared reducer, comparison helper and emitter script. OpenSim warned of
missing optional visual meshes; entrypoint XML does not close that resource
set.

The actual source-bound TDD test was RED on missing CustomJoint MTP module,
then GREEN. Negatives cover changed source bytes under an old digest, rehashed
nonzero or unlocked target, rehashed altered rotation law, a mutable target
declaration, unknown removed-coordinate consumers and XML external entities.
The separately retained original→abdomen→subtalar→MTP composition receipt
SHA-256 `c82e77aa1a760e6965a45cee22be530bab67fb8ed71de710521b51f9c7f3b5d7`
reports three complete prepared coupled states, rank-56 lift, zero sampled
mechanical errors and successful `MocoStudy.initCasADiSolver`. Its first
attempt failed honestly on the PinJoint-only admission. Neither a local final
step nor that independent initialization proves Moco solve feasibility,
physiological muscle state or captured swing fit.

Reproduce the focused native checks with exact retained source and composed
parent artifacts:

```powershell
$env:CASADIPATH='C:/Users/diete/Repositories/.venv-feedback-opensim/Lib/site-packages/opensim'
$env:PYTHONPATH=(Get-Location).Path
$env:UD_BUET_HAMNER_ASSEMBLED_SOURCE='<owned-immutable-v4-source.osim>'
$env:UD_BUET_HAMNER_COMPOSED_SUBTALAR_SOURCE='<owned-abdomen-subtalar.osim>'
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/unit/engines/opensim/test_native_custom_mtp_contract.py tests/unit/engines/opensim/test_native_subtalar_reduction_contract.py tests/opensim/test_native_custom_mtp_reduction.py tests/opensim/test_native_subtalar_reduction.py -q
```

The bounded emitter is `scripts/opensim/emit_zero_custom_mtp_receipt.py` with
`--source`, `--derived`, and `--receipt` for exact fresh paths. Keep derived
XML and native logs outside the public repository. Native Moco solve,
physiological passive/tendon/activation state, complete SimTK restart,
uninterrupted same-input excitation replay, own contact and grip, native marker
registration, private-capture matching, all 17 variants and six engines remain
open. Canonical calculation detail is chapter 47.
