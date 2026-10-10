# Native Zero-Subtalar Reduction Turnover

Issue #12167 is a narrowly declared F07 mechanical preparation under #11791.
Branch `feat/f07-buet-zero-subtalar-12167` starts from BUET–Hamner assembly
#12165, with the zero-MTP #12153 native comparison helpers in ancestry. The
assembled source and reduced XML stay in owned fleet-local planning, not this
public repository. The source is a 557-muscle diagnostic candidate, not an
accepted anatomical or physiological golf model.

The exact source is SHA-256
`8d4d349747060efbac31c707c4e01f452746a8fba32be77a425b5ac5be1d34a3`.
Its bilateral subtalar joints are **single-coordinate CustomJoints**, each
locked at zero with one LinearFunction(1,0) rotation and five Constant(0)
axes. A prior PinJoint assumption was corrected from the actual XML before
native code. The reducer checks both source and achieved zero values/speeds,
native parent/child frame coincidence, the exact spatial law and absent
coordinate consumers, then welds _only_ those two whole joints in a derived
copy. A stock weld without these gates could lose a nonzero target.

The [public formatted receipt](F07_SUBTALAR_REDUCTION_RECEIPT.json) SHA-256 is
`93d2f343426645d27a25f3b8e1eef76db68a9eddca81851f14ea04bfc28d4059`.
Its JSON values equal the retained native producer receipt in owned staging,
whose original bytes have SHA-256
`21425d9bc75ffb2af2be64831c7aee21c4802e7cf6909d33a7601f99787106be`.
It binds the exact source, freshly loaded derived XML SHA-256
`a646075a42b978f8a9c45b737849c6438101b6503621882e5fad322ad2d84434`,
reducer SHA-256 `7ed03c412ec035b9c03c40d72f25749ed3bfe346ec71203ced417631ea2d9945`,
comparison helper SHA-256
`e2e01cad795e8fc2a4be703019051d4953ae059a3893de81d37fba53499a5b73`,
OpenSim 4.6 version and three loaded extension hashes. The derived XML is
2,262,113 bytes. At initialized, right-ankle, right-knee and pelvis-tilt
poses with nonzero speeds, the reduced mobility lift had rank 60 and
condition number 1. Native body transforms/velocities, all 557 muscle path
lengths/speeds, projected and full mass, actual body-Jacobian-plus-mobility
applied force, and QErr/UErr differed by zero at all four sampled poses.
This is sampled native mechanics; force/bias/reaction equivalence over the
complete state domain is not established.

To repeat the scoped native tests in the qualified existing runtime, set
`UD_BUET_HAMNER_ASSEMBLED_SOURCE` to the immutable source XML path and verify
its hash above before running:

```powershell
$env:CASADIPATH='C:/Users/diete/Repositories/.venv-feedback-opensim/Lib/site-packages/opensim'
$env:PYTHONPATH=(Get-Location).Path
$env:UD_BUET_HAMNER_ASSEMBLED_SOURCE='<owned-immutable-v4-source.osim>'
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/unit/engines/opensim/test_native_subtalar_reduction_contract.py tests/opensim/test_native_subtalar_reduction.py -q
```

The native TDD first failed on the missing module; source-preserving test
corrections and implementation now pass. Negatives cover nonzero lock target,
unlocked target, changed rotation law even with updated XML digest, changed
source bytes under an old digest, missing/partial target declaration, unknown
coordinate consumer and external XML entity. Source defaults and native
achieved state are checked separately. The public receipt omits local source
paths and raw capture data. Entry XML SHA-256 does not close optional visual
mesh resources; OpenSim emitted missing-mesh warnings.

For #12161, two separately retained fleet-local observations of this same
assembled candidate exist at
`docs/development/feedback_controls_planning/buet_hamner_candidate_12157/`:
`complete-state-v2-receipt.json` (SHA-256
`0f1d35451dd76ec185814e25bb30c0eeb17a9acc26b72b927b136fbbdddda165`)
restores 1,054 BUET-prepared and 184 Hamner-initial named values without
postrestore equilibrium. Its maximum observed passive fiber force is about
4.51 billion N. `single-equilibrium-v2-receipt.json` (SHA-256
`21f1095267f1b0d476c39df655118f134da2e7292aada7a4b9d0c45e9b9eca9b`)
records one explicit native `equilibrateMuscles` intervention: 554 fiber
lengths changed, q/u/time/activations/options/locks/constraints and source
properties did not, and the maximum passive fiber force became 6,314.18 N.
The failed pre-call stale-cache attempt is retained separately. Neither
observation is a physiological or native-restart admission; the latter is
never selected silently over the former.

The abdominal partial-lock reduction, this subtalar reduction and the MTP
reduction need an explicit single-source composition and renewed full
mechanical comparison before Moco initialization. Root owns that composition.
Native Moco solve, full-state excitation replay, passive-force suitability,
active-wrap power, support/contact/grip, marker/anatomical registration and
captured full-swing matching remain open. All 17 variants and six engine
families remain in the denominator.
