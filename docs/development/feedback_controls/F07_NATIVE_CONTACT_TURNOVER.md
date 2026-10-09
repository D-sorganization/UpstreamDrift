# F07 Native Contact Replay Turnover

## Scope

Child #11815 extends early replay #11809; it does not close parent F07 #11791,
F08 #11792 or epic #11784. The new optional exact native contact path policy
reuses `OpenSimForceTorqueSource`, preserving native force calculation and
continuous initialization. No callback, external force or state correction
is introduced. Contact wrench evidence is complete or execution fails.

## Test-First Evidence

Commit `d9e35b05c5` records two intended failures: the replay API did not accept
the contact policy. The undeclared-contact rejection already passed. Native
OpenSim 4.6 then passed the initial three tests and four additional checks for
contact-driven motion, stricter integration, invalid paths and external forces.
Astra's native audit exposed a sphere-side-only evidence hole for moving
half-spaces. The bounded policy now requires a ground-fixed plane, a non-ground
sphere and exactly two HuntCrossley geometries; native regressions reject
moving planes and extra geometry. Both supported contact laws execute in tests.

The combined native replay, existing force provider, dependency graph, CMC and
Moco contract suite passes **108 tests**, with one smooth-law inapplicable extra-
geometry case skipped and three missing private-fixture opt-ins deselected.

Reproduce with Python 3.12 plus OpenSim 4.6 and project runtime dependencies:

```powershell
python3 -m pytest tests/opensim/test_native_muscle_replay.py --override-ini='addopts=' -q
```

The dedicated local native runtime is separate from the default interpreter;
default CI without OpenSim skips native tests and cannot qualify physics.

## Design and Limits

See canonical `manuals/upstreamdrift/chapters/17-native-muscle-contact.qmd`.
The model hash covers serialized material and smoothing parameters. Full
referenced resource/plugin/discrete state identity remains unqualified.
The synthetic slider fixture is not a golf-swing substitute. Required next
steps include full-body anatomy and wrist coverage, calibrated native contact,
bilateral grip, private protocol and full-horizon muscle excitation replay.
The native law permits compliant deformation and smooth off-contact leakage;
their acceptance budgets must be measured explicitly.
