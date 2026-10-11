# F07 Structural Weld Admission — Issue #12280

## Scope

This branch adds a source-bound structural rule to constrained-muscle replay.
It admits only exact native `WeldJoint` components with zero coordinates and
valid, distinct absolute-path parent/child frames whose concrete classes are
`Body`, `Ground`, or `PhysicalOffsetFrame`. The original strict
`native_muscle_bundle._COMPONENTS` set is unchanged. `WeldConstraint`, plugin
classes, controllers, prescribed motions, and undeclared non-muscle forces
remain rejected. The constrained adapter policy version is now `2.1.0`, and
the adapter source digest changes with its implementation.

## Validation

Pure policy tests pass: 14 tests cover the positive structural case, native
cast failure, nonzero coordinate count, missing/invalid/unsupported/equal
frames, other weld-like classes, unchanged base policy, and non-waiver of
controller/force rejection. The test uses the pinned Tools source at
`C:/Users/diete/Repositories/Worktrees/feedback-controls-main-integration-12202/vendor/ud-tools/src`.
The working pytest bootstrap is:

```powershell
$env:PYTHONPATH='C:/Users/diete/Repositories/Worktrees/feedback-controls-main-integration-12202/vendor/ud-tools/src;C:/Users/diete/Repositories/Worktrees/feedback-weldjoint-12280'
python -c "import shared.python, pytest; raise SystemExit(pytest.main(['tests/opensim/test_native_constrained_weld_policy.py','-q']))"
```

The native fixture is prepared but has not been run. Entry points for the
future separately reviewed bounded OpenSim execution are:

```text
tests/opensim/test_native_constrained_muscle.py::test_structural_weld_fresh_reload_and_saved_excitation_replay
tests/opensim/test_native_constrained_muscle.py::test_weld_transform_or_socket_mutation_invalidates_old_bundle
```

The fixture includes nonzero parent/child offset translation and rotation, a
muscle attachment on the welded body, a movable coordinate, and a coupled
coordinate. It verifies fresh XML transform/length preservation, complete
named-state restoration, deterministic same-input replay, changed future
excitation response, and old-bundle invalidation after source mutation. It
must not be included in broad Moco test selection. Native execution remains
pending root review; no native result is claimed here.

## Limits

This is a software-admission change only. It does not establish capture
matching, anatomy, muscle physiology, contact, support/grip, full-clock replay,
or qualification of any production model. The calculation registry and
publication status remain blocked.
