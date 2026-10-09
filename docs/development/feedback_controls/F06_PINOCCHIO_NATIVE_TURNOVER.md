# Native Pinocchio Replay Turnover

## Delivered and Unqualified Boundaries

Child #11900 uses `PinocchioPhysicsEngine` public integration with the Tools
T01 replay contract. Canonical chapter25 records calculations, source
admission, state/cache distinctions, effective input mapping and limitations.
The implementation shares contract loading, fixed history validation and
identity comparison with existing native adapters to avoid duplicate policy.
Those shared helper bytes remain bound in each provider hash.

## TDD and Native Execution

Native RED commit `f273e45f2f` precedes implementation: nine missing-module
errors with actual Pinocchio4.1.0, then twelve passing native checks including
refinement. Source discovery found the existing public native integrator;
no ABA/RK4 kernel was copied. Original floating URDF probe confirmed `nq=8`,
`nv=7` and native model serialization before implementation.

The existing Linux Pinocchio4.1/Crocoddyl3.2 environment is unchanged. An owned
system-site-packages overlay adds Matplotlib required by the existing engine's
visualization imports; packages were installed without download caching.
Tests import complete production source and the exact Tools T01 pin
`2e7665111b06f92ffbfe178b92d74d6a81c95388`. An isolated test copy avoids global
conftest mocks. Runs use one numerical worker and a 120-second timeout.

Reproduce with real Pinocchio and the pinned Tools seam on `PYTHONPATH`:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=<repo>:<repo>/vendor/ud-tools/src:<repo>/vendor/ud-tools/src/shared/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest <isolated-copy>/test_native_pinocchio_torque_replay.py -q
```

## Resource and Failed Setup Record

The checkout shares existing Git objects and excludes unrelated surfaces.
The Tools submodule is sparse and borrows the retained primary object store;
retire no donor until borrower audits pass. An initial Windows CRLF alternates
file failed Git path resolution; rewriting the owned pointer with LF restored
the actual pinned commit and clean submodule state. No source objects changed.
Initial test setup also exposed missing canonical `shared` import roots;
adding the pinned Tools `src` path fixed setup without fake packages.

## Next Work

Keep the PR stacked on actual native Drake/shared admission ancestry until
dependencies merge. Eleven actual Drake1.57 regressions pass on this shared
helper revision. Thirteen MuJoCo regressions also pass on supported native3.8.0 after an
earlier development-only3.3.4 run. The owned Python3.12 environment received
MuJoCo3.8 without changing existing NumPy2.5.3 or SciPy1.18.1. All five
central pre-PR gates, scoped mypy, DRY, LoD and architecture checks pass.
All normal commit and push hooks passed on implementation39341c76d9.
Then bind the real model inventory and F09 executor, extend constrained native
contact and muscle policies, and qualify the full private-capture horizon.
No private matching, new video, physiological validity or full parity claim
is supplied by these synthetic tests.
