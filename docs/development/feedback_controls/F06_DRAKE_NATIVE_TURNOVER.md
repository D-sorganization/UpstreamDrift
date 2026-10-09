# Native Drake Replay Turnover

## Delivered Boundary

Child #11838 extends F06 #11790 without closing six-engine parity. Public
build/replay functions live in
`src/engines/physics_engines/drake/python/native_torque_replay.py` and consume
the merged Tools T01 pin `2e7665111b06f92ffbfe178b92d74d6a81c95388`.
Canonical equations and admission policy are in manual chapter22.

## TDD and Native Reproduction

Commit `69dc588e67` recorded seven missing-module RED tests before code.
Actual native execution exposed three additional implementation failures:
default abstract parameters were incorrectly rejected; the Python API names
the numeric parameter count `num_numeric_parameter_groups`; unused native
numeric parameters contain NaN sentinels and cannot be JSON floats. The
restricted factory-default abstract policy and exact byte digests resolve
those failures without discarding parameter identity or modifying physics.

Eleven tests pass with actual Drake1.57.0/Python3.12.3 in an owned Linux
environment. The production source and pinned vendor are imported directly;
an isolated test copy avoids the repository's optional-provider mocking
conftest. No fake package namespace or replay schema is installed. On
Windows without native Drake these tests skip and supply no physics evidence.

From an environment with the real provider and pinned Tools seam:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=<repo>/vendor/ud-tools/src/shared/python:<repo>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest <owned-isolated-test-copy>/test_native_drake_torque_replay.py -q
```

The fixture is original synthetic URDF, with no capture or donor assets.
Its full state, saturated inputs, native net effort and fresh replay agree
within declared deterministic tolerances; no physiological accuracy or
numerical convergence is inferred.

## Resources and Failed Routes

The fleet Brick probe confirmed Drake1.57.0 and the unsampled-state layout.
A partial source export failed because installed regular package parents
masked the namespace copy. A complete production-source export was then
transferred to an owned remote directory; SSH transport stalled temporarily.
The existing transfer completed without launching duplicate jobs. Native
qualification used a new owned local Linux environment instead of changing
foreign fleet services or environments. Installations disabled download
caching, and runs used one numerical worker with bounded timeouts.

## Remaining Work

This adapter deliberately excludes collision geometry, constraint parameter
maps with entries, external resources, nonunit transmissions and providers
other than the reviewed1.57.0 layout. It does not accept arbitrary native
caller contexts. Native default numeric parameter bytes include unused
sentinels; state/input/output finiteness remains independently enforced.
Provider hashes do not attest every transitive shared library.

Next integrate native F03 candidates and the F09 acceptance registry, then
measure refinement and full-horizon time-to-accepted costs. Required engines
and full-body/muscle variants remain in the denominator while unsupported.
No private capture match or new preview video is claimed by this slice.
