# F07 Constrained Native Muscle and Moco Turnover

Issue #12143 is a focused child of F07 #11791 and epic #11784. Branch
`feat/f07-constrained-muscle-replay-12143` stacks on the reviewed #12136
declared cold-start source; the PR remains unarmed until its prerequisite
lands. Canonical calculation detail is chapter 42. The versioned authority
is `tour_matching/native_constrained_muscle.py`, consumed by the maintained
Moco request, preparation, replay, scoring, and CLI modules. The old muscle
bundle policy remains strict; mixed muscle/mechanical commands belong to a
separate reviewed profile.

## TDD and Native Evidence

The native MovingPathPoint test first failed at the component firewall, then
passed only after the new constrained profile checked all coordinate
references against declared charts and all coordinate functions against a
finite linear/constant law. A source SimmSpline mutation, changed CustomJoint
rotation coefficient, wrong lock target, source-byte mutation, and unreviewed
force fail closed. The source fixture has a CustomJoint, a native
CoordinateCouplerConstraint, two actual Millard muscles with activation and
fiber states, and a coordinate-linear moving guide. Direct native guide
motion measures 0.01 m/rad at the declared chart. Two fresh independent
replays reproduce frozen applied excitations and named states; a changed
future excitation preserves the common prefix and increases subsequent
flexor activation. Marker sampling restores replay states on a fresh model
without assembling or integrating again. The JSON request round-trip binds
the exact declared policy version; unknown versions reject.

The first endpoint experiment used path geometry with **zero native muscle
force**. Under a loose $10^{-4}$ Moco constraint tolerance, a nominally
successful $10^{-5}$ rad endpoint produced nearly zero speeds and a fresh
replay discrepancy exceeding the demand. Tight tolerance made this source
infeasible. It is retained as a negative scientific finding, not as accepted
matching evidence. The repaired fixture has about 0.498 N flexor and
0.459 N extensor force initially and genuine native q/u motion.

On the repaired coupler-only fixture, a 10 ms Moco problem with 10 mesh
intervals, 300 maximum iterations, convergence $10^{-6}$ and constraint
$10^{-7}$ reaches a $+5\times10^{-6}$ rad endpoint. It exports actual
native muscle controls at all 21 solution knots to T01 and independently
replays a fresh source model. Maximum solution-to-replay differences are:

| State Group  |                 Maximum Absolute Error |
| ------------ | -------------------------------------: |
| Coordinates  |   $1.934629343391947\times10^{-7}$ rad |
| Speeds       | $3.826581494812548\times10^{-9}$ rad/s |
| Activation   |     $1.6462886609502903\times10^{-11}$ |
| Fiber Length |     $4.993112409645839\times10^{-9}$ m |

The fixture declares a $10^{-9}$ native q/u constraint residual tolerance;
all replay knots pass. The independent coordinate replay ends within
$2\times10^{-7}$ rad of its target. These are source-specific numerical
criteria, not physiological tolerances or private marker matching. The
locked variant remains replay-capable but OpenSim 4.6 Moco rejects its
locked coordinate during solver initialization. No silent weld reduction or
lock-target inference is used.

## Reproduction and Boundaries

Set `CASADIPATH` to the pinned OpenSim 4.6 installation's package directory
and run the native suite serially with its Python 3.12 interpreter:

```text
python -m pytest tests/opensim/test_native_constrained_muscle.py -q -x
python -m pytest tests/opensim/test_native_moco_runner.py tests/opensim/test_native_prepared_state.py tests/opensim/test_native_muscle_bundle.py -q -x
python -m scripts.check_design_manual_governance
```

The 13 focused actual-native tests pass. Source entrypoint hashing assumes a
self-contained XML model and does not certify external file closure. The
related native Moco runner, prepared-state, and prior muscle-bundle suite
passes 64 tests with one explicit opt-in physical-source skip. Ruff, direct
Mypy, canonical manual governance, Semgrep/import policy, and the other
central pre-PR checks pass. The central generic interpreter lacks OpenSim;
its affected suite reports **241 passed, 35 skipped, one failed**. The sole
failure is the inherited SG optimizer `test_cli_end_to_end` bare-flag versus
required-`run` parser mismatch. The SG source and test Git blobs are exactly
`705f8bee8fbd66f85cea8d4691d7a95cf53d7446` and
`c30792cba4ebde726b6a371872b2e44146673600`, identical to the
independently reproduced unchanged-main baseline retained in
`F07_NATIVE_MOCO_RUNNER_TURNOVER.md` at main
`50784017c607f764e9c28db2d3c66b8211211f7a`. This issue does not edit
either path; the affected-test gate is truthfully not counted as passed.

Native continuous states plus lock targets are a declared cold start, not arbitrary
SimTK numerical/discrete state serialization. Sampled chart checks do not
prove internal integrator stages stayed inside the chart. This profile
excludes nonlinear/conditional/wrapped path laws, contact, other forces,
controllers, prescribed coordinates, and mixed mechanical assistance.

The unchanged 520-muscle source still needs approved source anatomy and
passive-force disposition, complete native state preparation, private marker
registration, contact/grip/constraint policy and assistance exclusion. The
OpenSim locked-coordinate Moco limitation requires a separately verified,
target-preserving source reduction. No production solve, physiological
release, private capture fit, all-variant parity, or full-horizon muscular
OpenSim replay is claimed by this child.
