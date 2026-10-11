# MyoSuite Native Muscle Actuator Classification Turnover — F09q #12249

## Change

`MyoSuiteMuscleAnalyzer` previously treated numeric dynamic type `2` as the
muscle enum. On MuJoCo 3.8.0, native `mjDYN_MUSCLE` is `4` and
`mjDYN_FILTER` is `2`; the analyzer therefore selected a filter and omitted
the muscle in a mixed model. The classifier now compares against the installed
`mujoco.mjtDyn.mjDYN_MUSCLE` enum. Existing compatibility fallback behavior is
unchanged.

## Native Regression

`tests/unit/engines/myosuite/test_muscle_analysis.py` builds a native model
containing one muscle, one filtered actuator, and one motor. It asserts the
compiled enum identities and verifies that only the muscle is returned. The
test instantiates the analyzer directly and does not depend on Gym or MyoSuite
task registration.

The fixture passed with Python 3.12.10 and MuJoCo 3.8.0. This establishes
actuator dynamic-type classification on that runtime only. It does not establish
muscle physiology, capture mapping, excitation replay, or cross-engine parity.

## Validation

The regression failed before the fix by selecting actuator ID 1 (the filter)
instead of ID 0 (the muscle), then passed after the enum comparison. The
combined focused run reported one native test passed and six legacy MyoSuite
integration tests skipped because MyoSuite/Gym is unavailable. Ruff check and
format, the design-manual governance check, SPEC duplicate check, and tracked
file-size budget passed. A full-repository title-case scan reports existing
violations outside this change; no whole-repository title edits were made.
