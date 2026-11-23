# Same-Input Replay Unit Scoring

The v2 replay and closed-loop receipts separate rotational coordinate error in radians from translational error in metres. Units come from each coordinate's joint primitive in the embedded model specification; an incomplete, duplicate or unsupported primitive inventory is rejected. The previous v1 scalar compared unlike units and must not be interpreted as a rotational maximum. Existing v1 receipts remain historical records; regenerate them to obtain v2 results.

The first failed recorded state determines the coordinate horizon using separate numerical parity thresholds of 1e-6 rad and 1e-6 m. Frame-position error has a 1e-3 m threshold but is evaluated only at the selected frame stride. The reported horizon is the first recorded failure, not a continuous-time guarantee. These are numerical engine-parity bounds, not anatomical limits or proof of a matched swing. The growth exponent applies to the rotational error history only. Segmented output reports rotation and translation separately.

Run `python -m pytest -q tests/unit/motion_matching/same_input/test_bundle_integrator.py tests/unit/motion_matching/same_input/test_bundle_ball_impact.py tests/unit/motion_matching/same_input/test_effort_hold.py` with the repository's headless environment. The accompanying standalone LaTeX reference is not the engineering design manual.
