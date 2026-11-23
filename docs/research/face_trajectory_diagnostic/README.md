# Sampled Finite-Face Trajectory Diagnostic

The experimental `motion_matching.face_trajectory_diagnostic.diagnose` uses
the static finite-face gap and a caller-provided advancing trajectory to scan
a bounded interval. It rejects a nonpositive initial gap, refines a sampled
positive-to-nonpositive bracket, and returns a candidate only for a front-side,
strict triangle-interior point with positive material-point approach speed.
State and velocity come from the caller's advance callback, not interpolation.

Every result is **unqualified** and its coverage **uncertified**. Sampling can
miss earlier narrow crossings or tangencies; the recorded sampled minimum is
only over samples inspected up to the first sampled bracket, not a global
closest approach. No outside-solid, mesh topology, physical velocity bound,
continuous-path coverage or first-root claim is made. A triangle seam is
conservatively rejected even when it is internal to a face. This diagnostic
must not trigger impact, flight, or admission.

The pinned-v2 MuJoCo test uses a synthetic closure-consistent state and ball,
not a matched golfer or baseline. See the LaTeX reference for the calculation.
Run `python -m pytest -q
tests/unit/motion_matching/test_face_trajectory_diagnostic_15493.py` with
headless MuJoCo configured.
