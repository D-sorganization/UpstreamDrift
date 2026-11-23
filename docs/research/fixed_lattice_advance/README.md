# Fixed-Lattice RK4 Prefix Diagnostic

`motion_matching.fixed_lattice_advance.FixedLatticeAdvance` evaluates a single
fixed-effort hold from cached, fixed-RK4-substep endpoints. Interior queries
start at the preceding endpoint rather than repartitioning the whole prefix.
Caller-supplied projected q/v must match the cached end exactly or construction
fails; without a supplied endpoint the projection state is unchecked. Use a
separate lattice for each effort hold, impact split and projection boundary.

All coverage remains uncertified. Fixed-lattice anchoring removes one known
partition seam, but floating-point endpoint identity does not prove continuity
inside a step, bound ODE/RK4 error, constrain MuJoCo's state-dependent contact
regimes, or certify motion between sampled times. The evaluator is not a
validated dense solution and does not establish a first root or feasible shot.
See the editable LaTeX reference for scope and numerical assumptions.

Run headlessly with `python -m pytest -q
tests/unit/motion_matching/test_fixed_lattice_advance_15493.py`. The MuJoCo
regression uses a synthetic slider, not the pinned best-match golfer.
