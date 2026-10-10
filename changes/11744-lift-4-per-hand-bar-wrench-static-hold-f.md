---
issue: 11744
summary: "LIFT-4: per-hand bar wrench in a settled static hold for the MuJoCo lift pack via the shared GCV-7 grip analysis (hand forces sum to the 120 kg bar weight within 3.3e-7, even split); other engines report unavailable"
---

Pinocchio slice: since the pack URDF fuses the bar into the left hand's rigid
body (Pinocchio merges fixed-joint subtrees), the adapter instead computes
the bar's mass/COM from its URDF `<inertial>` elements transported by frame
placement, forms the static-equilibrium wrench the hands must exert, and
splits it with the shared `allocate_min_norm` (GCV-7); bar mass matches
`Anthropometry.bar_total_mass_kg` to 2.5e-8 relative and the split is even to
within 2.8e-7 on deadlift/bench_press/snatch/clean_and_jerk, with squat
correctly reported unavailable (bar welded to the torso, not the hands).

OpenSim slice: the pack welds the bar to `hand_l` with a zero-DOF
`WeldJoint` and to `hand_r` with a `WeldConstraint`, both at zero local
offset from the hand body, so (unlike Pinocchio) both hands' own frames are
their grip points directly with no bar-side anchor to disambiguate; the
adapter sums `getMass`/`getMassCenter` over the `barbell`-prefixed bodies
and reuses the same `allocate_min_norm` split (now shared via a new
`static_hold_result` helper in `bar_hold_static.py`, also adopted by
Pinocchio). Measured on the real pack: the split is even to within 1.1e-14
and the hand-grip residuals are within 2.6e-14 m on
deadlift/bench_press/snatch/clean_and_jerk, with squat again unavailable.
