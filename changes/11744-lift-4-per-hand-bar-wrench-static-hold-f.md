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
