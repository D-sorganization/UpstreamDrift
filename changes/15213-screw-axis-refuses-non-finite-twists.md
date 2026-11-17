---
issue: 15213
summary: "compute_screw_axis refuses non-finite twists, bad thresholds and bad endpoint lengths instead of returning NaN screw axes."
---

`screw_theory.kinematics.compute_screw_axis` now validates the twist's angular, linear and reference-point vectors (finite, shape `(3,)`) and the `singularity_threshold` (finite, positive); `compute_screw_endpoints` validates `length` (finite, non-negative). Previously a NaN or infinite twist produced a NaN `ScrewAxis` flagged `is_singular=False`. Adds contract tests and a round-trip check that a twist built from a chosen axis, point and pitch is recovered exactly.
