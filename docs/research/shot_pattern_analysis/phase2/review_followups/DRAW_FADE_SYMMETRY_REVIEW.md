# Draw/Fade Symmetry Review

Do not claim general draw/fade distribution symmetry for the phase-two study.
Symmetric targets and mirrored zero-error nominals are insufficient: the full
club geometry, input distribution, contact conventions, and scoring map must
also transform correctly under reflection.

## Fixed Right-Handed Shaft Coupling

`club_geometry.py` pitches and yaws a right-handed shaft, then applies twist
about that shaft. At zero twist the previously verified derivatives are
`d(face)/dθ = −sin(β) cos(L) / cos(L−lean)` and `d(loft)/dθ = −cos(β)`.
Both are nonzero for these presets. A lateral reflection would reverse the
face perturbation while preserving loft. Replacing θ by −θ reverses both;
keeping θ preserves both. Therefore this one-dimensional twist perturbation
family is not itself closed under that reflection for fixed handedness.

This issue exists even with diagonal primitive covariance. In particular,
nonzero twist variance induces a face/loft covariance whose sign would have
to reverse under reflection. Arbitrary covariance adds further restrictions;
for a genuine primitive sign transformation D, the covariance must transform
as `D Σ Dᵀ`, not remain silently unchanged.

A pure public-geometry evaluation at zero lean/yaw, without flight simulation:

| Club   | Twist (°) | Delivered Face (°) | Delivered Loft (°) |
| ------ | --------: | -----------------: | -----------------: |
| Driver |        −1 |        +0.85335046 |        10.52247926 |
| Driver |        +1 |        −0.85197935 |         9.47752070 |
| 7-Iron |        −1 |        +0.89341907 |        34.45397213 |
| 7-Iron |        +1 |        −0.88865672 |        33.54602773 |
| PW     |        −1 |        +0.90240065 |        46.43835306 |
| PW     |        +1 |        −0.89527894 |        45.56164673 |

The static loft/lie inputs are the study presets (10°/58.5°, 34°/63°, 46°/64°).
These examples establish a geometry-level obstruction, not a predicted sign or
magnitude of final draw/fade differences. Sampling error remains possible;
structural differences must not be dismissed as sampling error by assumption.

## Existing Coverage and Limits

The contact unit test `test_rotation_covariance_with_offcenter_spin` and
`independent_contact_audit.py` check proper rotations. They do not establish
improper spatial-reflection covariance, nor reflection invariance of the full
right-handed delivery/noise pipeline. Mirrored nominal probes establish only
the zero-error cases they execute.

A future low-cost contact reflection check should reflect polar vectors
(positions, normals, impulses, translational velocity) with R, axial vectors
(angular velocity and torque) with `det(R) R`, and inertia with `R I Rᵀ`.
Even a passing contact reflection test would not prove stochastic study
symmetry without the corresponding handedness/geometry and input-law mapping.
No new full-flight parity run was needed for this source-level conclusion.
