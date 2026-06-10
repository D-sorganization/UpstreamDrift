# Cross-engine contact models

This note documents the qualitative differences between the contact models of
the physics engines used in this project. The corresponding **numeric**
invariants are enforced in
`tests/integration/test_contact_cross_engine.py` (issue #7153) — the prose here
is reference material, not a test.

## MuJoCo — soft penalty (spring-damper)

- **Type:** soft penalty-based (spring-damper).
- **Parameters:** `solref` / `solimp` per geom, plus `<option>` settings such as
  `impratio` (frictional-to-normal impedance) and `noslip_iterations`.
- **Pros:** fast, stable, handles complex geometries.
- **Cons:** not perfectly rigid — visible penetration is allowed.
- **Energy:** dissipative (configured via damping).
- **Test tolerance:** penetration up to `5e-3 m` (`_MUJOCO_PENETRATION_TOL_M`).

References: MuJoCo docs (Contact Modeling); Todorov (2014), "Convex and smooth
formulations…".

## Drake — rigid / compliant

- **Type:** hybrid compliant + time-stepping rigid; point contact or
  hydroelastic (pressure field).
- **Pros:** physically accurate, well documented.
- **Cons:** more complex to configure.
- **Energy:** can be conservative or dissipative.
- **Test tolerance:** penetration under `1e-3 m` (`_RIGID_PENETRATION_TOL_M`) —
  an order of magnitude tighter than MuJoCo, which is the documented "rigid vs
  soft" distinction made concrete.

References: Drake docs (Multibody Dynamics); Elandt et al. (2019), "A pressure
field model…".

## Pinocchio — constraint-based (algorithmic)

- **Type:** constraint-based; contact forces from constraint resolution
  (contact LCP / QP).
- **Pros:** mathematically rigorous.
- **Cons:** requires explicit contact-point specification; no built-in penalty
  floor.
- **Energy:** depends on solver configuration.
- **Test invariant:** free-flight forward dynamics conserve mechanical energy to
  `O(dt)`, the correctness precondition for any subsequent constraint
  resolution. Note Pinocchio has **no** `computeTotalEnergy`; use
  `computeKineticEnergy` + `computePotentialEnergy` separately.

References: Pinocchio docs (Dynamics); Carpentier et al. (2019), "Pinocchio:
fast algorithms…".

## Recommendation

Use MuJoCo for simulation, Drake for trajectory optimization, and Pinocchio for
analytical/kinematic dynamics (where contact is less critical).
