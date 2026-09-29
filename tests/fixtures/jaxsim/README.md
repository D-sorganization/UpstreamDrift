# JaxSim Test Fixtures

Minimal model descriptions used by the JaxSim gates. These are intentionally
tiny: the gates verify cross-engine _conventions_ and integrated forward-sim
behaviour, not full humanoid coverage.

| Fixture           | Description                                                                             | Used by                                                              |
| ----------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `single_link.sdf` | A single free rigid body with unit mass and identity inertia, no joints, no collisions. | `tests/unit/test_jaxsim_optional_dependency.py` (smoke load + step). |

The forward-sim and parity gates
(`tests/cross_engine/test_jaxsim_forward_sim.py`,
`tests/cross_engine/test_jaxsim_vs_pinocchio.py`) build their single-body SDF
inline so the asymmetric diagonal inertia stays adjacent to the analytic
reference that consumes it.

All fixtures are SDF, not URDF: URDF→SDF conversion needs Linux `gz`/`ign`
sdformat tooling and is tracked separately by the gated issue #6648.
