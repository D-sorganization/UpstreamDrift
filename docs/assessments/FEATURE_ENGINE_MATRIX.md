# Feature × Engine Support Matrix

## Golf Modeling Suite - January 2026

**Last Updated:** 2026-01-09  
**Source:** Assessment C (Cross-Engine Validation)

---

## Quick Reference Matrix

| Feature                     | MuJoCo | Drake | Pinocchio | Pendulum | OpenSim | MyoSuite |
| --------------------------- | :----: | :---: | :-------: | :------: | :-----: | :------: |
| **Forward Dynamics**        |   ✅   |  ✅   |    ✅     |    ✅    |   ⚠️    |    ⚠️    |
| **Inverse Dynamics**        |   ✅   |  ✅   |    ✅     |    ✅    |   ❌    |    ❌    |
| **Mass Matrix M(q)**        |   ✅   |  ✅   |    ✅     |    ✅    |   ❌    |    ❌    |
| **Jacobians (Body)**        |   ✅   |  ✅   |    ✅     |    ⚠️    |   ⚠️    |    ⚠️    |
| **Drift-Control Decomp**    |   ✅   |  ✅   |    ✅     |    ✅    |   ✅    |    ✅    |
| **Contact/Collision**       |   ✅   |  ✅   |    ⚠️     |    ❌    |   ❌    |    ⚠️    |
| **Closed-Loop Constraints** |   ✅   |  ✅   |    ✅     |    ❌    |   ❌    |    ❌    |
| **Muscle Models (Hill)**    |   ❌   |  ❌   |    ❌     |    ❌    |   ✅    |    ✅    |
| **Grip Modeling**           |   ✅   |  ❌   |    ❌     |    ❌    |   ✅    |    ✅    |
| **Neural Control (RL)**     |   ⚠️   |  ⚠️   |    ❌     |    ❌    |   ❌    |    ✅    |

**Legend:**

- ✅ Fully Supported
- ⚠️ Partially Supported
- ❌ Not Supported

---

## Engine Selection Guide

### For Rigid Body Dynamics (No Muscles)

| Use Case                 | Recommended Engine | Reason               |
| ------------------------ | ------------------ | -------------------- |
| **Batch simulations**    | MuJoCo             | Fastest (0.5ms/step) |
| **High precision**       | Drake              | 3rd-order integrator |
| **Analytical Jacobians** | Pinocchio          | Exact derivatives    |
| **Ground truth**         | Pendulum           | Analytical solution  |

### For Muscle-Driven Biomechanics

| Use Case              | Recommended Engine | Reason                    |
| --------------------- | ------------------ | ------------------------- |
| **Hill-type muscles** | OpenSim            | Industry standard         |
| **RL training**       | MyoSuite           | Native support            |
| **Cross-validation**  | OpenSim + MyoSuite | Different implementations |

---

## Performance Characteristics

| Engine        | Step Time (avg) | Memory  | GPU Support |
| ------------- | --------------- | ------- | ----------- |
| **MuJoCo**    | 0.5 ms          | Low     | Yes (3.3+)  |
| **Drake**     | 15 ms           | Medium  | Limited     |
| **Pinocchio** | 1 ms            | Low     | No          |
| **Pendulum**  | 0.01 ms         | Minimal | N/A         |
| **OpenSim**   | 10 ms           | High    | No          |
| **MyoSuite**  | 2 ms            | Medium  | Yes         |

---

## Integration Method Comparison

| Engine        | Integrator          | Order | Stability     | Recommended dt |
| ------------- | ------------------- | ----- | ------------- | -------------- |
| **MuJoCo**    | Semi-implicit Euler | 1st   | Unconditional | < 0.01s        |
| **Drake**     | RK3                 | 3rd   | Conditional   | < 0.001s       |
| **Pinocchio** | Euler               | 1st   | Conditional   | < 0.0005s      |
| **Pendulum**  | Analytical          | N/A   | N/A           | Any            |

---

## Cross-Validation Status

### Currently Validated (CI Runs on Every PR)

| Engine Pair        | Validation Level | Status    |
| ------------------ | ---------------- | --------- |
| MuJoCo ↔ Pinocchio | Full suite       | ✅ Active |
| MuJoCo ↔ Drake     | Full suite       | ✅ Active |
| Drake ↔ Pinocchio  | Full suite       | ✅ Active |

### Nightly Validation

| Engine Pair   | Validation Level | Status                |
| ------------- | ---------------- | --------------------- |
| All 6 engines | Comprehensive    | ✅ Nightly @ 2 AM UTC |

---

## Tolerance Targets (Guideline P3)

| Metric           | Tolerance | Units         |
| ---------------- | --------- | ------------- |
| **Position**     | ± 1e-6    | meters        |
| **Velocity**     | ± 1e-5    | m/s           |
| **Acceleration** | ± 1e-4    | m/s²          |
| **Torque**       | ± 1e-3    | N·m           |
| **Jacobian**     | ± 1e-8    | dimensionless |

---

## Feature Details

### Forward Dynamics

| Engine        | Method          | Notes                         |
| ------------- | --------------- | ----------------------------- |
| **MuJoCo**    | CRBA + Contacts | Full contact handling         |
| **Drake**     | CRBA + Contacts | Full constraint solving       |
| **Pinocchio** | ABA             | Pure rigid body               |
| **OpenSim**   | SimTK           | Stub only (integrate pending) |
| **MyoSuite**  | MuJoCo Backend  | Muscle layer on MuJoCo        |

### Muscle Integration

| Engine       | Muscle Model       | Activation Dynamics | Wrapping           |
| ------------ | ------------------ | ------------------- | ------------------ |
| **OpenSim**  | Hill-type (Thelen) | First-order ODE     | Cylinder/Ellipsoid |
| **MyoSuite** | MuJoCo actuators   | Numerical           | Via points         |

### Contact Physics

| Engine        | Method            | Friction | Soft Contacts |
| ------------- | ----------------- | -------- | ------------- |
| **MuJoCo**    | Convex collisions | Coulomb  | Yes           |
| **Drake**     | Point contacts    | Coulomb  | Yes           |
| **Pinocchio** | Limited           | N/A      | No            |

---

## Known Limitations

### MuJoCo

- No analytical Hill muscles
- GPU requires MuJoCo 3.3+

### Drake

- 30× slower than MuJoCo
- Memory intensive for large models

### Pinocchio

- Contact support limited
- Requires manual collision geometry

### OpenSim

- Python bindings require separate install
- Simulation integration still developing

### MyoSuite

- Requires gym environment
- Limited to MuJoCo-based models

---

## Related Documents

- [Assessment C (Full Analysis)](./Assessment_C_Cross_Engine_Jan2026.md)
- [Cross-Engine Troubleshooting](./troubleshooting/cross_engine_deviations.md)
- [Engine Selection Guide](../engine_selection_guide.md)
- [Project Design Guidelines](./project_design_guidelines.qmd)

---

_This matrix is auto-updated based on engine capability tests._
