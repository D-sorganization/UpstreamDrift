# Feature × Engine Support Matrix

## UpstreamDrift - February 2026

**Last Updated:** 2026-02-01
**Source:** Comprehensive Codebase Analysis
**Status:** All Engines Fully Implemented

---

## Quick Reference Matrix

| Feature                       | MuJoCo | Drake | Pinocchio | Pendulum | OpenSim | MyoSuite |
| ----------------------------- | :----: | :---: | :-------: | :------: | :-----: | :------: |
| **Forward Dynamics**          |   ✅   |  ✅   |    ✅     |    ✅    |   ✅    |    ✅    |
| **Inverse Dynamics**          |   ✅   |  ✅   |    ✅     |    ✅    |   ✅    |    ✅    |
| **Mass Matrix M(q)**          |   ✅   |  ✅   |    ✅     |    ✅    |   ✅    |    ✅    |
| **Jacobians (Body)**          |   ✅   |  ✅   |    ✅     |    ⚠️    |   ⚠️    |    ⚠️    |
| **Drift-Control Decomp**      |   ✅   |  ✅   |    ✅     |    ✅    |   ✅    |    ✅    |
| **ZTCF/ZVCF Counterfactuals** |   ✅   |  ✅   |    ✅     |    ✅    |   ✅    |    ✅    |
| **Contact/Collision**         |   ✅   |  ✅   |    ⚠️     |    ❌    |   ❌    |    ⚠️    |
| **Closed-Loop Constraints**   |   ✅   |  ✅   |    ✅     |    ❌    |   ❌    |    ❌    |
| **Muscle Models (Hill)**      |   ❌   |  ❌   |    ❌     |    ❌    |   ✅    |    ✅    |
| **Grip Modeling**             |   ✅   |  ❌   |    ❌     |    ❌    |   ✅    |    ✅    |
| **Neural Control (RL)**       |   ⚠️   |  ⚠️   |    ❌     |    ❌    |   ❌    |    ✅    |

**Legend:**

- ✅ Fully Supported
- ⚠️ Partially Supported
- ❌ Not Supported

---

## Platform-Wide Features (All Engines)

These features are available regardless of which physics engine is selected:

| Feature                     | Status | Description                                      |
| --------------------------- | ------ | ------------------------------------------------ |
| **Unified API**             | ✅     | PhysicsEngine Protocol with consistent interface |
| **State Checkpoint**        | ✅     | Save/restore simulation state                    |
| **Cross-Engine Validation** | ✅     | Automatic comparison across engines              |
| **Design by Contract**      | ✅     | Runtime contract enforcement                     |
| **Ball Flight Physics**     | ✅     | Magnus effect, environmental factors             |
| **Impact Dynamics**         | ✅     | Club-ball collision modeling                     |
| **Flexible Shaft**          | ✅     | Shaft deflection and dynamics                    |
| **Ground Reaction Forces**  | ✅     | GRF computation and analysis                     |

---

## AI-First Implementation (NEW - Beyond Original Scope)

| Feature               | Status | Description                          |
| --------------------- | ------ | ------------------------------------ |
| **OpenAI Adapter**    | ✅     | GPT-4 integration with tool use      |
| **Anthropic Adapter** | ✅     | Claude integration                   |
| **Gemini Adapter**    | ✅     | Google AI integration                |
| **Ollama Adapter**    | ✅     | Free local LLM support               |
| **Tool Registry**     | ✅     | Self-describing API for AI agents    |
| **Workflow Engine**   | ✅     | Guided multi-step analysis workflows |
| **Education System**  | ✅     | 4-level progressive disclosure       |
| **RAG System**        | ✅     | Documentation retrieval              |
| **GUI Integration**   | ✅     | Assistant panel with settings        |

---

## Data Ingestion Features

| Feature                   | Status | Description                        |
| ------------------------- | ------ | ---------------------------------- |
| **C3D Reader**            | ✅     | Motion capture data import         |
| **C3D Viewer**            | ✅     | Interactive visualization          |
| **Video Pose Estimation** | ✅     | OpenPose/MediaPipe integration     |
| **URDF Import/Export**    | ✅     | Universal robot description format |
| **Model Library**         | ✅     | Pre-built humanoid and club models |

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
| **RL training**       | MyoSuite           | Native Gym support        |
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

| Engine Pair         | Validation Level | Status    |
| ------------------- | ---------------- | --------- |
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

| Engine        | Method          | Notes                   |
| ------------- | --------------- | ----------------------- |
| **MuJoCo**    | CRBA + Contacts | Full contact handling   |
| **Drake**     | CRBA + Contacts | Full constraint solving |
| **Pinocchio** | ABA             | Pure rigid body         |
| **OpenSim**   | SimTK           | Full implementation     |
| **MyoSuite**  | MuJoCo Backend  | Muscle layer on MuJoCo  |

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

- No analytical Hill muscles (use MyoSuite for muscles)
- GPU requires MuJoCo 3.3+

### Drake

- 30× slower than MuJoCo
- Memory intensive for large models

### Pinocchio

- Contact support limited (placeholder zeros returned)
- Requires manual collision geometry

### OpenSim

- Python bindings require separate install
- Windows may need specific configuration

### MyoSuite

- Requires gym environment wrapper
- Limited to MuJoCo-based models

---

## Related Documents

- [Comprehensive Assessment](./Comprehensive_Assessment.md)
- [PATH_FORWARD](./PATH_FORWARD.md)
- [Engine Selection Guide](../engine_selection_guide.md)
- [Project Design Guidelines](../project_design_guidelines.qmd)
- [AI Implementation Plan](../ai_implementation/AI_IMPLEMENTATION_MASTER_PLAN.md)

---

_This matrix reflects the actual codebase state as of 2026-02-01._
