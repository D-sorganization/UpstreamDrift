# OpenSim Sample Models

This directory contains OpenSim (`.osim`) model files for use with the Golf Modeling Suite.

---

## Quick Start

1. Launch the Golf Modeling Suite
2. Select "OpenSim Golf" from the launcher
3. Click "Load Model" and select one of these files

---

## Official OpenSim Models (Submodule)

The `opensim-models/` directory contains the **official OpenSim models repository** from Stanford/SimTK.

### Recommended Starting Models

| Model              | Path                                                        | Description                                 |
| ------------------ | ----------------------------------------------------------- | ------------------------------------------- |
| **Arm26**          | `opensim-models/Models/Arm26/arm26.osim`                    | Simple 2-joint arm with 6 muscles per joint |
| **Pendulum**       | `opensim-models/Models/Pendulum/double_pendulum.osim`       | Simple pendulum for validation              |
| **DoublePendulum** | `opensim-models/Models/DoublePendulum/double_pendulum.osim` | Two-link pendulum                           |
| **Gait2392**       | `opensim-models/Models/Gait2392_Simbody/gait2392.osim`      | Full lower body (92 muscles)                |
| **Rajagopal**      | `opensim-models/Models/Rajagopal/Rajagopal2015.osim`        | Full body model                             |

### All Available Models

```
opensim-models/Models/
├── Arm26/              # Upper arm (26 muscles)
├── BouncingBlock/      # Contact testing
├── DoublePendulum/     # 2-DOF pendulum
├── Gait10dof18musc/    # Simplified gait
├── Gait2354_Simbody/   # 23-DOF gait
├── Gait2392_Simbody/   # 23-DOF gait (92 muscles)
├── Hamner/             # Full body running
├── Jumper/             # Jumping model
├── Leg39/              # Leg model (39 muscles)
├── Leg6Dof9Musc/       # Simplified leg
├── Pendulum/           # Single pendulum
├── Rajagopal/          # Full body (37 muscles)
├── SoccerKick/         # Kicking motion
├── ToyLanding/         # Landing mechanics
├── Tug_of_War/         # Basic muscle demo
├── WalkerModel/        # Walking model
└── WristModel/         # Wrist biomechanics
```

---

## Custom Models (Local)

### Model Sources

| Model              | Description               | URL                                                                                    |
| ------------------ | ------------------------- | -------------------------------------------------------------------------------------- |
| **OpenSim Models** | Official distribution     | [github.com/opensim-org/opensim-models](https://github.com/opensim-org/opensim-models) |
| **SimTK**          | Research community models | [simtk.org/projects/opensim](https://simtk.org/projects/opensim)                       |
| **Rajagopal 2015** | Full body 37-muscle       | Included in OpenSim 4.x                                                                |

---

## Official Tutorials

OpenSim provides comprehensive tutorials at:
https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088695/Examples+and+Tutorials

### Introductory Examples

| Tutorial                                                                                                                                                                             | Description                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| [Tutorial 1 - Musculoskeletal Modeling](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088700/Tutorial+1+-+Intro+to+Musculoskeletal+Modeling)                   | Introduction to OpenSim    |
| [Tutorial 2 - Tendon Transfer Surgery](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088644/Tutorial+2+-+Simulation+and+Analysis+of+a+Tendon+Transfer+Surgery) | Surgery simulation         |
| [Tutorial 3 - Scaling, IK, ID](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089741/Tutorial+3+-+Scaling+Inverse+Kinematics+and+Inverse+Dynamics)              | Motion analysis pipeline   |
| [OpenSense - IMU Data](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53084203/OpenSense+-+Kinematics+with+IMU+Data)                                              | Inertial measurement units |
| [Soccer Kick Example](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53084205/Soccer+Kick+Example)                                                                | Sport motion analysis      |

### Intermediate Examples

| Tutorial                                                                                                                                      | Description                    |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [Static Optimization](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085189/Working+with+Static+Optimization)            | Muscle force estimation        |
| [Computed Muscle Control](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088683/Example+-+Computed+Muscle+Control)       | Forward dynamics with tracking |
| [Joint Reaction Loads](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088669/Example+-+Estimating+Joint+Reaction+Loads)  | Internal force estimation      |
| [Joint Moments](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53084241/Computing+Joint+Moments+with+Experimental+Data)    | Inverse dynamics               |
| [Metabolic Cost](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53088704/Simulation-Based+Design+to+Reduce+Metabolic+Cost) | Energy expenditure             |

### Tutorials in the Submodule

The `opensim-models/Tutorials/` directory contains:

- IK/ID/CMC pipelines with data files
- Complete workflow examples
- Setup files for tools

---

## Model File Format

OpenSim uses XML-based `.osim` files:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="my_model">
    <BodySet>
      <objects>
        <Body name="arm">
          <mass>5.0</mass>
          <inertia>0.1 0.1 0.1</inertia>
        </Body>
      </objects>
    </BodySet>
    <JointSet>
      <objects>
        <PinJoint name="elbow">
          <!-- Joint definition -->
        </PinJoint>
      </objects>
    </JointSet>
    <ForceSet>
      <objects>
        <Thelen2003Muscle name="biceps">
          <max_isometric_force>800</max_isometric_force>
          <optimal_fiber_length>0.12</optimal_fiber_length>
        </Thelen2003Muscle>
      </objects>
    </ForceSet>
  </Model>
</OpenSimDocument>
```

---

## Creating Custom Golf Models

### Option 1: URDF Generator

Use the URDF Generator tool in the main launcher to create models visually,
then convert URDF → OpenSim format.

### Option 2: Python API

```python
import opensim

# Create a new model
model = opensim.Model()
model.setName("golf_swing")

# Add a body
arm = opensim.Body("upper_arm", 3.0, opensim.Vec3(0), opensim.Inertia(0.1))
model.addBody(arm)

# Add a joint
shoulder = opensim.PinJoint("shoulder",
    model.getGround(), opensim.Vec3(0),
    arm, opensim.Vec3(0, -0.3, 0))
model.addJoint(shoulder)

# Add a muscle
muscle = opensim.Thelen2003Muscle("deltoid", 600, 0.15, 0.05, 0.0)
muscle.addNewPathPoint("origin", model.getGround(), opensim.Vec3(0))
muscle.addNewPathPoint("insertion", arm, opensim.Vec3(0, -0.1, 0))
model.addForce(muscle)

# Save
model.printToXML("custom_golf_model.osim")
```

---

## Cross-Validation

All models should produce consistent results across engines.

```python
from shared.python.cross_engine_validator import CrossEngineValidator

# Load same model in OpenSim and MuJoCo (via URDF conversion)
# Compare inverse dynamics results
# Expected: ±1e-3 N·m agreement for torques
```

---

## Troubleshooting

### Model Won't Load

- Verify file path is correct
- Check OpenSim version compatibility (models may require specific versions)
- Ensure all referenced geometry files exist

### Missing Geometry

- Copy geometry files (`.vtp`, `.stl`) to same directory as model
- Or use absolute paths in the model file

### Performance Issues

- For large models (>100 muscles), consider:
  - Reducing simulation timestep
  - Using simplified muscle models
  - Running on MuJoCo for initial analysis

---

_For more details, see `/docs/OPENSIM_INTEGRATION.md`_
