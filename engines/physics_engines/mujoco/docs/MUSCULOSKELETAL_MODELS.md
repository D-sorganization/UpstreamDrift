# Musculoskeletal Models Integration

## Overview

The MuJoCo Golf Swing Simulator now includes complete human musculoskeletal models from the [MyoSuite](https://github.com/MyoHub/myosuite) framework. These models provide physiologically realistic muscle-tendon units, biomechanical constraints, and anatomically accurate body segments for advanced golf swing analysis.

## Available Models

### 1. MyoUpperBody (19 DOF, 20 Actuators)

**Description:** Comprehensive upper body musculoskeletal model including torso, head, and bilateral arms.

**Specifications:**
- **Degrees of Freedom:** 19
- **Actuators:** 20 (muscle-based torque actuators)
- **Bodies:** 15 anatomical segments
- **Joints:** 19

**Body Segments:**
- **Torso:** Sacrum, lumbar spine, thoracic spine
- **Arms (bilateral):** Humerus, ulna, radius, hand
- **Head:** Included with neck articulation

**Joint Structure:**
- **Lumbar spine (3 DOF):**
  - Extension/flexion
  - Lateral bending
  - Axial rotation

- **Shoulders (3 DOF each):**
  - Flexion/extension
  - Adduction/abduction
  - Internal/external rotation

- **Elbows (1 DOF each):**
  - Flexion/extension

- **Forearms (1 DOF each):**
  - Pronation/supination

- **Wrists (2 DOF each):**
  - Flexion/extension
  - Radial/ulnar deviation

**Muscles:**
- Right arm: 7 actuators (shoulder, elbow, forearm, wrist)
- Left arm: 7 actuators (shoulder, elbow, forearm, wrist)
- Torso: 6 actuators (erector spinae, obliques)

**Best For:** Upper body golf swing biomechanics, bilateral arm coordination, torso rotation analysis

---

### 2. MyoBody Full (52 DOF, 290 Actuators)

**Description:** Complete full-body musculoskeletal model with comprehensive muscle-tendon units for whole-body biomechanical analysis.

**Specifications:**
- **Degrees of Freedom:** 52
- **Actuators:** 290 muscle-tendon units
- **Bodies:** 26 anatomical segments
- **Joints:** 47

**Body Segments:**
- **Lower body:** Pelvis, femur, tibia, fibula, foot (bilateral)
- **Torso:** Sacrum, lumbar vertebrae, thoracic spine, ribs
- **Upper body:** Scapulae, clavicles, humeri, forearms, hands (bilateral)
- **Head and neck**

**Muscle Groups (290 total):**
- **Spine muscles:** Psoas major, iliocostalis, multifidus, etc. (~40)
- **Hip muscles:** Gluteus maximus/medius/minimus, iliopsoas, etc. (~30)
- **Thigh muscles:** Quadriceps, hamstrings, adductors, etc. (~40)
- **Shank muscles:** Gastrocnemius, soleus, tibialis, etc. (~30)
- **Torso muscles:** Rectus abdominis, obliques, erector spinae, etc. (~30)
- **Shoulder muscles:** Deltoids, rotator cuff, etc. (~40)
- **Arm muscles:** Biceps, triceps, brachialis, etc. (~40)
- **Forearm/hand muscles:** (~40)

**Physiological Features:**
- Realistic muscle force-length relationships
- Tendon compliance and elasticity
- Moment arm variations with joint angles
- Muscle activation dynamics

**Best For:** Complete golf swing analysis including weight transfer, lower body kinetics, ground reaction forces, and full kinematic chain

---

### 3. MyoArm Simple (14 DOF, 14 Actuators)

**Description:** Simplified bilateral arm model with minimal torso for focused upper extremity analysis.

**Specifications:**
- **Degrees of Freedom:** 14
- **Actuators:** 14
- **Bodies:** 10
- **Joints:** 14

**Body Segments:**
- Simplified torso (ribs and scapulae)
- Bilateral arms: Humerus, ulna, radius, hand

**Joint Structure (per arm):**
- Shoulder (3 DOF): Flexion, adduction, rotation
- Elbow (1 DOF): Flexion
- Forearm (1 DOF): Pronation/supination
- Wrist (2 DOF): Flexion, deviation

**Best For:** Rapid prototyping, arm coordination studies, reduced computational complexity

---

## Model Source and Conversion

### Origin
All musculoskeletal models are derived from the [MyoSuite](https://github.com/MyoHub/myosuite) framework, which provides physiologically accurate models converted from OpenSim using the MyoConverter tool.

### Conversion Process
1. **Base model:** OpenSim MoBL-ARMS (upper extremity) and Rajagopal (lower extremity)
2. **Mesh conversion:** Bone geometries converted to MuJoCo-compatible formats
3. **Moment arm optimization:** Muscle wrapping and via points optimized for kinematic accuracy
4. **Force optimization:** Muscle force-length curves matched to physiological data
5. **Manual refinement:** Contact geometries and joint constraints refined for stability

### Performance
- **Speed advantage:** 60-4000× faster than equivalent OpenSim models
- **Accuracy:** Kinematics and muscle moment arms match OpenSim within 2-5%
- **Stability:** Optimized for contact-rich simulations and reinforcement learning

---

## Using Musculoskeletal Models

### GUI Selection

1. Launch the simulator:
   ```bash
   cd python
   python -m mujoco_golf_pendulum
   ```

2. Select a musculoskeletal model from the dropdown:
   - "Musculoskeletal: Upper body (19 DOF, 20 muscles)"
   - "Musculoskeletal: Full body (52 DOF, 290 muscles)"
   - "Musculoskeletal: Bilateral arms (14 DOF)"

3. The model will load with appropriate actuator controls for each muscle/actuator

### Programmatic Usage

```python
from mujoco_golf_pendulum.sim_widget import MuJoCoSimWidget
from mujoco_golf_pendulum.models import MYOUPPERBODY_PATH, MYOBODY_PATH

# Create simulation widget
sim = MuJoCoSimWidget(width=1280, height=720, fps=60)

# Load upper body model
sim.load_model_from_file(MYOUPPERBODY_PATH)

# Or load full body model
sim.load_model_from_file(MYOBODY_PATH)

# Access model data
print(f"DOF: {sim.model.nv}")
print(f"Muscles: {sim.model.nu}")
```

### Direct MuJoCo Access

```python
import mujoco

# Load model directly
model = mujoco.MjModel.from_xml_path("myo_sim/body/myoupperbody.xml")
data = mujoco.MjData(model)

# Simulate
for _ in range(1000):
    mujoco.mj_step(model, data)
```

---

## Biomechanical Analysis Capabilities

### Muscle Activation Analysis
- Real-time muscle activation patterns
- Co-contraction analysis
- Synergy extraction
- Fatigue modeling (future enhancement)

### Force and Torque Analysis
- Joint reaction forces
- Muscle forces and moments
- Contact forces (ball-club, foot-ground)
- Inverse dynamics for required activations

### Kinematic Analysis
- Segment positions, velocities, accelerations
- Joint angles and angular velocities
- End-effector trajectories (clubhead, ball)
- Center of mass dynamics

### Performance Metrics
- Clubhead speed and trajectory
- Ball launch angle and speed
- Power generation and transfer
- Timing and sequencing analysis

---

## Advantages Over Traditional Models

| Feature | Traditional Golf Models | Musculoskeletal Models |
|---------|------------------------|------------------------|
| **Actuators** | Direct torque motors | Muscle-tendon units |
| **Physiological Realism** | Limited | High - includes muscle dynamics |
| **Force Generation** | Instantaneous | Subject to activation dynamics |
| **Muscle Coordination** | Not modeled | Explicit co-activation patterns |
| **Fatigue Effects** | Not available | Can be modeled |
| **Research Validity** | Good for kinematics | Better for biomechanics |
| **Computational Cost** | Lower | Higher (but optimized) |

---

## Research Applications

### Golf Biomechanics
- Study muscle recruitment patterns during different swing phases
- Analyze power generation and transfer through kinematic chain
- Investigate injury mechanisms and prevention strategies
- Optimize swing mechanics for different player anthropometries

### Sports Science
- Validate muscle coordination strategies
- Test hypothesis about optimal swing sequences
- Develop training protocols based on muscle activation
- Study bilateral asymmetries and compensations

### Rehabilitation
- Model post-injury swing mechanics
- Design rehabilitation exercises targeting specific muscles
- Predict return-to-play readiness
- Optimize adaptive equipment

### Robotics and Control
- Develop bio-inspired control strategies
- Test reinforcement learning algorithms
- Study human-like motion generation
- Benchmark against human performance

---

## Technical Details

### Model Files Structure
```
myo_sim/
├── arm/
│   ├── myoarm.xml                    # Full arm model (27 DOF, 63 muscles)
│   ├── myoarm_simple.xml             # Simplified bilateral (14 DOF)
│   └── assets/                       # Muscle definitions and geometries
├── body/
│   ├── myoupperbody.xml              # Upper body (19 DOF, 20 actuators)
│   ├── myobody.xml                   # Full body (52 DOF, 290 muscles)
│   └── assets/                       # Body assets and muscle definitions
├── leg/
│   └── assets/                       # Leg muscles and definitions
├── meshes/                           # STL bone meshes
├── textures/                         # Visual textures
└── scene/                            # Scene lighting and environment
```

### Actuator Types
- **General actuators with muscle dynamics:**
  - `dyntype="muscle"`: First-order activation dynamics
  - `gaintype="muscle"`: Force-length-velocity relationships
  - `biastype="muscle"`: Passive elastic forces

### Integration with Existing Code
- **Backward compatible:** All existing golf models still work
- **Unified interface:** Same API for all models
- **Automatic adaptation:** GUI dynamically creates controls for any model
- **Analysis tools:** Biomechanical analyzers work with all models

---

## Future Enhancements

### Planned Features
1. **Golf-specific muscle tuning:** Optimize muscle parameters for golf athletes
2. **Club attachment:** Add golf club to musculoskeletal hand models
3. **Ground contact:** Enhance foot-ground interaction models
4. **Muscle fatigue:** Implement fatigue models for endurance analysis
5. **Personalization:** Tools to adapt models to individual anthropometry
6. **EMG validation:** Compare simulated activations with measured EMG

### Research Opportunities
- Muscle coordination pattern analysis during golf swing
- Power generation and transfer efficiency studies
- Injury risk assessment and prevention
- Performance optimization through muscle strengthening targets
- Age and skill level biomechanical differences

---

## References

### MyoSuite Framework
- **Paper:** Caggiano et al. (2022). "MyoSuite: A contact-rich simulation suite for musculoskeletal motor control"
- **GitHub:** https://github.com/MyoHub/myosuite
- **Models:** https://github.com/MyoHub/myo_sim

### Model Conversion
- **MyoConverter:** https://github.com/MyoHub/myoconverter
- **Performance:** Wang et al. (2022). "MyoSim: Fast and physiologically realistic MuJoCo models"

### Base Models
- **Upper extremity:** MoBL-ARMS (Chadwick et al., 2014)
- **Full body:** Rajagopal et al. (2016) - OpenSim full body model
- **Muscle dynamics:** Hill-type muscle model

---

## License

The musculoskeletal models are provided under the **Apache License 2.0** by the MyoSuite team (Vikash Kumar, Vittorio Caggiano, and collaborators).

See the [MyoSuite License](https://github.com/MyoHub/myosuite/blob/main/LICENSE) for full details.

---

## Support and Citation

If you use these musculoskeletal models in your research, please cite:

```bibtex
@article{caggiano2022myosuite,
  title={MyoSuite: A contact-rich simulation suite for musculoskeletal motor control},
  author={Caggiano, Vittorio and Wang, Huiyi and Durandau, Guillaume and Sartori, Massimo and Kumar, Vikash},
  journal={arXiv preprint arXiv:2205.13600},
  year={2022}
}
```

For questions or issues specific to the golf swing integration, please open an issue on the project repository.
