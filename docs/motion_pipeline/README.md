# Motion Pipeline — User Workflow Guide

> Part of epic #4558. Make the new pipeline immediately usable by anyone joining the project.

## From Video to Tracked Golf Swing in 5 Commands

This guide walks you through processing a golf swing video into a fully tracked motion using the UpstreamDrift motion pipeline.

### Step 1: Extract Frames from Video

```bash
python3 -m src.shared.python.motion_pipeline.extract_frames \
    --input swing_video.mp4 \
    --output frames/ \
    --fps 60
```

### Step 2: Run Pose Estimation (OpenPose or MediaPipe)

```bash
# Option A: OpenPose (higher accuracy, requires GPU)
python3 -m src.shared.python.motion_pipeline.pose_estimation \
    --input frames/ \
    --output pose_2d.json \
    --engine openpose

# Option B: MediaPipe (CPU-friendly, faster)
python3 -m src.shared.python.motion_pipeline.pose_estimation \
    --input frames/ \
    --output pose_2d.json \
    --engine mediapipe
```

### Step 3: Lift 2D to 3D (Optional — Use Existing Mocap)

```bash
python3 -m src.shared.python.motion_pipeline.lift_3d \
    --input pose_2d.json \
    --output motion_3d.json \
    --method triangulation  # or 'learned' for ML-based lifting
```

### Step 4: Retarget to Humanoid Model

```bash
python3 -m src.shared.python.motion_pipeline.retarget \
    --input motion_3d.json \
    --output retargeted_motion.json \
    --model preset:golfer_standard
```

### Step 5: Run Inverse Kinematics and Export

```bash
python3 -m src.shared.python.motion_pipeline.inverse_kinematics \
    --input retargeted_motion.json \
    --output final_motion.mocap \
    --engine mujoco  # or drake, pinocchio
```

---

## Worked Example for Each Engine

### MuJoCo Pipeline

```bash
# Complete MuJoCo workflow
python3 scripts/motion_pipeline/run_mujoco_pipeline.py \
    --video swing_video.mp4 \
    --output mujoco_output/
```

### Drake Trajectory Optimization

```bash
# Drake trajopt workflow
python3 scripts/motion_pipeline/run_drake_pipeline.py \
    --input motion_3d.json \
    --output drake_trajopt/
```

### Pinocchio Inverse Kinematics

```bash
# Pinocchio IK workflow
python3 scripts/motion_pipeline/run_pinocchio_pipeline.py \
    --input motion_3d.json \
    --output pinocchio_ik/
```

---

## When to Choose Each Engine

| Use Case | Recommended Engine | Why |
|----------|-------------------|-----|
| **Contact-rich dynamics** (ground reaction, ball impact) | MuJoCo | Best contact handling, day-to-day development |
| **Trajectory optimization** (finding optimal swing) | Drake | Built-in trajopt solvers, system analysis |
| **Fast IK solutions** (real-time retargeting) | Pinocchio | Optimized rigid-body algorithms |
| **Biomechanics validation** | OpenSim | Gold-standard musculoskeletal models |
| **Muscle dynamics** | MyoSuite | Detailed muscle activation modeling |

---

## Next Steps

- [Format Matrix](formats.md) — Auto-generated support matrix for each mocap source
- [Troubleshooting](troubleshooting.md) — Common failure modes and fixes
- [Architecture ADR](../adr/0007-motion-pipeline-architecture.md) — Design decisions and alternatives