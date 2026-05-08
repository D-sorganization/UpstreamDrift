# Motion Pipeline — User Workflow Guide

> Part of epic #4558. Make the new pipeline immediately usable by anyone joining the project.

## Processing Motion Capture via REST API

The motion pipeline is exposed as a FastAPI REST service. You can run the server and send motion files (like C3D) directly to it.

### Step 1: Start the API Server

```bash
python -m uvicorn src.shared.python.motion_pipeline.api:create_app --factory --host 0.0.0.0 --port 8000
```

### Step 2: Run the Pipeline

You can process a file by sending a `POST` request to `/api/v1/motion-pipeline/run`:

```bash
curl -X POST http://localhost:8000/api/v1/motion-pipeline/run \
  -F "file=@your_motion_data.c3d" \
  -F "source_format=c3d" \
  -F "ik_backend=mujoco" \
  -F "matching_backend=mujoco"
```

This will return a JSON response containing the solved kinematics and motion matching metrics.

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