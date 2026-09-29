# ADR-0007: Motion Pipeline Architecture

- Status: Proposed
- Date: 2026-05-08
- Decision Makers: Dieter Olson
- Related Issues/PRs: #4572, #4558 (epic)

## Context

The UpstreamDrift project needs a unified motion capture pipeline that can:

1. Accept input from multiple mocap sources (Theia, OpenPose, MediaPipe, BVH, C3D)
2. Process 2D poses into 3D motion
3. Retarget motion to various humanoid character models
4. Export to multiple physics engines (MuJoCo, Drake, Pinocchio, OpenSim)

Currently, each engine has its own ad-hoc mocap loading code, leading to:

- Duplicated conversion logic
- Inconsistent coordinate system handling
- Difficult onboarding for new contributors
- Hard to maintain and test

## Decision

We will implement a **canonical CIR (Canonical Intermediate Representation)** motion pipeline with the following architecture:

### Module Structure

```
src/shared/python/motion_pipeline/
├── sources/           # Mocap format loaders (Theia, OpenPose, etc.)
├── converters/        # Unit conversion, axis reorientation
├── cleaning/          # Gap filling, filtering, virtual markers
├── lifting/           # 2D to 3D estimation
├── retargeting/       # Source motion → character model
├── exporters/         # CIR → engine-specific formats
└── core/
    ├── motion.py      # Canonical Intermediate Representation
    ├── config.py      # Pipeline configuration
    └── pipeline.py    # Orchestrator
```

### Canonical Intermediate Representation (CIR)

The CIR is a unified data structure that all sources convert to and all exporters convert from:

```python
@dataclass
class MotionFrame:
    timestamp: float           # Seconds
    markers: dict[str, Vec3]   # Marker name → 3D position (meters)
    confidences: dict[str, float]  # Optional per-marker confidence
    root_transform: Transform  # Root position and orientation

@dataclass
class Motion:
    frames: list[MotionFrame]
    units: str = "m"           # Always meters internally
    up_axis: str = "Z"         # Always Z-up internally
    fps: float
    metadata: dict
```

### LoD-Driven Module Split

The pipeline is split into levels of detail:

| Level | Module         | Responsibility            |
| ----- | -------------- | ------------------------- |
| L1    | `sources/`     | Parse raw formats → CIR   |
| L2    | `converters/`  | Unit/axis normalization   |
| L3    | `cleaning/`    | Data quality improvements |
| L4    | `lifting/`     | 2D → 3D estimation        |
| L5    | `retargeting/` | Skeleton mapping          |
| L6    | `exporters/`   | CIR → engine format       |

### Rejected Alternatives

#### Alternative A: Per-Engine Pipelines

Each engine (MuJoCo, Drake, Pinocchio) maintains its own pipeline.

**Rejected because:**

- Duplicates 80% of conversion logic
- Inconsistent behavior across engines
- Harder to add new engines
- No shared testing surface

#### Alternative B: Single Monolith Refactor

One large `motion_pipeline.py` handling all cases.

**Rejected because:**

- Violates single responsibility principle
- Hard to test individual transformations
- Difficult to extend with new formats
- Merge conflicts likely with multiple contributors

#### Alternative C: External Library Dependency

Use existing libraries (e.g., `bvh`, `c3d`, `mediapipe`) directly.

**Rejected because:**

- No unified interface
- Inconsistent coordinate systems
- Hard to add custom processing steps
- We still need a CIR layer anyway

## Consequences

### Positive

- Single source of truth for mocap processing
- Easy to add new formats (implement one converter)
- Consistent behavior across all engines
- Testable at each transformation stage
- Clear onboarding path for contributors

### Negative

- Initial implementation effort
- Need to migrate existing engine-specific loaders
- CIR adds one extra conversion step (minor performance cost)

### Follow-ups

1. Implement CIR data structures (`core/motion.py`)
2. Build source converters for Theia, OpenPose, MediaPipe
3. Create exporter for MuJoCo (priority engine)
4. Add exporters for Drake and Pinocchio
5. Write integration tests for end-to-end pipeline
6. Document user workflow (this issue #4572)

## Validation

CI will validate this architecture via:

1. **Unit tests** for each converter module
2. **Integration tests** for full pipeline (video → engine format)
3. **Format matrix test** (#4571) auto-generates format support docs
4. **Cross-engine validation** — same input → consistent output across engines

```bash
# Run pipeline validation
pytest tests/unit/motion_pipeline/ -v
pytest tests/integration/motion_pipeline/ -v --timeout=120
```

## Related Documents

- [User Workflow Guide](../motion_pipeline/README.md)
- [Format Matrix](../motion_pipeline/formats.md)
- [Troubleshooting](../motion_pipeline/troubleshooting.md)
