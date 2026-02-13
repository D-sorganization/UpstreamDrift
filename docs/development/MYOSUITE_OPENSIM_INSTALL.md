# MyoSuite & OpenSim Installation

## Changes

### Docker Environment
Added MyoSuite and OpenSim to the Docker build process:

**Dockerfile**:
```dockerfile
RUN pip install --no-cache-dir \
    ...
    myosuite \
    opensim \
    ...
```

**requirements.txt**:
```
myosuite>=2.0.0
opensim>=4.4.0
```

## Installation Details

### MyoSuite
- **Package**: `myosuite`
- **Purpose**: Musculoskeletal simulation with 290-muscle models
- **Features**: Muscle activation, tendon dynamics, realistic biomechanics
- **Use Case**: Golf swing muscle activation analysis

### OpenSim
- **Package**: `opensim`
- **Purpose**: Biomechanical modeling and simulation
- **Features**: Kinematic analysis, inverse dynamics, muscle force estimation
- **Use Case**: Joint angles, moments, and muscle forces during golf swing

## Testing

After rebuilding the Docker image, verify installations:

```bash
# Rebuild Docker image
docker-compose build

# Start container
docker-compose up -d

# Verify installations
docker exec golf-backend python -c "import myosuite; print(f'MyoSuite: {myosuite.__version__}')"
docker exec golf-backend python -c "import opensim; print(f'OpenSim: {opensim.__version__}')"
```

## API Integration

Both engines are already integrated in the engine loader system:
- `EngineType.MYOSIM` - MyoSuite engine
- `EngineType.OPENSIM` - OpenSim engine

Engine probes will automatically detect these installations once the Docker image is rebuilt.

## Closes

- Fixes #1141 (Install MyoSuite)
- Fixes #1140 (Install OpenSim)
