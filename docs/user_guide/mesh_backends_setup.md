# Mesh Backends Setup Guide: MakeHuman and SMPLX

This guide explains how to set up and use the MakeHuman and SMPLX mesh generation backends for the Character Builder.

## Overview

The Character Builder supports multiple mesh generation backends:

| Backend   | Status        | Description                                     |
| --------- | ------------- | ----------------------------------------------- |
| Primitive | ✅ Production | Capsule/box primitive meshes (no external deps) |
| MakeHuman | ⚠️ Alpha      | Realistic human meshes from MakeHuman           |
| SMPL-X    | ⚠️ Alpha      | Parametric body model from SMPL-X               |

## MakeHuman Backend

### What is MakeHuman?

[MakeHuman](https://www.makehumancommunity.org) is an open-source 3D human modeling application. The MakeHuman backend uses exported MakeHuman models to generate realistic humanoid meshes.

### Installation

#### Step 1: Install MakeHuman

1. Download MakeHuman from [makehumancommunity.org](https://www.makehumancommunity.org/download.html)
2. Install for your platform:
   - **Windows**: Run the installer
   - **Linux**: Extract the tarball to `/opt/makehuman` or `~/.local/share/makehuman`
   - **macOS**: Drag to Applications folder

#### Step 2: Export Base Meshes

1. Open MakeHuman
2. Use the default human model (or customize as desired)
3. Go to **File → Export → Mesh**
4. Export settings:
   - Format: **OBJ** or **FBX**
   - Include: **Body only** (no clothes, hair, etc.)
   - Units: **Meters**
5. Save to a known location (e.g., `~/makehuman_exports/base_male.obj`)

#### Step 3: Configure Character Builder

Set the `MAKEHUMAN_MESH_DIR` environment variable:

```bash
# Linux/macOS
export MAKEHUMAN_MESH_DIR="$HOME/makehuman_exports"

# Windows (PowerShell)
$env:MAKEHUMAN_MESH_DIR="$HOME/makehuman_exports"
```

Or configure in Python:

```python
from humanoid_character_builder.generators import MakeHumanConfig

config = MakeHumanConfig(mesh_directory="~/makehuman_exports")
```

### File Structure

```
makehuman_exports/
├── base_male.obj          # Male base mesh
├── base_female.obj        # Female base mesh
├── textures/              # Optional texture files
│   ├── skin_diffuse.png
│   └── skin_normal.png
└── README.txt             # Notes about exported models
```

### Usage Example

```python
from humanoid_character_builder import CharacterBuilder, BodyParameters
from humanoid_character_builder.generators import MeshGeneratorBackend

# Create builder with MakeHuman backend
builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.MAKEHUMAN)

# Build character
params = BodyParameters(height_m=1.80, mass_kg=80.0, gender_factor=1.0)
result = builder.build(params, generate_meshes=True)

# Export
result.export_urdf("./output/makehuman_character")
```

### Troubleshooting

| Issue                      | Solution                                      |
| -------------------------- | --------------------------------------------- |
| "MakeHuman mesh not found" | Verify `MAKEHUMAN_MESH_DIR` is set correctly  |
| "Invalid mesh format"      | Ensure OBJ files are exported in meters       |
| "Mesh import failed"       | Check that mesh has valid topology (no holes) |

---

## SMPL-X Backend

### What is SMPL-X?

[SMPL-X](https://smpl-x.is.tue.mpg.de) is a parametric 3D human body model that includes hands and face. It provides more realistic body shapes than primitive meshes.

### Installation

#### Step 1: Install Python Dependencies

```bash
pip install smplx trimesh torch
```

#### Step 2: Download SMPL-X Models

1. Register at [SMPL-X website](https://smpl-x.is.tue.mpg.de)
2. Download the model files:
   - `SMPLX_NEUTRAL.npz`
   - `SMPLX_MALE.npz` (optional)
   - `SMPLX_FEMALE.npz` (optional)

#### Step 3: Configure Model Directory

Place the downloaded files in a directory and set the environment variable:

```bash
# Linux/macOS
export SMPLX_MODEL_DIR="$HOME/.smplx_models"
mkdir -p "$SMPLX_MODEL_DIR"
mv ~/Downloads/SMPLX_*.npz "$SMPLX_MODEL_DIR/"

# Windows (PowerShell)
$env:SMPLX_MODEL_DIR="$HOME/.smplx_models"
New-Item -ItemType Directory -Force -Path $env:SMPLX_MODEL_DIR
Move-Item ~/Downloads/SMPLX_*.npz $env:SMPLX_MODEL_DIR/
```

Expected directory structure:

```
.smplx_models/
├── SMPLX_NEUTRAL.npz    # Required: neutral model
├── SMPLX_MALE.npz       # Optional: male-specific
├── SMPLX_FEMALE.npz     # Optional: female-specific
└── smplx_version.txt    # Version info
```

### Usage Example

```python
from humanoid_character_builder import CharacterBuilder, BodyParameters
from humanoid_character_builder.generators import MeshGeneratorBackend

# Create builder with SMPL-X backend
builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.SMPLX)

# Build character
params = BodyParameters(
    height_m=1.75,
    mass_kg=70.0,
    gender_factor=0.5,  # 0=female, 1=male, 0.5=neutral
)
result = builder.build(params, generate_meshes=True)

# Export
result.export_urdf("./output/smplx_character")
```

### SMPL-X Parameters

The SMPL-X backend supports additional parameters:

```python
from humanoid_character_builder.generators import SMPLXConfig

config = SMPLXConfig(
    model_path="/path/to/smplx_models",
    gender="neutral",  # "neutral", "male", "female"
    batch_size=1,
    use_face_contour=True,  # Include face details
    use_hands=True,         # Include hand details
)

builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.SMPLX)
# Pass config to mesh generator
```

### Troubleshooting

| Issue                   | Solution                                               |
| ----------------------- | ------------------------------------------------------ |
| "SMPLX model not found" | Verify `SMPLX_MODEL_DIR` and file names                |
| "Torch DLL load failed" | Reinstall torch: `pip install --force-reinstall torch` |
| "NPZ file corrupted"    | Re-download from SMPL-X website                        |
| "Out of memory"         | Reduce batch_size in config                            |

---

## CI Strategy for Mesh Backends

### Why Separate CI?

MakeHuman and SMPLX backends require:

- Large binary assets (mesh files, model weights)
- Optional dependencies (torch, trimesh)
- Platform-specific configuration

These are not suitable for the main CI pipeline.

### Proposed CI Workflow

Create `.github/workflows/mesh-backends-test.yml`:

```yaml
name: Mesh Backends Test

on:
  schedule:
    - cron: "0 2 * * *" # Daily at 2 AM
  workflow_dispatch: # Manual trigger

jobs:
  test-smplx:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e ".[biomechanics]"
          pip install smplx trimesh torch

      - name: Cache SMPL-X models
        uses: actions/cache@v4
        with:
          path: ~/.smplx_models
          key: smplx-models-v1

      - name: Download SMPL-X models (if not cached)
        run: |
          mkdir -p ~/.smplx_models
          # Download from authorized source

      - name: Run SMPL-X tests
        run: pytest tests/unit/tools/humanoid_character_builder/test_mesh_generators.py -v -m smplx

  test-makehuman:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[biomechanics]"

      - name: Cache MakeHuman meshes
        uses: actions/cache@v4
        with:
          path: ~/.makehuman_meshes
          key: makehuman-meshes-v1

      - name: Run MakeHuman tests
        run: pytest tests/unit/tools/humanoid_character_builder/test_mesh_generators.py -v -m makehuman
```

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.smplx
@pytest.mark.slow
def test_smplx_mesh_generation():
    """SMPL-X mesh generation test."""
    pass

@pytest.mark.makehuman
@pytest.mark.slow
def test_makehuman_mesh_generation():
    """MakeHuman mesh generation test."""
    pass

@pytest.mark.live_simulation
def test_end_to_end_mesh_pipeline():
    """Full pipeline test with real assets."""
    pass
```

### Running Tests Locally

```bash
# Run all mesh backend tests
pytest tests/unit/tools/humanoid_character_builder/test_mesh_generators.py -v

# Run only SMPL-X tests
pytest -m smplx -v

# Run only MakeHuman tests
pytest -m makehuman -v

# Run slow/end-to-end tests
pytest -m "slow or live_simulation" -v
```

---

## Related Documentation

- [Character Builder Quickstart](character_builder_quickstart.md)
- [URDF Readiness Report](../../reports/subsystem_status/URDF_READINESS.md)
- [Testing Guide](../../docs/testing/testing-guide.md)

---

**Last Updated:** 2026-05-08  
**Maintainer:** tools-team  
**Status:** Alpha documentation (under active development)
