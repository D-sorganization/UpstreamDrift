# URDF Generator Bundled Assets

This directory contains mesh assets bundled with the repository.

## Purpose

Mesh files are committed directly to the repository to ensure:

1. **No runtime downloads** - Models work offline out of the box
2. **Version stability** - Meshes match the URDF files exactly
3. **Reproducibility** - Same assets every time

## Directory Structure

```
bundled_assets/
├── human_models/
│   └── human_subject_with_meshes/
│       ├── model.urdf          # The URDF file
│       ├── meshes/             # All STL/DAE mesh files
│       └── metadata.json       # License and source info
└── golf_equipment/
    ├── driver/
    ├── iron_5/
    └── putter/
```

## Asset Sources

### Human Models

- **Source**: [human-gazebo](https://github.com/gbionics/human-gazebo)
- **Commit**: 39cfb24fd1e16cdaa24d06b55bd16850f1825fae
- **License**: CC-BY-SA 2.0

### Golf Equipment

- **Source**: Generated procedurally from specifications
- **License**: MIT (same as repository)

## Adding New Assets

1. Download the model files to this directory
2. Update the URDF paths to use relative paths
3. Add metadata.json with license information
4. Commit to repository

## Usage

```python
from tools.urdf_generator.bundled_assets import get_bundled_model_path

# Get path to bundled human model
model_path = get_bundled_model_path("human_subject_with_meshes")
```
