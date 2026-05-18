# URDF Subsystem Boundary: humanoid_character_builder vs model_generation

This document clarifies the architectural boundary between two URDF generation subsystems in the UpstreamDrift repository:

1. **`humanoid_character_builder`** - Character-focused URDF generation
2. **`model_generation`** - General-purpose model generation

## Decision Tree

Use this flowchart to determine which subsystem to use:

```
                                    ┌─────────────────────────────────────┐
                                    │  What type of model do you need?    │
                                    └─────────────────┬───────────────────┘
                                                      │
                    ┌─────────────────────────────────┴──────────────────────────────────┐
                    │                                                                    │
                    ▼                                                                    ▼
        ┌───────────────────────┐                                          ┌───────────────────────┐
        │  Humanoid character   │                                          │  Non-humanoid or      │
        │  (biped, anthropo-    │                                          │  specialized model    │
        │   metric proportions) │                                          │  (quadruped, robot,   │
        │                       │                                          │   custom geometry)    │
        └───────────┬───────────┘                                          └───────────┬───────────┘
                    │                                                                    │
                    ▼                                                                    ▼
        ┌───────────────────────┐                                          ┌───────────────────────┐
        │  Need SMPL-X or       │                                          │  Use model_generation │
        │  MakeHuman mesh       │                                          │  with custom URDF     │
        │  generation?          │                                          │  templates            │
        └───────────┬───────────┘                                          └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        │          Yes          │          No
        │                       │
        ▼                       ▼
┌───────────────────┐   ┌───────────────────┐
│ Use SMPLXMesh     │   │ Use               │
│ Generator or      │   │ PrimitiveMesh     │
│ MakeHumanMesh     │   │ Generator         │
│ Generator         │   │                   │
└─────────┬─────────┘   └─────────┬─────────┘
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │ humanoid_character_   │
          │ builder               │
          └───────────────────────┘
```

## When to Use humanoid_character_builder

Use `humanoid_character_builder` when:

1. **Building human-like characters** with anthropometric proportions
2. **Need preset body types** (athletic, average, child, senior, etc.)
3. **Require SMPL-X or MakeHuman mesh generation** for realistic body shapes
4. **Working with motion capture data** that expects standard humanoid segments
5. **Need gender-specific models** with appropriate anatomical differences
6. **Building characters for biomechanics simulation** with validated segment masses

### Example Use Cases

```python
from humanoid_character_builder import CharacterBuilder, BodyParameters

# Quick preset-based character
builder = CharacterBuilder()
params = builder.create_from_preset("pro_golfer_male")
result = builder.build(params)

# Custom anthropometric character
params = BodyParameters(
    height_m=1.75,
    mass_kg=70.0,
    build_type=BuildType.AVERAGE,
    gender_model=GenderModel.FEMALE,
    muscularity=0.5,
)
result = builder.build(params)
result.export_urdf("./output/my_character")
```

### Available Presets

| Category        | Presets                                                              |
| --------------- | -------------------------------------------------------------------- |
| Basic types     | `athletic`, `average`, `heavy`, `lean`, `compact`, `tall`, `minimal` |
| Gender-specific | `male_average`, `female_average`, `tall_male`, `petite_female`       |
| Age-specific    | `child_8yo`, `senior_70yo`                                           |
| Sport-specific  | `golfer_pro`, `pro_golfer_male`, `pro_golfer_female`                 |

## When to Use model_generation

Use `model_generation` when:

1. **Building non-humanoid models** (quadrupeds, robots, manipulators)
2. **Need custom geometry** not based on human anthropometry
3. **Working with CAD imports** or external mesh sources
4. **Building specialized mechanisms** (grippers, legs, arms)
5. **Need Simscape Multibody export** format
6. **Creating models from scratch** with custom joint definitions

### Example Use Cases

```python
from model_generation import ModelBuilder

# Custom robot model
builder = ModelBuilder()
builder.add_link("base", geometry=BoxGeometry(0.5, 0.3, 0.2))
builder.add_joint("joint1", parent="base", child="link1", joint_type="revolute")
model = builder.build()
model.export_urdf("./output/my_robot.urdf")
model.export_simscape("./output/my_robot.smdb")
```

## Module Dependencies

```
humanoid_character_builder
├── core/
│   ├── body_parameters.py      # BodyParameters dataclass
│   ├── anthropometry.py        # Mass/length estimation from CDC/NHANES data
│   └── segment_definitions.py  # HUMANOID_SEGMENTS constant
├── generators/
│   ├── urdf_generator.py       # URDF generation for humanoids
│   ├── mesh_generator.py       # Mesh generation backends
│   ├── _mesh_smplx.py          # SMPL-X body model integration
│   └── _mesh_makehuman.py      # MakeHuman integration
├── presets/
│   └── loader.py               # Body type presets with citations
└── interfaces/
    └── api.py                  # CharacterBuilder public API

model_generation
├── builders/
│   ├── model_builder.py        # General model construction
│   └── link_builder.py         # Link/joint definitions
├── exporters/
│   ├── urdf_exporter.py        # URDF export (general purpose)
│   └── simscape_exporter.py    # Simscape Multibody export
├── converters/
│   └── urdf_to_simscape.py     # Format conversion
└── library/
    └── model_library.py        # Pre-built model repository
```

## Shared Dependencies

Both subsystems share:

- `BodyParameters` - Defined in `humanoid_character_builder.core.body_parameters`
- Standard URDF schema (ROS/URDF specification)
- Mesh processing utilities (trimesh, numpy)

## Migration Guide

### From model_generation to humanoid_character_builder

If you're building humanoids and currently using `model_generation`:

```python
# Old (model_generation)
from model_generation import create_humanoid
model = create_humanoid(height=1.75, mass=75.0)

# New (humanoid_character_builder)
from humanoid_character_builder import CharacterBuilder, BodyParameters
builder = CharacterBuilder()
params = BodyParameters(height_m=1.75, mass_kg=75.0)
result = builder.build(params)
```

### From humanoid_character_builder to model_generation

If you need Simscape export for a humanoid:

```python
# humanoid_character_builder doesn't directly support Simscape export
# Use the URDF output and convert:

from model_generation.converters import urdf_to_simscape
urdf_to_simscape("my_humanoid.urdf", "my_humanoid.smdb")
```

## Summary Table

| Feature                | humanoid_character_builder | model_generation |
| ---------------------- | -------------------------- | ---------------- |
| Humanoid presets       | ✅ Yes (16 presets)        | ❌ No            |
| SMPL-X integration     | ✅ Yes                     | ❌ No            |
| MakeHuman integration  | ✅ Yes                     | ❌ No            |
| Anthropometric scaling | ✅ CDC/NHANES based        | ❌ Manual        |
| Gender-specific models | ✅ Yes                     | ❌ Manual        |
| Simscape export        | ❌ Via converter           | ✅ Native        |
| Custom robots          | ❌ No                      | ✅ Yes           |
| Quadrupeds             | ❌ No                      | ✅ Yes           |
| CAD import             | ❌ No                      | ✅ Yes           |

## References

- [Character Builder Quickstart](../user_guide/character_builder_quickstart.md)
- [Character Presets Reference](../user_guide/character_presets.md)
- [Model Generation API](../../src/shared/python/model_generation/README.md)
