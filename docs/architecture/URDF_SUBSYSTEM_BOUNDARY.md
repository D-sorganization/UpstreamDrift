# URDF Subsystem Boundary: humanoid_character_builder vs model_generation

This document clarifies the accepted architectural boundary between the
two URDF-related packages in UpstreamDrift. It reflects ADR-0020
(`Option B: Layer`) rather than the earlier "pick one stack" framing.

1. **`model_generation`** is the canonical low-level URDF / mesh /
   inertia toolkit.
2. **`humanoid_character_builder`** is the anthropometric domain layer
   that builds humanoid-specific workflows on top of that toolkit.
3. **`model_generation.humanoid`** is a compatibility facade that
   re-exports the humanoid-domain public API for callers already rooted
   in the `model_generation` namespace.

## Current layering

The boundary is about abstraction level, not "humanoid vs non-humanoid"
ownership of all code:

- `model_generation`
  - owns generic URDF XML writing, inertia primitives, conversion, and
    reusable model-building infrastructure
  - is the package new low-level URDF helpers should target
- `humanoid_character_builder`
  - owns anthropometry, body presets, humanoid segment definitions, and
    the `CharacterBuilder` user workflow
  - may compose `model_generation` but should not reintroduce duplicate
    low-level URDF or inertia implementations
- `model_generation.humanoid`
  - is a compatibility facade for callers that want the humanoid-domain
    API from inside the `model_generation` namespace
  - should stay thin; it is not the home for new anthropometric logic

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

1. **Building low-level URDF/MJCF helpers** or reusable exporter logic
2. **Need custom geometry** not based on human anthropometry
3. **Working with CAD imports** or external mesh sources
4. **Building specialized mechanisms** (grippers, legs, arms)
5. **Need Simscape Multibody export** format
6. **Creating models from scratch** with custom joint definitions
7. **Need the compatibility facade** at `model_generation.humanoid`
   rather than importing `humanoid_character_builder` directly

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

## Code-level evidence

Current source code already points in the accepted direction:

- `humanoid_character_builder.mesh.inertia_calculator` imports
  `model_generation.inertia.calculator`
- `humanoid_character_builder.mesh.primitive_inertia` imports
  `model_generation.inertia.primitives`
- `humanoid_character_builder.generators.urdf_xml_builder` imports
  `model_generation.builders.urdf_writer`
- `model_generation.humanoid.__init__` re-exports the public humanoid
  API as a compatibility facade

## Summary Table

| Concern                         | Canonical owner                 | Notes |
| ------------------------------- | ------------------------------- | ----- |
| URDF XML writer                 | `model_generation`              | Composed by `humanoid_character_builder` |
| Inertia primitives/calculators  | `model_generation`              | Reused by `humanoid_character_builder.mesh.*` |
| Anthropometry + body presets    | `humanoid_character_builder`    | Domain-specific human modeling |
| CharacterBuilder user workflow  | `humanoid_character_builder`    | Public humanoid entry point |
| Humanoid namespace compatibility| `model_generation.humanoid`     | Compatibility facade only |

## References

- [ADR-0020](../adr/0020-canonical-urdf-subsystem.md)
- [Character Builder Quickstart](../user_guide/character_builder_quickstart.md)
- [Character Presets Reference](../user_guide/character_presets.md)
- [Model Generation API](../../src/shared/python/model_generation/README.md)
