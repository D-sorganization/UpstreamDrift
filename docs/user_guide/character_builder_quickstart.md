# Character Builder Quickstart

Generate your first humanoid URDF in under 5 minutes using the Character Builder.

## Overview

The Character Builder allows you to create parametric humanoid characters and export them as URDF files for use in physics simulators like MuJoCo, Drake, and Pinocchio.

## Installation

### Python Dependencies

```bash
# Install the core package
pip install upstream-drift

# For full functionality (optional)
pip install upstream-drift[full]
```

### Optional: SMPL-X Support

For advanced mesh generation from body scans:

```bash
pip install smplx
```

### Optional: MakeHuman Support

For detailed character creation from MakeHuman models:

1. Download MakeHuman from [makehumancommunity.org](https://www.makehumancommunity.org)
2. Export your character as FBX or OBJ
3. Use the converter tools to import into Character Builder

## 5-Line Minimal Example

The fastest way to generate a URDF:

```python
from humanoid_character_builder import quick_urdf

# Generate URDF with default parameters
urdf_xml = quick_urdf(height_m=1.75, mass_kg=75.0)

# Save to file
with open("my_humanoid.urdf", "w") as f:
    f.write(urdf_xml)

print("URDF generated successfully!")
```

## Adjusting Body Parameters

### Basic Parameters

```python
from humanoid_character_builder import BodyParameters, CharacterBuilder

# Create custom body parameters
params = BodyParameters(
    height_m=1.80,           # Height in meters
    mass_kg=80.0,            # Mass in kilograms
    muscularity=0.7,         # 0.0 (slim) to 1.0 (muscular)
    gender_factor=0.5,       # 0.0 (female) to 1.0 (male)
)

# Build character
builder = CharacterBuilder()
result = builder.build(params)

# Export
result.export_urdf("./output/my_character")
```

### Body Segment Customization

```python
from humanoid_character_builder import BodyParameters

params = BodyParameters(
    height_m=1.75,
    mass_kg=70.0,
    # Override specific body segments
    segment_mass_ratios={
        "thigh": 0.15,  # Override thigh mass ratio
        "torso": 0.40,  # Override torso mass ratio
    },
)
```

### Using Presets

```python
from humanoid_character_builder import CharacterBuilder

builder = CharacterBuilder()

# List available presets
print("Available presets:", builder.list_presets())

# Create from preset
params = builder.create_from_preset(
    preset_name="athletic",
    height_m=1.85,  # Override preset defaults
    mass_kg=85.0,
)

# Generate URDF
urdf_xml = builder.generate_urdf(params)
```

## Loading URDF in Simulators

### MuJoCo

```python
import mujoco
import mujoco.viewer

# Load URDF
model = mujoco.MjModel.from_xml_path("my_humanoid.urdf")
data = mujoco.MjData(model)

# Run simulation
for _ in range(1000):
    mujoco.mj_step(model, data)

# Visualize
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Drake

```python
from pydrake.all import (
    DiagramBuilder,
    MeshcatVisualizer,
    MultiposePlant,
    Parser,
    Simulator,
)

builder = DiagramBuilder()

# Add robot model
plant = builder.AddSystem(MultiposePlant(num_positions=0))
parser = Parser(plant)
parser.AddModelFromFile("my_humanoid.urdf")
plant.Finalize()

# Add visualizer
visualizer = MeshcatVisualizer.AddToBuilder(builder)

# Simulate
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()
simulator.AdvanceTo(10.0)
```

### Pinocchio

```python
import pinocchio as pin

# Load URDF model
model = pin.buildModelFromUrdf("my_humanoid.urdf")
data = model.createData()

# Forward kinematics
q = pin.neutral(model)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# Print end-effector position
print("Hand position:", data.oMf[model.getFrameId("hand_r")].translation)
```

## Where to Go Next

### Frankenstein Editor

Combine body parts from different sources:

```python
from humanoid_character_builder.frankenstein import FrankensteinEditor

editor = FrankensteinEditor()
editor.add_source("torso", "athlete_torso.urdf")
editor.add_source("legs", "dancer_legs.urdf")
merged = editor.merge()
merged.export_urdf("./merged_character")
```

### Simscape Converter

Convert URDF to Simscape Multibody:

```python
from humanoid_character_builder.converters import urdf_to_simscape

urdf_to_simscape(
    "my_humanoid.urdf",
    output_dir="./simscape_model",
)
```

### GUI Editor

Launch the interactive character editor:

```bash
python -m humanoid_character_builder.gui
```

Or programmatically:

```python
from humanoid_character_builder.gui import CharacterEditorGUI

editor = CharacterEditorGUI()
editor.show()
```

## Troubleshooting

### URDF Validation Errors

```python
from humanoid_character_builder import CharacterBuilder
from humanoid_character_builder.validation import validate_urdf

builder = CharacterBuilder()
params = BodyParameters(height_m=1.75, mass_kg=75.0)
urdf_xml = builder.generate_urdf(params)

# Validate
errors = validate_urdf(urdf_xml)
if errors:
    print("Validation errors:", errors)
```

### Mesh Generation Issues

```python
from humanoid_character_builder import CharacterBuilder, BodyParameters
from humanoid_character_builder.generators import MeshGeneratorBackend

# Use primitive backend for faster generation
builder = CharacterBuilder(mesh_backend=MeshGeneratorBackend.PRIMITIVE)
params = BodyParameters(height_m=1.75)
result = builder.build(params, generate_meshes=True)
```

## API Reference

### quick_urdf()

```python
def quick_urdf(
    height_m: float = 1.75,
    mass_kg: float = 75.0,
    preset: str | None = None,
) -> str:
    """Generate URDF XML string with minimal configuration."""
```

### BodyParameters

```python
class BodyParameters:
    def __init__(
        self,
        height_m: float = 1.75,
        mass_kg: float = 75.0,
        muscularity: float = 0.5,
        gender_factor: float = 0.5,
        segment_mass_ratios: dict[str, float] | None = None,
    ):
        """Define body parameters for character generation."""
```

### CharacterBuilder

```python
class CharacterBuilder:
    def generate_urdf(self, params: BodyParameters) -> str:
        """Generate URDF XML from body parameters."""

    def build(self, params: BodyParameters) -> CharacterBuildResult:
        """Build complete character with meshes and URDF."""

    @staticmethod
    def list_presets() -> list[str]:
        """List available body presets."""

    @staticmethod
    def list_segments() -> list[str]:
        """List all available body segments."""
```
