# Shared Python Utilities

This directory contains reusable Python libraries and utilities shared across multiple repositories in the fleet.

## Available Packages

### `humanoid_character_builder/`

Standalone URDF humanoid model generation with video game-style character customization.

```python
from humanoid_character_builder import CharacterBuilder, BodyParameters, InertiaMode

params = BodyParameters(height_m=1.80, mass_kg=80.0, build_type="athletic")
builder = CharacterBuilder()
result = builder.build(params)
result.export_urdf("./output/my_humanoid")
```

**Features:**

- Parametric body generation from height/mass
- Mesh-based inertia calculation (trimesh)
- LOD generation for rendering/physics
- De Leva anthropometry data

### `model_generation/`

Comprehensive URDF/MJCF model building, editing, and conversion tools.

```python
from model_generation import quick_urdf, ManualBuilder, FrankensteinEditor

# Quick parametric generation
urdf = quick_urdf(height_m=1.85, preset="athletic")

# Manual construction
builder = ManualBuilder("robot")
builder.add_link(Link(name="base", inertia=Inertia.from_box(10, 1, 1, 0.5)))

# Component composition
editor = FrankensteinEditor()
editor.load_model("source.urdf")
editor.load_model("target.urdf")
editor.copy_component("left_arm", source=0, target=1)
```

**Features:**

- Parametric and manual URDF builders
- Frankenstein editor for component composition
- URDF/MJCF/SDF format conversion
- SimScape to URDF conversion
- Model library with repository integration
- REST API for headless operation

### `signal_toolkit/`

Comprehensive signal processing library for control systems, simulation, and data analysis.

```python
from signal_toolkit import Signal, SignalGenerator, FunctionFitter, apply_filter

# Generate a signal
import numpy as np
t = np.linspace(0, 10, 1000)
signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=2.0)

# Fit a function
fitter = FunctionFitter()
result = fitter.auto_fit(signal)

# Apply filtering
from signal_toolkit import create_butterworth_filter
filtered = apply_filter(signal, create_butterworth_filter('lowpass', cutoff=5, fs=100))
```

**Features:**

- 13 signal types (sine, cosine, chirp, square, etc.)
- Curve fitting (sinusoid, exponential, polynomial, custom)
- Digital filters (Butterworth, Chebyshev, Bessel, adaptive)
- Calculus (differentiation, integration, curvature)
- Noise generation (white, pink, brown, blue, violet)
- Limits (saturation, rate limiting, deadband, hysteresis)
- I/O (CSV, JSON, MAT, NPZ)

### `upstream_drift_tools/`

Process engineering calculators for chemical and industrial applications.

```python
from upstream_drift_tools.process_calculators import (
    FlareCalculator,
    ScrubberCalculator,
    FinancialCalculator,
)

flare = FlareCalculator()
design = flare.calculate_flare_size(
    total_flow=1000,
    gas_composition={"H2": 50, "CO": 30, "CH4": 20},
    temperature=500,
    pressure=1.5,
)
```

**Features:**

- Equipment sizing (flare, scrubber, baghouse)
- Thermodynamic calculators (acid gas dewpoint, WGS reactor)
- Financial analysis (NPV/IRR)
- ODE solver and optimization tools

### `notes/`

Reusable project notes workspace with file-backed persistence and reversible deletion.

```python
from notes import NotesStorage

storage = NotesStorage(project_dir="./my_project")
storage.save_text("Design notes and copied snippets")
recycled = storage.move_to_recycle(reason="cleanup")
storage.restore(recycled.item_id)
```

**Features:**

- Plain-text notes file stored with each project (`project.notes.txt`)
- Safe deletion to per-project recycle bin (`.notes_recycle_bin`)
- Restore/purge controls for reversible workflows
- Optional PyQt dock widget for embedded or pop-out usage

## Dependencies

| Package                      | Required          | Optional                   |
| ---------------------------- | ----------------- | -------------------------- |
| `humanoid_character_builder` | numpy, PyYAML     | trimesh                    |
| `model_generation`           | numpy, defusedxml | trimesh, mujoco            |
| `signal_toolkit`             | numpy, scipy      | PyQt6, matplotlib, sympy   |
| `upstream_drift_tools`       | numpy             | scipy, CoolProp, PyQt6     |
| `notes`                      | none              | PyQt6 (for dock widget UI) |

## Usage in Other Repositories

These packages can be imported directly when Tools is installed or added to PYTHONPATH:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Tools/src/shared/python"

# Or install in development mode
pip install -e /path/to/Tools
```
