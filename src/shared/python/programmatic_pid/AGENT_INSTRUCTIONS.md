# Programmatic-PID: Agent Instructions

**Purpose:** Step-by-step guide for an IDE agent (Cursor, Claude Code, Copilot, etc.) to work with the Programmatic-PID library — from creating YAML specs to generating P&ID drawings.

---

## 1. Project Overview

Programmatic-PID converts YAML specifications into P&ID (Piping & Instrumentation Diagram) drawings in DXF and SVG format. It is designed for agentic use: an LLM generates or edits the YAML, the engine validates and renders it, and the user reviews the output.

### Architecture

```
programmatic_pid/
  __init__.py          # Public API exports
  types.py             # Core types: BBox, Point, ValidationIssue, SpecValidationError
  geometry.py          # Coordinate math, bounding boxes, collision detection
  profiles.py          # Layout profiles: review, presentation, compact
  spec_loader.py       # YAML loading, SpecAccessor, config accessors
  validation.py        # Schema + referential integrity validation
  equipment.py         # Equipment registry + ISA 5.1 symbol renderers
  rendering.py         # Low-level DXF primitives: text, box, arrow, layer, SVG export
  layout.py            # Canvas layout, LabelPlacer, instrument spreading
  streams.py           # Pipe/flow path rendering
  instruments.py       # Instrument bubble rendering
  controls.py          # Control loop signal line routing
  title_block.py       # Title block, notes panels, mass balance
  document.py          # PIDDocument facade (primary high-level API)
  cli.py               # Sheet generation orchestration + CLI entry point
  generator.py         # Backward-compat shim (imports from all modules above)
```

### Key Principles

- **TDD:** Every module has matching tests in `tests/`. Run `pytest` after every change.
- **DbC:** Public functions document preconditions and postconditions. Use `assert` for preconditions in debug mode.
- **DRY:** Shared logic lives in `geometry.py`, `types.py`, and `rendering.py`.
- **Backward compatibility:** `generator.py` re-exports everything, so old imports still work.

---

## 2. Getting Started

### Install

```bash
cd Programmatic-PID
pip install -e ".[dev]" --break-system-packages
# Or if in the Tools monorepo:
pip install -e ".[pid,dev]" --break-system-packages
```

### Quick Test

```bash
python -m pytest tests/ -q
```

### Generate a P&ID

```bash
python -m programmatic_pid.cli \
  --spec examples/biochar/biochar_pid_spec.yml \
  --out output.dxf \
  --svg output.svg \
  --profile presentation
```

### Programmatic Usage

```python
from programmatic_pid import PIDDocument
from pathlib import Path

doc = PIDDocument.from_yaml("spec.yml", profile="compact")
doc.export_dxf(Path("output.dxf"))
doc.export_svg(Path("output.svg"))

# Structured validation (agent-friendly)
issues = doc.validate_json()
# Returns: [{"path": "...", "message": "...", "severity": "error"}]

# Spatial queries
bbox = doc.equipment_bbox("V-101")  # -> BBox(x_min, y_min, x_max, y_max)
pos = doc.equipment_position("V-101")  # -> Point(x, y)
free = doc.find_free_region(20, 15)  # -> BBox for new equipment
```

---

## 3. YAML Spec Structure

A P&ID spec is a YAML file with these top-level keys:

```yaml
project:
  title: "My Process"
  document_number: "PID-001"
  revision: "A"
  company: "ACME Corp"
  author: "Agent"
  date: "2026-03-10"

drawing:
  paper:
    width: 420
    height: 297
    units: mm

equipment:
  - id: V-101
    type: vessel          # vessel, hopper, fan, pump, tank, heat_exchanger, etc.
    service: "Feed Hopper"
    x: 20
    y: 80
    width: 15
    height: 25
    notes:
      - "Capacity: 10 m³"

instruments:
  - id: TI-101
    tag: TI-101
    service: "Reactor inlet temperature"
    x: 55
    y: 90

streams:
  - id: S-1
    from:
      equipment: V-101
      side: right
    to:
      equipment: R-201
      side: left
    label: "Raw feed"
    # OR: label: {text: "Raw feed", x: 45, y: 85}
    # OR: vertices: [[20, 80], [40, 80], [60, 80]]

control_loops:
  - id: TIC-101
    measurement: TI-101
    final_element: TV-101
    objective: "Maintain reactor temperature at 600°C"

interlocks:
  - id: ESD-1
    trigger: "TI-101 > 750°C"
    action: "Close FV-101, open BV-102"
```

### Equipment Types

The engine has built-in renderers for these types (case-insensitive):

| Type | Description |
|------|-------------|
| `vessel`, `box`, `dryer`, `combustor`, `auger` | Rectangular outline |
| `hopper` | Trapezoidal with discharge |
| `fan` | Circle with blade arcs |
| `rotary_valve` | Diamond shape |
| `burner` | Flame-shaped symbol |
| `bin` | Rectangular with angled bottom |
| `gate_valve` | ISA 5.1 gate valve |
| `globe_valve` | ISA 5.1 globe valve |
| `ball_valve` | ISA 5.1 ball valve |
| `check_valve` | ISA 5.1 check valve |
| `control_valve` | ISA 5.1 control valve |
| `relief_valve`, `psv` | ISA 5.1 relief valve |
| `rupture_disk` | ISA 5.1 rupture disk |
| `heat_exchanger` | Shell-and-tube symbol |
| `pump` | Circle with discharge triangle |
| `tank` | Cylindrical tank outline |

Unknown types fall back to a labeled rectangle.

### Stream Connection Modes

Streams can be defined three ways:

1. **Equipment references:** `from: {equipment: V-101, side: right}` / `to: {equipment: R-201, side: left}`
2. **Explicit coordinates:** `start: [x, y]` / `end: [x, y]`
3. **Vertex list:** `vertices: [[x1,y1], [x2,y2], ...]` (polyline with arrowhead)

### Layout Profiles

Three built-in profiles adjust visual density:

| Profile | Use case |
|---------|----------|
| `presentation` | Large text, wide spacing — for slides/meetings |
| `review` | Medium text — for engineering review |
| `compact` | Small text, tight spacing — for detail drawings |

---

## 4. Agent Workflow: Concept to Drawing

Follow this workflow when creating a P&ID from a user's process description:

### Step 1: Create the YAML Spec

Start with equipment, then add streams, instruments, and control loops. Place equipment on a grid with reasonable spacing (20-30 units between items).

```python
import yaml

spec = {
    "project": {"title": "User's Process", "document_number": "PID-001"},
    "equipment": [
        {"id": "V-101", "type": "hopper", "service": "Feed Bin",
         "x": 10, "y": 80, "width": 15, "height": 20},
        {"id": "R-201", "type": "vessel", "service": "Reactor",
         "x": 50, "y": 70, "width": 20, "height": 35},
    ],
    "streams": [
        {"id": "S-1", "from": {"equipment": "V-101", "side": "right"},
         "to": {"equipment": "R-201", "side": "left"}, "label": "Feed"},
    ],
    "instruments": [],
    "control_loops": [],
}

with open("spec.yml", "w") as f:
    yaml.dump(spec, f, default_flow_style=False)
```

### Step 2: Validate

```python
from programmatic_pid import validate_spec_json
issues = validate_spec_json(spec)
if issues:
    for issue in issues:
        print(f"  [{issue['severity']}] {issue['path']}: {issue['message']}")
    # Fix the issues and re-validate
```

### Step 3: Render Preview

```python
from programmatic_pid import PIDDocument
from pathlib import Path

doc = PIDDocument(spec, profile="review")
doc.export_dxf(Path("preview.dxf"))
doc.export_svg(Path("preview.svg"))
```

### Step 4: Iterate

Use spatial queries to make smart placement decisions:

```python
# Where is existing equipment?
bbox = doc.equipment_bbox("R-201")  # BBox(x_min, y_min, x_max, y_max)

# Find free space for new equipment
free = doc.find_free_region(15, 20)  # width=15, height=20
# Returns BBox where you can place the next piece of equipment
```

### Step 5: Add Detail

Add instruments, control loops, and interlocks incrementally. Re-validate after each addition.

### Step 6: Final Export

```python
from programmatic_pid.cli import generate
generate("spec.yml", "final.dxf", svg_path="final.svg", profile="presentation")
```

---

## 5. Extending Equipment Types

Register custom equipment symbols without modifying core code:

```python
from programmatic_pid import register_equipment

@register_equipment("my_reactor")
def render_my_reactor(msp, x, y, w, h, layer):
    """Draw a custom reactor symbol."""
    # Use ezdxf modelspace primitives
    msp.add_circle((x + w/2, y + h/2), radius=min(w, h)/2,
                    dxfattribs={"layer": layer})
    # Add internal details...
```

After registration, use `type: my_reactor` in the YAML spec.

---

## 6. Module Reference (for developers)

### Adding a New Module

1. Create `src/programmatic_pid/new_module.py` with docstring, preconditions, postconditions
2. Create `tests/test_new_module.py` with matching tests
3. Add exports to `__init__.py`
4. Add backward-compat re-export to `generator.py`
5. Run `pytest` to verify

### Running Tests

```bash
# All tests
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_equipment.py -v

# With coverage
python -m pytest tests/ --cov=programmatic_pid --cov-report=term-missing
```

### Code Style

- Line length: 88 (black/ruff default)
- Type hints on all public functions
- Docstrings with Preconditions/Postconditions sections
- `logging` module (not `print()`) for diagnostics

---

## 7. Remaining Work (for future agents)

These items are documented in the migration plan but not yet implemented:

### High Priority
- **Line numbering / pipe specs:** Add `line_spec` field to streams (e.g., `4"-CS-150#-S-001`)
- **Nozzle schedule support:** Explicit equipment nozzle positions with sizes and ratings
- **Reducer/expander symbols:** Visual pipe size changes

### Medium Priority
- **Auto-routing with obstacle avoidance:** Manhattan routing that avoids crossing equipment
- **Template library:** Pre-built specs for common process units
- **Spec diffing / change tracking:** Redline/clouding for spec version changes
- **Natural-language spec generation:** LLM wrapper to convert English to YAML fragments

### Lower Priority
- **Legend / symbol key generation**
- **Bill of Materials extraction** (CSV/Excel from spec)
- **Revision tracking in drawing**
- **Multi-drawing projects** with off-sheet connectors
- **Units-aware fields** (SI vs Imperial)
- **Visual regression tests** (golden image comparison)
- **Property-based testing** with hypothesis
- **Bidirectional DXF round-tripping**

---

## 8. Tools Monorepo Location

After migration, the library lives at:

```
Tools/
  src/shared/python/programmatic_pid/   # Library (all modules)
  src/pid_generator/launch_cli.py       # CLI launcher
  tests/programmatic_pid/               # All tests
  schema/pid_spec.schema.json           # JSON Schema for spec validation
  examples/pid/biochar/                 # Example specs
```

Install with: `pip install -e ".[pid]"`
