# Quick Start Guide

## Installation

1. **Clone and setup environment**:

   ```bash
   git clone <repo-url>
   cd Pinocchio_Golf_Model
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scriptsctivate
   pip install -r python/requirements.txt
   ```

2. **Install backend dependencies**:
   ```bash
   pip install pin mujoco meshcat
   # Optional: conda install -c conda-forge gepetto-viewer
   ```

## Basic Usage

### 1. Load Canonical Model

```python
from dtack.utils.urdf_exporter import URDFExporter
from dtack.backends import BackendFactory, BackendType

# Export URDF from canonical YAML
exporter = URDFExporter("models/spec/golfer_canonical.yaml")
exporter.export("models/generated/golfer.urdf")

# Load in Pinocchio
backend = BackendFactory.create(BackendType.PINOCCHIO, "models/generated/golfer.urdf")
```

### 2. Visualize Model

```python
from dtack.viz import MeshCatViewer

viewer = MeshCatViewer()
viewer.load_model(backend.model, backend.visual_model)
q = backend.model.neutral()
viewer.display(q)
```

### 3. Compute Dynamics

```python
import numpy as np

q = np.zeros(backend.model.nq)
v = np.zeros(backend.model.nv)
a = np.zeros(backend.model.nv)

# Inverse dynamics
tau = backend.compute_inverse_dynamics(q, v, a)

# Forward dynamics
a_computed = backend.compute_forward_dynamics(q, v, tau)

# Mass matrix
M = backend.compute_mass_matrix(q)
```

## Running the GUI

```bash
python -m dtack.gui.main_window
```

## Next Steps

- See [Pinocchio Project Outline](../Pinocchio_Project_Outline.md) for full roadmap
- Check [API Documentation](../api/) for detailed API reference
- Explore examples in `python/examples/`
