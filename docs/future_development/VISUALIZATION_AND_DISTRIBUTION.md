# Visualization Options and Standalone Distribution Guide

**Date:** 2026-01-26
**Status:** Evaluation and Implementation Guide

This document evaluates adding Geppetto-Viewer as an optional visualization backend for Pinocchio and provides recommendations for improving standalone software distribution.

---

## Table of Contents

1. [Geppetto-Viewer Integration](#geppetto-viewer-integration)
   - [Current State](#current-state)
   - [Feasibility Assessment](#feasibility-assessment)
   - [Comparison: Meshcat vs Geppetto](#comparison-meshcat-vs-geppetto)
   - [Implementation Plan](#implementation-plan)
2. [Standalone Distribution](#standalone-distribution)
   - [Current Distribution Methods](#current-distribution-methods)
   - [Binary Dependency Challenges](#binary-dependency-challenges)
   - [Recommended Improvements](#recommended-improvements)
   - [Docker-Based Distribution](#docker-based-distribution)
3. [Implementation Checklist](#implementation-checklist)

---

## Geppetto-Viewer Integration

### Current State

The codebase already includes a basic Geppetto-Viewer wrapper:

```
src/engines/physics_engines/pinocchio/python/dtack/viz/geppetto_viewer.py
```

**Existing visualization backends:**

| Backend | Location | Status |
|---------|----------|--------|
| Meshcat | `dtack/viz/meshcat_viewer.py` | Primary, fully integrated |
| Geppetto | `dtack/viz/geppetto_viewer.py` | Exists, not exposed in GUI |
| Rob Neal | `dtack/viz/rob_neal_viewer.py` | Specialized data viewer |

The main Pinocchio GUI (`pinocchio_golf/gui.py`) currently only uses Meshcat. Geppetto-Viewer exists but is not selectable as an option.

### Feasibility Assessment

**Verdict: Doable and Contained**

| Aspect | Assessment |
|--------|------------|
| **Code Impact** | Low - isolated to Pinocchio visualization layer |
| **Architecture** | Already supports multi-backend pattern |
| **Dependencies** | Optional (`gepetto-viewer` via conda-forge) |
| **Risk** | Very low - fails gracefully if not installed |
| **Effort** | 1-2 days of development |

### Comparison: Meshcat vs Geppetto

| Feature | Meshcat (Current) | Geppetto-Viewer (Optional) |
|---------|-------------------|---------------------------|
| **Rendering** | WebGL in browser | Native OpenGL desktop |
| **Access** | Remote via HTTP URL | Local desktop only |
| **Performance** | Good for most models | Better for complex scenes |
| **Setup** | No extra process needed | Requires `gepetto-gui` server |
| **Container Support** | Excellent | Limited (needs X11) |
| **Use Case** | General visualization, remote access | Joint validation, debugging |
| **Installation** | `pip install meshcat` | `conda install -c conda-forge gepetto-viewer` |

**Recommendation:** Keep Meshcat as default, offer Geppetto as optional for users who prefer desktop visualization or need better performance for complex models.

### Implementation Plan

#### Step 1: Add Availability Flag

Add to `src/shared/python/engine_availability.py`:

```python
# Check Geppetto-Viewer
GEPETTO_AVAILABLE: bool = False
try:
    import gepetto.corbaserver
    GEPETTO_AVAILABLE = True
except ImportError:
    pass
```

Add to `_ENGINE_FLAGS` dictionary:

```python
"gepetto": GEPETTO_AVAILABLE,
"gepetto-viewer": GEPETTO_AVAILABLE,  # Alias
```

#### Step 2: Add Viewer Selection to GUI

In `pinocchio_golf/gui.py`, add viewer selection UI:

```python
# In the settings/visualization panel
viewer_combo = QComboBox()
viewer_combo.addItem("Meshcat (Browser)")
if GEPETTO_AVAILABLE:
    viewer_combo.addItem("Geppetto (Desktop)")
viewer_combo.currentTextChanged.connect(self._on_viewer_changed)
```

#### Step 3: Implement Viewer Switching

```python
def _initialize_viewer(self, viewer_type: str = "meshcat") -> None:
    """Initialize the selected visualization backend."""
    if viewer_type == "geppetto" and GEPETTO_AVAILABLE:
        from dtack.viz.geppetto_viewer import GeppettoViewer
        self.viewer = GeppettoViewer()
        self.viewer.load_model(self.model, self.visual_model)
    else:
        # Default to Meshcat
        self.viewer = viz.Visualizer(server_args=["--port", "7000"])
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(viewer=self.viewer, open=False)
        self.viz.loadViewerModel()
```

#### Step 4: Complete GeppettoViewer Implementation

The existing `geppetto_viewer.py` needs:

1. `display(q)` method to update robot configuration
2. Overlay support (frames, COMs) if desired
3. Proper error handling for CORBA connection failures

```python
def display(self, q: np.ndarray) -> None:
    """Update the robot configuration in Geppetto viewer."""
    if self.viz is not None:
        self.viz.display(q)

def add_frame(self, name: str, transform: np.ndarray) -> None:
    """Add a coordinate frame visualization."""
    # Implementation using gepetto-viewer primitives
    pass
```

---

## Standalone Distribution

### Current Distribution Methods

| Method | Status | Notes |
|--------|--------|-------|
| **Source (pip/conda)** | Working | Requires manual dependency setup |
| **Docker** | Working | Best "no setup" option currently |
| **Windows MSI** | Infrastructure exists | cx_Freeze setup, not automated |
| **PyPI** | Not published | Would help Python developers |
| **Conda-forge** | Not available | Would be ideal for scientific users |
| **macOS App** | Not implemented | Potential future option |

### Binary Dependency Challenges

The project depends on libraries requiring native binary components:

| Dependency | Challenge |
|------------|-----------|
| **MuJoCo** | Native C physics engine |
| **Drake** | Complex C++ robotics toolkit |
| **Pinocchio** | C++ rigid body dynamics |
| **PyQt6** | Platform-specific Qt bindings |
| **OpenGL** | System graphics drivers |

These make "download and double-click" distribution challenging across platforms.

### Recommended Improvements

#### Priority 1: Docker as Primary Standalone Option (Low Effort)

Docker is currently the best path to "it just works" distribution.

**Create `docker-compose.yml` for easy startup:**

```yaml
version: '3.8'
services:
  golf-suite:
    image: ghcr.io/d-sorganization/golf-modeling-suite:latest
    ports:
      - "7000:7000"  # Meshcat visualization
      - "8080:8080"  # Optional web UI
    volumes:
      - ./data:/app/data  # User data persistence
    environment:
      - DISPLAY=${DISPLAY:-:0}
    # For GUI support on Linux:
    # volumes:
    #   - /tmp/.X11-unix:/tmp/.X11-unix
```

**User experience:**

```bash
# One-command startup
docker-compose up

# Open browser to http://localhost:7000 for visualization
```

**CI/CD additions needed:**

```yaml
# In .github/workflows/release.yml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: ghcr.io/d-sorganization/golf-modeling-suite:${{ github.ref_name }}
```

#### Priority 2: Automate Windows MSI (Medium Effort)

The cx_Freeze infrastructure exists at `installer/windows/setup.py`.

**Add to CI/CD:**

```yaml
windows-installer:
  runs-on: windows-latest
  steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install cx_Freeze
        pip install -e .[dev]
    - name: Build MSI
      run: python installer/windows/setup.py bdist_msi
    - name: Upload MSI
      uses: actions/upload-artifact@v4
      with:
        name: windows-installer
        path: dist/*.msi
```

#### Priority 3: Publish to PyPI (Low Effort)

For users who can manage binary dependencies themselves:

```yaml
# In .github/workflows/release.yml
publish-pypi:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
    - name: Build package
      run: |
        pip install build
        python -m build
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

**User experience:**

```bash
pip install golf-modeling-suite
```

Note: Users still need to install MuJoCo, Pinocchio, etc. separately.

#### Priority 4: Conda-forge Package (High Effort)

This would be the ideal solution as conda handles binary dependencies.

**Benefits:**
- Single command install with all dependencies
- Cross-platform binary distribution
- Scientific community standard

**Requirements:**
- Create `meta.yaml` recipe
- Submit to conda-forge staged-recipes
- Maintain recipe for updates

**Estimated effort:** 2-4 weeks

**User experience (once published):**

```bash
conda install -c conda-forge golf-modeling-suite
```

#### Priority 5: Web Demo Mode (Optional)

For zero-install demos:
- Host a read-only demo instance
- Users access via browser (Meshcat)
- Consider Streamlit/Gradio for simple UI

### Docker-Based Distribution

#### Recommended Dockerfile Structure

```dockerfile
# Multi-stage build for smaller image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Final image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application
COPY . /app
WORKDIR /app

# Expose Meshcat port
EXPOSE 7000

# Default command
CMD ["python", "-m", "src.launcher.main"]
```

#### Docker Usage Documentation

Add to README.md:

```markdown
## Quick Start with Docker

The easiest way to run Golf Modeling Suite without any setup:

```bash
# Pull the latest image
docker pull ghcr.io/d-sorganization/golf-modeling-suite:latest

# Run with Meshcat visualization
docker run -p 7000:7000 ghcr.io/d-sorganization/golf-modeling-suite:latest

# Open http://localhost:7000 in your browser
```

For persistent data:

```bash
docker run -p 7000:7000 -v $(pwd)/data:/app/data golf-modeling-suite:latest
```
```

---

## Implementation Checklist

### Geppetto-Viewer Integration

- [ ] Add `GEPETTO_AVAILABLE` flag to `engine_availability.py`
- [ ] Add "gepetto" to `_ENGINE_FLAGS` dictionary
- [ ] Add viewer selection dropdown to Pinocchio GUI
- [ ] Implement `_on_viewer_changed()` handler
- [ ] Complete `GeppettoViewer.display()` method
- [ ] Add error handling for CORBA connection failures
- [ ] Update GUI to gracefully handle viewer unavailability
- [ ] Add tests for viewer selection logic
- [ ] Document Geppetto installation in README

### Standalone Distribution

- [ ] Create `docker-compose.yml` for easy startup
- [ ] Add Docker image publishing to CI/CD
- [ ] Document Docker usage in README
- [ ] Add PyPI publishing workflow
- [ ] Test Windows MSI build in CI/CD
- [ ] Create conda-forge recipe (future)

---

## Related Files

| File | Purpose |
|------|---------|
| `src/shared/python/engine_availability.py` | Availability flags for optional dependencies |
| `src/engines/physics_engines/pinocchio/python/dtack/viz/geppetto_viewer.py` | Geppetto viewer wrapper |
| `src/engines/physics_engines/pinocchio/python/dtack/viz/meshcat_viewer.py` | Meshcat viewer wrapper |
| `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py` | Main Pinocchio GUI |
| `installer/windows/setup.py` | Windows MSI installer configuration |
| `Dockerfile` | Docker image definition |

---

## References

- [Meshcat Documentation](https://github.com/meshcat-dev/meshcat-python)
- [Gepetto-Viewer Documentation](https://github.com/Gepetto/gepetto-viewer)
- [Pinocchio Visualization Tutorial](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_display_b-gepetto-viewer.html)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Conda-forge Contributing](https://conda-forge.org/docs/maintainer/adding_pkgs.html)
