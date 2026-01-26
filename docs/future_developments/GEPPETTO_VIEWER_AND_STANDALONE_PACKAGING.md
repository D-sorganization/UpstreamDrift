# Geppetto Viewer & Standalone Packaging Evaluation

**Date:** 2026-01-26
**Status:** Under Review
**Priority:** Medium

---

## Executive Summary

This document evaluates two potential improvements to the Golf Modeling Suite:

1. **Geppetto Viewer for Pinocchio** - Adding as a selectable visualization option
2. **Standalone Software Packaging** - Reducing setup requirements for end users

---

## Part 1: Geppetto Viewer for Pinocchio

### Current State

A basic Geppetto viewer implementation already exists in the codebase:

```
src/engines/physics_engines/pinocchio/python/dtack/viz/geppetto_viewer.py
```

| Aspect | Status |
|--------|--------|
| Implementation | Basic wrapper exists (~82 lines) |
| Error handling | Graceful fallback with helpful messages |
| Optional import | Uses `GEPETTO_AVAILABLE` flag |
| Pinocchio integration | Uses `GepettoVisualizer` from pinocchio.visualize |

### Why Geppetto is More Contained Than MeshCat

| Feature | MeshCat | Geppetto |
|---------|---------|----------|
| Transport | ZMQ (network socket) | CORBA (local server) |
| Rendering | Browser (WebGL) | Native desktop (OpenGL) |
| External process | Spawns browser | gepetto-gui server |
| Dependencies | `meshcat` (pip) | `gepetto-viewer` (conda-forge only) |
| Best for | Remote/cloud/notebooks | Local desktop validation |

**Geppetto advantages:**
- No browser dependency
- Native desktop rendering (faster for large models)
- Better for local development/debugging
- No network socket exposure

### Implementation Requirements

#### Issues to Address

1. **Not in `pyproject.toml` optional dependencies** - `gepetto-viewer-corba` missing from extras
2. **Incomplete `display()` method** - Currently logs but doesn't persist visualizer instance
3. **No viewer selection mechanism** - No unified way to choose between MeshCat/Geppetto at runtime

#### Estimated Effort

| Task | Effort |
|------|--------|
| Add to optional dependencies | 15 min |
| Fix `display()` method | 1 hour |
| Create `ViewerFactory` for selection | 2 hours |
| Update documentation | 1 hour |
| **Total** | **~4 hours** |

#### Files to Modify

1. `pyproject.toml` - Add `gepetto-viewer-corba` to optional deps
2. `src/engines/physics_engines/pinocchio/python/dtack/viz/geppetto_viewer.py` - Fix display method
3. `src/engines/physics_engines/pinocchio/python/dtack/viz/__init__.py` - Add ViewerFactory
4. `environment.yml` - Document conda-forge requirement

#### Proposed Changes

**pyproject.toml addition:**
```toml
[project.optional-dependencies]
engines = [
    # ... existing deps ...
    "meshcat>=0.0.18",
]
visualization = [
    "gepetto-viewer-corba>=5.0.0",  # conda-forge only
]
```

**ViewerFactory pattern:**
```python
class ViewerFactory:
    @staticmethod
    def create(viewer_type: str = "meshcat", **kwargs):
        if viewer_type == "meshcat":
            return MeshcatViewer(**kwargs)
        elif viewer_type == "geppetto":
            return GeppettoViewer(**kwargs)
        else:
            raise ValueError(f"Unknown viewer type: {viewer_type}")
```

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Geppetto not available on all platforms | Medium | Low | Document as optional, graceful fallback |
| CORBA server issues | Low | Medium | Clear error messages, fallback to MeshCat |
| Conda-only dependency | Medium | Low | Document installation path |

### Recommendation

**Proceed with implementation.** The changes are isolated, low-risk, and provide users with a desktop-native visualization option that doesn't require a browser.

---

## Part 2: Standalone Software Packaging

### Current Distribution Model

The project currently uses a developer-friendly setup:

```bash
# Current installation requires:
conda create -n golf-suite python=3.11
conda activate golf-suite
pip install golf-modeling-suite[engines]
```

**Requirements for users:**
- Python 3.11+
- Conda (for binary dependencies)
- System OpenGL drivers
- ~2GB disk space

### Challenges for "No Setup" Distribution

| Challenge | Severity | Description |
|-----------|----------|-------------|
| Python runtime | High | Users must install Python |
| Binary physics libs | High | MuJoCo, Pinocchio require specific builds |
| OpenGL/GPU drivers | Medium | Platform-specific graphics requirements |
| Large model files | Medium | URDF/mesh files add ~500MB |
| Platform-specific | High | Different builds for Windows/macOS/Linux |

### Packaging Options Comparison

#### Option 1: PyInstaller/Nuitka Bundling

**Description:** Bundle Python + all dependencies into single executable

| Pros | Cons |
|------|------|
| Single executable | Large bundle (~500MB-1GB) |
| No Python install needed | Platform-specific builds required |
| Familiar distribution model | Complex build configuration |
| | Binary deps (MuJoCo) may have issues |

**Effort:** 2-3 days basic, 1-2 weeks polished
**Best for:** Desktop application distribution

#### Option 2: Docker Container

**Description:** Complete containerized environment

| Pros | Cons |
|------|------|
| Complete isolation | Docker required on host |
| Reproducible environment | Not ideal for GUI apps |
| Easy updates | Additional complexity for users |
| Works on any Docker host | Performance overhead |

**Effort:** 1-2 days
**Best for:** Server deployments, CI/CD, headless computation

**Example Dockerfile structure:**
```dockerfile
FROM condaforge/mambaforge:latest
COPY environment.yml .
RUN mamba env create -f environment.yml
COPY . /app
WORKDIR /app
RUN pip install -e .
ENTRYPOINT ["conda", "run", "-n", "golf-suite", "golf-suite"]
```

#### Option 3: Conda Constructor (Recommended)

**Description:** Self-contained installer with bundled Python and all dependencies

| Pros | Cons |
|------|------|
| Professional installer experience | Large installer (~1-2GB) |
| Includes Python + all deps | Build per platform |
| No conda needed by user | Requires conda-constructor setup |
| Handles binary deps well | Update mechanism needed |

**Effort:** 3-5 days
**Best for:** End-user distribution, scientific software

**Example construct.yaml:**
```yaml
name: GolfModelingSuite
version: 1.0.0
channels:
  - conda-forge
  - defaults
specs:
  - python=3.11
  - numpy
  - scipy
  - mujoco
  - pin
  - meshcat-python
  - golf-modeling-suite
```

#### Option 4: Flatpak/Snap (Linux Only)

**Description:** Sandboxed Linux application packages

| Pros | Cons |
|------|------|
| Auto-updates | Linux only |
| Sandboxed | Limited platform reach |
| Handles dependencies | Complex manifest |

**Effort:** 2-3 days
**Best for:** Linux desktop users

#### Option 5: Web-Based Architecture

**Description:** Server-side computation with browser-based UI

| Pros | Cons |
|------|------|
| No local install | Requires server hosting |
| Cross-platform | Network latency |
| Already have FastAPI + MeshCat | Ongoing hosting costs |
| Easy updates | Not suitable for offline use |

**Effort:** 1-2 weeks for full implementation
**Best for:** Cloud deployment, collaborative use

### Recommended Approach

#### Short-term (1-2 days)

1. **Create Dockerfile** for containerized deployment
2. **Document one-command install script:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash
   ```

#### Medium-term (1 week)

1. **Implement conda-constructor** for platform installers
2. Build installers for:
   - Windows (.exe)
   - macOS (.pkg)
   - Linux (.sh)

#### Long-term (2-4 weeks)

1. **Split architecture:**
   - Computation backend (server/cloud)
   - Visualization frontend (web browser)
2. Deploy to cloud platform (AWS/GCP/Azure)
3. Users access via `https://golf-suite.example.com`

### Effort Summary

| Approach | Effort | User Experience | Maintenance |
|----------|--------|-----------------|-------------|
| Docker | 1-2 days | Medium | Low |
| Conda Constructor | 3-5 days | High | Medium |
| Web-based | 1-2 weeks | Highest | High |

---

## Action Items

### Phase 1: Geppetto Viewer (Priority: Medium)

- [ ] Add `gepetto-viewer-corba` to pyproject.toml optional deps
- [ ] Fix `GeppettoViewer.display()` method to persist visualizer
- [ ] Implement `ViewerFactory` for runtime viewer selection
- [ ] Add configuration option for default viewer
- [ ] Update documentation with installation instructions
- [ ] Add integration tests for Geppetto viewer

### Phase 2: Standalone Packaging (Priority: Low-Medium)

- [ ] Create Dockerfile for containerized deployment
- [ ] Write install script for one-command setup
- [ ] Evaluate conda-constructor for installer builds
- [ ] Document platform-specific requirements
- [ ] Consider web-based architecture for long-term

---

## References

- [Geppetto Viewer Documentation](https://github.com/Gepetto/gepetto-viewer)
- [Pinocchio Visualization](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_display_b-gepetto-viewer.html)
- [Conda Constructor](https://github.com/conda/constructor)
- [PyInstaller](https://pyinstaller.org/)
- Current implementation: `src/engines/physics_engines/pinocchio/python/dtack/viz/`

---

## Appendix: Current Viewer Implementations

| Viewer | File | Status |
|--------|------|--------|
| MeshCat | `dtack/viz/meshcat_viewer.py` | Fully functional |
| Geppetto | `dtack/viz/geppetto_viewer.py` | Needs fixes |
| Rob Neal Data | `dtack/viz/rob_neal_viewer.py` | Specialized for .mat files |
