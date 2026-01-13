# Golf Modeling Suite - Troubleshooting Guide

This guide covers common issues and solutions for the Golf Modeling Suite.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Engine Issues](#engine-issues)
3. [GUI/Launcher Issues](#guilauncher-issues)
4. [Performance Issues](#performance-issues)
5. [Test Failures](#test-failures)
6. [CI/CD Issues](#cicd-issues)

---

## Installation Issues

### Python Version Mismatch

**Symptom:** `SyntaxError` or `ModuleNotFoundError` on import

**Solution:**

```bash
# Verify Python version (requires 3.11+)
python --version

# If using pyenv:
pyenv install 3.11
pyenv local 3.11
```

### Missing System Dependencies (Linux)

**Symptom:** Qt crashes or GUI doesn't start

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libegl1 libgl1 xvfb \
    libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
```

### NumPy Compatibility Issues

**Symptom:** `numpy.ndarray size changed` or similar

**Solution:**

```bash
# Force reinstall NumPy and dependent packages
pip uninstall numpy scipy matplotlib -y
pip install numpy==1.26.4
pip install -r requirements.txt
```

### MuJoCo Not Found

**Symptom:** `mujoco` module not found

**Solution:**

```bash
# MuJoCo is now pip-installable (no longer requires license)
pip install mujoco

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"
```

---

## Engine Issues

### Drake Import Errors

**Symptom:** `ImportError: libdrake.so: cannot open shared object file`

**Solution:**

```bash
# Drake has specific pip requirements
pip install drake

# On some systems, set library path:
export LD_LIBRARY_PATH=/opt/drake/lib:$LD_LIBRARY_PATH
```

### Pinocchio Not Available

**Symptom:** `ModuleNotFoundError: No module named 'pinocchio'`

**Solution:**

```bash
# Pinocchio requires conda or building from source
conda install pinocchio -c conda-forge

# Or via pip (limited platform support):
pip install pin
```

### OpenSim Connection Failed

**Symptom:** `OpenSim adapter failed to connect`

**Solution:**

1. Ensure OpenSim is installed separately (not pip-installable)
2. Set `OPENSIM_HOME` environment variable:

```bash
export OPENSIM_HOME=/path/to/opensim
export PATH=$OPENSIM_HOME/bin:$PATH
```

### Engine Status Shows "Error"

**Symptom:** Engine appears in red in launcher

**Solution:**

1. Open diagnostic panel (Help > Show Diagnostics)
2. Check specific engine error message
3. Run validation:

```bash
python -c "from shared.python import EngineManager; m = EngineManager(); m.get_diagnostic_report()"
```

---

## GUI/Launcher Issues

### GUI Crashes on Startup

**Symptom:** Immediate crash or blank window

**Solution:**

1. Try running with verbose logging:

```bash
python launch_golf_suite.py --debug
```

2. Check Qt platform:

```bash
# Linux headless
export QT_QPA_PLATFORM=offscreen
python launch_golf_suite.py

# Or with Xvfb
xvfb-run python launch_golf_suite.py
```

### Plots Don't Render

**Symptom:** Empty or frozen matplotlib windows

**Solution:**

```python
# In your script, set backend before import:
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg' for fallback
import matplotlib.pyplot as plt
```

### High DPI Scaling Issues

**Symptom:** Tiny UI elements on 4K displays

**Solution:**

```bash
# Set environment variable
export QT_AUTO_SCREEN_SCALE_FACTOR=1
python launch_golf_suite.py
```

---

## Performance Issues

### Slow Startup (>10 seconds)

**Causes:**

- Heavy imports loading at startup
- Large model files being parsed

**Solution:**

1. Check which modules are slow:

```python
import time
start = time.time()
from shared.python import EngineManager
print(f"Import took {time.time()-start:.2f}s")
```

2. Ensure you're not importing heavy dependencies like matplotlib in `__init__.py`

### Simulation Running Slowly

**Causes:**

- Python overhead in tight loops
- Rendering every frame

**Solution:**

1. Use headless mode for batch processing:

```python
engine.set_render_mode("offscreen")
```

2. Reduce visualization frequency:

```python
engine.step(render_every=10)  # Only render every 10th frame
```

### Memory Usage High

**Symptom:** >4GB RAM usage

**Solution:**

1. Clear cached models:

```python
engine.cleanup()
```

2. Load models on-demand:

```python
# Instead of loading all at startup
engine.load_model_lazy("golf_swing_full.urdf")
```

---

## Test Failures

### Tests Pass Locally but Fail in CI

**Causes:**

- Missing environment variables
- Platform differences
- Random seed not fixed

**Solution:**

1. Run with same conditions as CI:

```bash
xvfb-run pytest -v  # On Linux
```

2. Check for random seed:

```python
import numpy as np
np.random.seed(42)  # Add to conftest.py
```

### Cross-Engine Validation Fails

**Symptom:** Engines produce different results beyond tolerance

**Solution:**

1. Check tolerance in test:

```python
# Increase if physics differences are expected
assert np.allclose(result_a, result_b, rtol=1e-3)  # 0.1% tolerance
```

2. See `docs/troubleshooting/cross_engine_deviations.md` for known differences

### Import Errors in Tests

**Symptom:** `ModuleNotFoundError` when running pytest

**Solution:**

```bash
# Install in development mode
pip install -e .[dev]

# Or ensure conftest.py adds paths:
# See conftest.py at repository root
```

---

## CI/CD Issues

### Ruff Check Fails

**Symptom:** `ruff check .` exits with non-zero

**Solution:**

```bash
# Auto-fix most issues
ruff check . --fix

# See what rules are failing
ruff check . --statistics
```

### Mypy Errors

**Symptom:** Type checking fails

**Solution:**

1. Install type stubs:

```bash
pip install types-PyYAML types-requests types-all
```

2. For third-party libs without stubs, add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["mujoco.*", "drake.*"]
ignore_missing_imports = true
```

### Black Formatting Fails

**Symptom:** `black --check .` finds differences

**Solution:**

```bash
# Auto-format
black .

# Verify
black --check .
```

### Pre-commit Hooks Not Running

**Solution:**

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Getting Help

If your issue isn't covered here:

1. **Search existing issues:** [GitHub Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
2. **Check assessment documents:** `docs/assessments/`
3. **Run diagnostics:** `python -c "from shared.python import EngineManager; print(EngineManager().get_diagnostic_report())"`
4. **Open a new issue** with:
   - Python version (`python --version`)
   - OS and version
   - Full error traceback
   - Steps to reproduce

---

_Last updated: January 2026_
