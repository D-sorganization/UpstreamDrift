# Installation Troubleshooting Guide

This guide addresses common installation issues identified in the project assessments.

## Quick Start

### Recommended: Conda Installation (Most Reliable)

```bash
# Clone the repository
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite

# Create conda environment (handles binary dependencies)
conda env create -f environment.yml
conda activate golf-suite

# Verify installation
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
python -c "from shared.python.interfaces import PhysicsEngine; print('✓ Interfaces loaded')"
```

### Alternative: Pip Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all dependencies
pip install -e ".[dev,engines,analysis]"
```

---

## Common Issues

### 1. MuJoCo Import Fails

**Symptom:**
```
ImportError: cannot import name 'mujoco' from 'mujoco'
```

**Cause:** MuJoCo requires specific system libraries and sometimes conflicts with pip installations.

**Solution (Windows):**
1. Install Visual C++ Redistributable 2015-2022:
   https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Use conda instead of pip:
   ```bash
   conda install -c conda-forge mujoco
   ```

**Solution (Linux):**
```bash
# Install OpenGL dependencies
sudo apt-get install libgl1-mesa-glx libosmesa6-dev libglfw3

# If still failing, try conda
conda install -c conda-forge mujoco
```

**Solution (macOS):**
```bash
# For Apple Silicon (M1/M2/M3)
CONDA_SUBDIR=osx-arm64 conda install -c conda-forge mujoco

# For Intel Macs
conda install -c conda-forge mujoco
```

### 2. OpenGL Version Error

**Symptom:**
```
mujoco.FatalError: OpenGL version 1.5 or higher required
```

**Cause:** Graphics driver too old or running in headless environment.

**Solution (Desktop):**
1. Update graphics drivers
2. Ensure you're not SSH'd without X forwarding

**Solution (Headless/CI):**
```bash
# Use OSMesa for software rendering
export MUJOCO_GL=osmesa

# Or use EGL
export MUJOCO_GL=egl
```

### 3. PyQt6 Fails to Import

**Symptom:**
```
ImportError: cannot import name 'QtCore' from 'PyQt6'
```

**Solution:**
```bash
# Conda installation (recommended)
conda install -c conda-forge pyqt

# Pip fallback
pip install PyQt6 --force-reinstall
```

### 4. Drake Installation Fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement drake
```

**Cause:** Drake has limited platform support (Ubuntu/macOS only for some versions).

**Solution:**
```bash
# Ubuntu 22.04
pip install drake

# macOS (Homebrew)
brew install drake

# Windows: Use Docker
docker-compose up drake-service
```

### 5. Pinocchio Installation Fails

**Symptom:**
```
ERROR: No matching distribution found for pin
```

**Solution:**
```bash
# Conda (recommended)
conda install -c conda-forge pinocchio

# Pip (may require additional system libraries)
pip install pin pin-pink
```

### 6. Permission Denied on Windows

**Symptom:**
```
PermissionError: [WinError 5] Access is denied
```

**Solution:**
1. Run terminal as Administrator
2. Or install in user directory:
   ```bash
   pip install --user -e .
   ```

### 7. Tests Fail During Collection

**Symptom:**
```
ERRORS during collection
ImportError: cannot import name 'mujoco'
```

**Cause:** Test collection imports all modules, including physics engines.

**Solution:**
```bash
# Skip tests requiring unavailable engines
pytest -m "not mujoco" tests/

# Or install the missing engine
conda install -c conda-forge mujoco
```

### 8. bcrypt Native Library Missing

**Symptom:**
```
passlib.exc.MissingBackendError: bcrypt: no backends available
```

**Solution:**
```bash
# Install bcrypt with native extension
pip install bcrypt --force-reinstall --no-binary :all:

# Or on Windows with conda
conda install -c conda-forge bcrypt
```

---

## Environment Verification Script

Run this script to verify your installation:

```python
#!/usr/bin/env python
"""Verify Golf Modeling Suite installation."""

import sys

def check_import(module_name: str, package: str | None = None) -> bool:
    """Try to import a module and report status."""
    try:
        __import__(package or module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

def main() -> int:
    """Run all checks."""
    print("=" * 50)
    print("Golf Modeling Suite - Installation Verification")
    print("=" * 50)
    
    checks = [
        ("numpy", None),
        ("scipy", None),
        ("pandas", None),
        ("matplotlib", None),
        ("PyQt6", "PyQt6.QtCore"),
        ("mujoco", None),
        ("fastapi", None),
        ("shared.python.interfaces", None),
        ("shared.python.ball_flight_physics", None),
    ]
    
    results = [check_import(name, pkg) for name, pkg in checks]
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ Installation verified successfully!")
        return 0
    else:
        print("✗ Some checks failed. See troubleshooting guide.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Save as `scripts/verify_installation.py` and run:
```bash
python scripts/verify_installation.py
```

---

## Light Installation (No Physics Engines)

For UI development without heavy physics dependencies:

```bash
# Install core only
pip install -e .

# Set mock engine mode
export GOLF_USE_MOCK_ENGINE=1

# Run launcher in light mode
python launchers/golf_suite_launcher.py --mock
```

---

## Docker Installation (Most Isolated)

For complete isolation and reproducibility:

```bash
# Build and run
docker-compose up golf-suite

# Or build manually
docker build -t golf-suite .
docker run -p 8000:8000 golf-suite
```

---

## Getting Help

If issues persist:

1. **Check GitHub Issues**: https://github.com/D-sorganization/Golf_Modeling_Suite/issues
2. **Create a new issue** with:
   - OS and version
   - Python version (`python --version`)
   - Full error traceback
   - Output of `pip freeze` or `conda list`
