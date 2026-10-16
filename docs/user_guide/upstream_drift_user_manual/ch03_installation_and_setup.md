# Chapter 3: Installation and Setup

## 3.1 System Prerequisites

Before installing UpstreamDrift, ensure your system meets the following
requirements:

### Required Software

| Software           | Minimum Version | Recommended Version | Notes                        |
| ------------------ | --------------- | ------------------- | ---------------------------- |
| Python             | 3.11            | 3.12                | Python 3.13 also supported   |
| Git                | 2.30+           | Latest              | Git LFS support required     |
| Git LFS            | 2.13+           | Latest              | For large model files        |
| pip                | 23.0+           | Latest              | Python package manager       |
| conda (optional)   | 23.0+           | Latest (Miniforge)  | Recommended for binary deps  |
| Docker (optional)  | 24.0+           | Latest              | For containerized deployment |
| Node.js (optional) | 18+             | 20 LTS              | Only for Tauri/React UI dev  |
| Rust (optional)    | 1.70+           | Latest              | Only for Tauri desktop app   |

### Required System Libraries

**Linux (Ubuntu/Debian)**:

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libeigen3-dev \
    patchelf \
    ffmpeg \
    xvfb
```

**macOS**:

```bash
xcode-select --install
brew install cmake eigen ffmpeg
```

**Windows**:

- Visual C++ Redistributable 2015-2022 (required for MuJoCo)
- CMake (for building native extensions)
- Git for Windows with Git LFS

### Optional Software

| Software           | Purpose                               |
| ------------------ | ------------------------------------- |
| MATLAB R2023a+     | Simscape Multibody models (2D/3D)     |
| Simulink           | Required by MATLAB models             |
| Simscape Multibody | Required by MATLAB models             |
| CUDA Toolkit       | GPU acceleration for learning modules |
| Xvfb               | Headless rendering on Linux servers   |

### Hardware Recommendations

| Component | Minimum      | Recommended           | Notes                                  |
| --------- | ------------ | --------------------- | -------------------------------------- |
| CPU       | 4 cores      | 8+ cores              | Multi-core for parallel simulation     |
| RAM       | 8 GB         | 16+ GB                | Musculoskeletal models need 8+ GB      |
| Storage   | 10 GB free   | 20+ GB free           | Models, data, conda env                |
| GPU       | Not required | NVIDIA (CUDA capable) | Only for RL and differentiable physics |
| Display   | 1920x1080    | 2560x1440+            | For GUI visualization                  |

## 3.2 Installation via Conda (Recommended)

Conda installation is the recommended approach because it correctly handles
binary dependencies such as MuJoCo, OpenCV, and PyQt6 that can be problematic
with pip on certain platforms.

### Step 1: Clone the Repository

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull
```

### Step 2: Create the Conda Environment

The `environment.yml` file defines the complete environment, including conda
packages (for binary dependencies) and pip packages (for Python-only libraries):

```bash
# Full installation (all engines and tools)
conda env create -f environment.yml
```

This creates a conda environment named `upstream-drift` with Python 3.11 and all
required dependencies. The environment includes:

- **Core scientific stack**: NumPy, SciPy, pandas, SymPy, Matplotlib
- **Physics engines**: MuJoCo (via conda), with Drake and Pinocchio available
  as optional uncommented entries
- **GUI framework**: PyQt6 (via conda, ensuring binary compatibility)
- **Web framework**: FastAPI, Uvicorn, SQLAlchemy, PyJWT, bcrypt (via pip)
- **Development tools**: pytest, black, mypy, ruff, pre-commit
- **Data formats**: PyYAML, defusedxml, ezc3d
- **Security**: structlog, simpleeval, cryptography

### Step 3: Activate the Environment

```bash
conda activate upstream-drift
```

### Step 4: Install the Package in Editable Mode

The `environment.yml` includes `-e .` in its pip section, which installs the
UpstreamDrift package in editable (development) mode. If this step did not
execute during environment creation, run it manually:

```bash
pip install -e ".[dev]"
```

### Step 5: (Optional) Enable Additional Engines

To install Drake and Pinocchio, either uncomment the relevant lines in
`environment.yml` and run `conda env update -f environment.yml --prune`, or
install them separately:

```bash
# Drake
pip install drake>=1.22.0

# Pinocchio (recommended via conda for binary compatibility)
conda install -c conda-forge pinocchio crocoddyl
```

### Updating an Existing Environment

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes packages that are no longer listed in the
environment file, keeping the environment clean.

## 3.3 Installation via Pip

For users who prefer pip over conda, or when conda is not available:

### Basic Installation

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull

# Install with core dependencies only
pip install -e .
```

This installs the minimum required packages: NumPy, SciPy, FastAPI, Uvicorn,
Pydantic, httpx, MuJoCo, simpleeval, and structlog.

### Installation With Optional Dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with all physics engines
pip install -e ".[all-engines]"

# Install with analysis extras (OpenCV, scikit-learn)
pip install -e ".[analysis]"

# Install with URDF generation tools
pip install -e ".[urdf]"

# Install with GUI test support
pip install -e ".[gui-test]"

# Install everything
pip install -e ".[all]"
```

### Light Installation (UI Development)

For frontend developers who only need to work on the UI without running
physics engines:

```bash
pip install -e .
export GOLF_USE_MOCK_ENGINE=1
```

The `GOLF_USE_MOCK_ENGINE` environment variable causes the platform to use a
mock engine that returns synthetic data, eliminating the need for heavy physics
libraries during UI development.

## 3.4 Docker Installation

Docker provides an isolated, reproducible environment that is particularly useful
for CI/CD, deployment, and users who want to avoid managing system dependencies.

### Building the Docker Image

```bash
# Standard image (MuJoCo + core tools)
docker build -t upstreamdrift:latest .

# Unified image (MuJoCo + Drake + Pinocchio + full toolchain)
docker build -f Dockerfile.unified -t upstreamdrift:unified .
```

The Dockerfile uses a multi-stage build:

1. **Builder stage** (`continuumio/miniconda3:24.11.1-0`): Installs build tools,
   compiles native extensions, and installs the full conda environment including
   MuJoCo, Drake, and Pinocchio.

2. **Runtime stage** (`continuumio/miniconda3:24.11.1-0`): Copies only the
   installed conda environment from the builder, installs minimal runtime
   dependencies (GL libraries, FFmpeg, curl), creates a non-root user (`golfer`),
   and sets up the workspace.

### Running the Container

```bash
# Interactive shell
docker run -it --rm \
    -p 8000:8000 \
    -v $(pwd):/workspace \
    upstreamdrift:latest

# Start API server
docker run -d --name upstreamdrift-api \
    -p 8000:8000 \
    -e SECRET_KEY="your-secret-key" \
    -e ENVIRONMENT="production" \
    upstreamdrift:latest \
    python start_api_server.py

# Run tests
docker run --rm \
    upstreamdrift:latest \
    pytest tests/ -v --tb=short
```

### Docker Environment

The container configures the following environment:

- `PYTHONPATH=/workspace:/workspace/shared/python:/workspace/engines`
- Working directory: `/workspace`
- User: `golfer` (UID 1000, non-root for security)
- Exposed port: `8000` (API server)
- Health check: `curl -f http://localhost:8000/api/health` every 30 seconds

### Headless Rendering

For servers without a display, the container includes Xvfb and OSMesa for
software-based OpenGL rendering:

```bash
# Run with virtual framebuffer for visualization tests
docker run --rm \
    upstreamdrift:latest \
    xvfb-run pytest tests/ -v
```

## 3.5 Environment Configuration

UpstreamDrift uses environment variables for runtime configuration, supporting
both development and production deployments.

### Core Environment Variables

| Variable               | Default                            | Description                               |
| ---------------------- | ---------------------------------- | ----------------------------------------- |
| `SECRET_KEY`           | (auto-generated)                   | JWT signing secret (set in production!)   |
| `GOLF_API_SECRET_KEY`  | (fallback for above)               | Alternative name for JWT secret           |
| `GOLF_ADMIN_PASSWORD`  | (auto-generated)                   | Admin user password                       |
| `DATABASE_URL`         | `sqlite:///golf_modeling_suite.db` | SQLAlchemy database URL                   |
| `ENVIRONMENT`          | `development`                      | `development` or `production`             |
| `API_HOST`             | `127.0.0.1`                        | API server bind address                   |
| `API_PORT`             | `8000`                             | API server bind port                      |
| `GOLF_PORT`            | `8000`                             | Alternative port configuration            |
| `GOLF_NO_BROWSER`      | (unset)                            | If set, don't auto-open browser           |
| `GOLF_USE_MOCK_ENGINE` | (unset)                            | If set, use mock engine (no real physics) |
| `GOLF_DEFAULT_ENGINE`  | (unset)                            | Default engine for web UI                 |
| `GOLF_LOG_LEVEL`       | `INFO`                             | Logging level (DEBUG, INFO, WARNING, etc) |

### Security Configuration

> **Warning**: In production, you **must** set `SECRET_KEY` to a strong,
> randomly generated value. Failure to do so will result in an auto-generated
> key that changes on each restart, invalidating all existing JWT tokens.

Generate a secure key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

Set it in your environment:

```bash
export SECRET_KEY="your-generated-secret-key-here"
export GOLF_ADMIN_PASSWORD="your-strong-admin-password"
export ENVIRONMENT="production"
```

### Configuration Precedence

Configuration values are resolved in the following order (highest priority first):

1. Command-line arguments (e.g., `--port 9000`)
2. Environment variables (e.g., `API_PORT=9000`)
3. `.env` file in the repository root (if present)
4. Default values defined in source code

## 3.6 Verifying the Installation

After installation, run the verification script to confirm that all core
dependencies are available and functional:

```bash
python scripts/ci/verify_installation.py
```

This script checks:

1. **Core scientific libraries**: NumPy, SciPy, pandas, Matplotlib, SymPy
2. **GUI framework**: PyQt6
3. **Physics engines**: MuJoCo (required), Drake (optional), Pinocchio (optional)
4. **Web framework**: FastAPI, Uvicorn
5. **Data format libraries**: PyYAML, defusedxml
6. **Security libraries**: PyJWT, bcrypt, cryptography

Expected output for a complete installation:

```
============================================================
Golf Modeling Suite - Installation Verification
============================================================

Checking core dependencies:
----------------------------------------
  numpy (v1.26.4)
  scipy (v1.13.1)
  pandas (v2.2.3)
  matplotlib (v3.8.5)
  sympy (v1.12)
  PyQt6 (v6.6.1)
  mujoco (v3.3.0)
  fastapi (v0.126.0)
  uvicorn (v0.30.1)
  yaml (v6.0.1)
  defusedxml (v0.7.1)

Checking physics engines:
----------------------------------------
  MuJoCo: Available
  Drake:  Available
  Pinocchio: Available
  OpenSim: Not installed (optional)
  MyoSuite: Not installed (optional)

============================================================
Installation verified: All core checks passed.
============================================================
```

### Running the Test Suite

For a more thorough verification, run the test suite:

```bash
# Quick verification (unit tests only, ~2 minutes)
make test-unit

# Full verification (all tests, ~10 minutes)
make test

# Specific test category
pytest tests/physics_validation/ -v --tb=short
```

### Checking System Health

The system health checker provides a comprehensive diagnostic:

```bash
python scripts/check_system_health.py
```

## 3.7 Platform-Specific Notes

### Linux

- **OpenGL**: Ensure Mesa OpenGL libraries are installed for MuJoCo visualization:
  ```bash
  sudo apt install libgl1-mesa-glx libosmesa6-dev
  ```
- **Headless servers**: Use Xvfb for rendering on servers without a display:
  ```bash
  xvfb-run python launch_upstream_drift.py
  xvfb-run pytest tests/ -v
  ```
- **WSL (Windows Subsystem for Linux)**: Refer to `docs/wsl_setup.md` for
  WSL-specific configuration including display forwarding.

### macOS

- **Apple Silicon (M1/M2/M3)**: For ARM64-native conda environments:
  ```bash
  CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
  ```
  Some engines (particularly Drake) may not yet have ARM64 builds. In this case,
  use the x86_64 conda subdir with Rosetta 2 emulation:
  ```bash
  CONDA_SUBDIR=osx-64 conda env create -f environment.yml
  ```
- **XQuartz**: Required for X11 forwarding if using SSH-based remote development.

### Windows

- **Visual C++ Redistributable**: MuJoCo requires the Visual C++ Redistributable
  2015-2022. Download from Microsoft's website if OpenGL errors occur.
- **Graphics drivers**: Update to the latest GPU drivers if you encounter
  OpenGL-related crashes.
- **Path length**: Windows has a 260-character path limit by default. Enable
  long paths via Group Policy or registry if you encounter path-related errors.
- **WSL recommended**: For the best experience on Windows, consider using WSL2
  with Ubuntu. See `run_wsl.sh` for a convenience launcher.

## 3.8 Development Environment Setup

For contributors and developers extending the platform:

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs pytest, pytest-cov, pytest-mock, pytest-timeout, pytest-xdist,
ruff, mypy, and build tools.

### Set up Pre-Commit Hooks

```bash
pip install pre-commit
pre-commit install
```

The pre-commit configuration runs the following checks on every commit:

- **ruff**: Fast Python linter (checks for errors, style issues, import ordering)
- **ruff format**: Code formatter (line length 88, target Python 3.11)
- **mypy**: Static type checker (advisory mode)
- **Trailing whitespace removal**
- **YAML validation**
- **Large file detection**

### IDE Configuration

**VS Code** (recommended):

- Install the Python, Pylance, and Ruff extensions.
- The project includes `pyproject.toml` settings that VS Code and Pylance
  will automatically detect for linting and formatting.

**PyCharm**:

- Mark `src/` as a source root.
- Configure the Python interpreter to use the `upstream-drift` conda environment.
- Enable ruff as an external tool for linting.

### Makefile Targets

The `Makefile` provides convenient shortcuts for common development tasks:

```bash
make help      # Show all available targets
make install   # Install dependencies (pip install -r requirements.txt && pip install -e ".[dev]")
make lint      # Run ruff check and mypy
make format    # Run ruff format and ruff check --fix
make test      # Run full test suite (pytest tests/ -v --tb=short)
make test-unit # Run unit tests only
make test-int  # Run integration tests only
make check     # Run lint + test
make docs      # Build Sphinx documentation
make clean     # Remove __pycache__, .pytest_cache, .mypy_cache, build artifacts
make all       # install + format + lint + test
```

## 3.9 Troubleshooting Installation Issues

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'mujoco'`

**Solution**: Ensure MuJoCo is installed. With conda: `conda install -c conda-forge mujoco`.
With pip: `pip install mujoco>=3.3.0`.

---

**Problem**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution**: Install Mesa OpenGL libraries:

```bash
sudo apt install libgl1-mesa-glx libgl1-mesa-dev
```

---

**Problem**: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`

**Solution**: Install XCB libraries or set the platform to offscreen:

```bash
sudo apt install libxcb-xinerama0 libxcb-cursor0
# or for headless:
export QT_QPA_PLATFORM=offscreen
```

---

**Problem**: Conda environment creation fails with solver conflicts.

**Solution**: Use the libmamba solver for faster, more reliable dependency resolution:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f environment.yml
```

---

**Problem**: `pip install -e .` fails with `build-backend` errors.

**Solution**: Ensure hatchling is installed: `pip install hatchling`. Then retry
the installation.

---

For additional troubleshooting, consult `docs/troubleshooting/installation.md`.

---
