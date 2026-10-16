## Appendix D: Configuration Reference

This appendix details all configuration files, their format, and available
options in the UpstreamDrift project.

---

### D.1 `pyproject.toml`

The primary project configuration file, following PEP 621 and using Hatchling
as the build backend.

**Location:** `/pyproject.toml`

#### D.1.1 Project Metadata

```toml
[project]
name = "upstream-drift"
version = "2.1.0"
description = "Biomechanical golf simulation and analysis suite"
requires-python = ">=3.11"
license = { text = "MIT" }
```

#### D.1.2 Core Dependencies

| Package           | Version Constraint | Purpose                    |
| ----------------- | ------------------ | -------------------------- |
| numpy             | >=1.26.4, <3.0.0   | Numerical computation      |
| scipy             | >=1.13.1           | Scientific algorithms      |
| fastapi           | >=0.126.0          | REST API framework         |
| uvicorn[standard] | >=0.30.0           | ASGI server                |
| pydantic          | >=2.5.0            | Data validation            |
| httpx             | >=0.27.0           | HTTP client                |
| mujoco            | >=3.3.0, <4.0.0    | Default physics engine     |
| simpleeval        | >=1.0.0            | Safe expression evaluation |
| structlog         | >=24.1.0           | Structured logging         |

#### D.1.3 Optional Dependency Groups

| Group         | Command                           | Packages                                       |
| ------------- | --------------------------------- | ---------------------------------------------- |
| `drake`       | `pip install -e ".[drake]"`       | drake>=1.22.0                                  |
| `pinocchio`   | `pip install -e ".[pinocchio]"`   | pin>=2.6.0, meshcat>=0.3.0                     |
| `all-engines` | `pip install -e ".[all-engines]"` | drake + pinocchio                              |
| `analysis`    | `pip install -e ".[analysis]"`    | opencv-python>=4.8.0, scikit-learn>=1.3.0      |
| `urdf`        | `pip install -e ".[urdf]"`        | trimesh>=4.0.0, PyYAML>=6.0, defusedxml>=0.7.0 |
| `dev`         | `pip install -e ".[dev]"`         | pytest, ruff, mypy, etc.                       |
| `gui-test`    | `pip install -e ".[gui-test]"`    | PyQt6>=6.5.0, pytest-qt>=4.2.0                 |
| `all`         | `pip install -e ".[all]"`         | All of the above (except gui-test)             |

#### D.1.4 Tool Configurations

**Ruff (linting):**

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501", "B008"]
```

| Rule Set | Description                      |
| -------- | -------------------------------- |
| E        | pycodestyle errors               |
| F        | Pyflakes                         |
| I        | isort (import ordering)          |
| UP       | pyupgrade (Python modernization) |
| B        | flake8-bugbear (common bugs)     |

**Mypy (type checking):**

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
check_untyped_defs = true
disallow_untyped_defs = false
```

**Pytest:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --strict-markers --strict-config"
markers = [
    "slow",
    "integration",
    "unit",
    "requires_gl",
    "headless_safe",
    "slow_numba",
    "asyncio",
    "benchmark",
]
```

---

### D.2 environment.yml

Conda environment specification for full installation with binary
dependencies.

**Location:** `/environment.yml`

**Environment name:** `upstream-drift`

**Channels:**

1. `conda-forge` (primary)
2. `defaults` (fallback)

#### D.2.1 Conda-Managed Dependencies

| Category      | Packages                                                                   |
| ------------- | -------------------------------------------------------------------------- |
| Core Python   | python=3.11, pip>=23.0                                                     |
| Scientific    | numpy>=1.26.4, scipy>=1.13.1, pandas>=2.2.3, sympy>=1.12                   |
| Visualization | matplotlib>=3.8.0, opencv>=4.9.0                                           |
| GUI           | pyqt>=6.6.0                                                                |
| Physics       | mujoco>=3.3.0                                                              |
| Dev tools     | pytest>=8.2.1, pytest-cov, pytest-mock, pytest-qt, black, mypy, pre-commit |

#### D.2.2 Pip-Managed Dependencies

| Category         | Packages                                           |
| ---------------- | -------------------------------------------------- |
| API              | fastapi, uvicorn, slowapi, python-multipart, httpx |
| Database/Auth    | sqlalchemy, pydantic, PyJWT, bcrypt, cryptography  |
| Data formats     | pyyaml, defusedxml, ezc3d                          |
| Logging/Security | structlog, simpleeval                              |
| Linting          | ruff==0.14.10                                      |
| Type stubs       | types-PyYAML, pandas-stubs, types-requests         |

#### D.2.3 Platform-Specific Notes

| Platform                  | Notes                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Windows**               | Requires Visual C++ Redistributable 2015-2022 for MuJoCo. Update GPU drivers if OpenGL errors occur.      |
| **macOS (Apple Silicon)** | Use `CONDA_SUBDIR=osx-arm64 conda env create ...` for M1/M2/M3 chips. Some engines may lack ARM64 builds. |
| **Linux**                 | Install OpenGL libraries: `sudo apt install libgl1-mesa-glx`. For headless servers: `xvfb-run pytest`.    |

#### D.2.4 Installation Commands

```bash
# Full installation
conda env create -f environment.yml
conda activate upstream-drift

# Update existing environment
conda env update -f environment.yml --prune

# Light installation (no heavy engines)
conda env create -f environment.yml --name golf-suite-light
```

---

### D.3 Makefile Targets

**Location:** `/Makefile`

| Target      | Command          | Description                                                                                        |
| ----------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `help`      | `make help`      | Display available targets and descriptions                                                         |
| `install`   | `make install`   | Install requirements.txt and package in editable mode with dev extras                              |
| `lint`      | `make lint`      | Run ruff check and mypy (mypy errors are advisory)                                                 |
| `format`    | `make format`    | Run ruff format and ruff check --fix                                                               |
| `test`      | `make test`      | Run full pytest suite with verbose output                                                          |
| `test-unit` | `make test-unit` | Run only `tests/unit/`                                                                             |
| `test-int`  | `make test-int`  | Run only `tests/integration/`                                                                      |
| `check`     | `make check`     | Run lint + test sequentially                                                                       |
| `docs`      | `make docs`      | Build Sphinx documentation (HTML)                                                                  |
| `clean`     | `make clean`     | Remove `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.egg-info`, build artifacts |
| `all`       | `make all`       | install, format, lint, test (in order)                                                             |

---

### D.4 Docker Configuration

#### D.4.1 `Dockerfile.unified`

Multi-stage build for the complete application container.

**Location:** `/Dockerfile.unified`

**Stage 1: UI Builder**

| Property   | Value                                     |
| ---------- | ----------------------------------------- |
| Base image | `node:20-slim`                            |
| Purpose    | Build the web UI (npm ci + npm run build) |
| Output     | `/app/ui/dist` (static assets)            |

**Stage 2: Python Runtime**

| Property     | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| Base image   | `mambaorg/micromamba:1.5-jammy`                                       |
| System deps  | libgl1-mesa-glx, libegl1, libglib2.0-0, libsm6, libxrender1, libxext6 |
| Python env   | Created from `environment.yml` via micromamba                         |
| Application  | Installed in editable mode                                            |
| User         | Non-root `golfer` (UID 1000)                                          |
| Exposed port | 8000                                                                  |
| Health check | `curl -f http://localhost:8000/api/health` (30s interval)             |
| Default CMD  | `upstream-drift --no-browser`                                         |

#### D.4.2 Running the Container

```bash
# Build
docker build -f Dockerfile.unified -t upstream-drift:latest .

# Run (API server mode)
docker run -p 8000:8000 upstream-drift:latest

# Run with GPU access (NVIDIA)
docker run --gpus all -p 8000:8000 upstream-drift:latest

# Run interactive CLI
docker run -it upstream-drift:latest bash
```

#### D.4.3 Environment Variables

| Variable                    | Default          | Description               |
| --------------------------- | ---------------- | ------------------------- |
| `ENV_NAME`                  | `upstream-drift` | Conda environment name    |
| `MAMBA_DOCKERFILE_ACTIVATE` | `1`              | Auto-activate environment |
| `GMS_LOG_LEVEL`             | `INFO`           | Logging verbosity         |
| `GMS_API_PORT`              | `8000`           | API server port           |
| `GMS_API_HOST`              | `0.0.0.0`        | API server bind address   |

---

### D.5 Logging Configuration

UpstreamDrift uses `structlog` for structured JSON logging throughout.

| Level      | Usage                                                                  |
| ---------- | ---------------------------------------------------------------------- |
| `DEBUG`    | Detailed computational values, per-step simulation data                |
| `INFO`     | Normal operations: engine loading, simulation start/stop, API requests |
| `WARNING`  | Recoverable issues: gimbal lock proximity, suboptimal convergence      |
| `ERROR`    | Operation failures: simulation divergence, file not found              |
| `CRITICAL` | Unrecoverable: engine crash, data corruption                           |

**Log format (production):**

```json
{
  "timestamp": "2026-02-05T10:30:00.000Z",
  "level": "info",
  "event": "simulation_complete",
  "engine": "mujoco",
  "model": "full_body_15dof",
  "duration_ms": 1523,
  "steps": 10000,
  "request_id": "req_abc123"
}
```

---

### D.6 Numerical Constants

Key numerical constants defined in `src/shared/python/physics_constants.py`
and `src/shared/python/constants.py`:

| Constant                      | Value      | Unit     | Source                   |
| ----------------------------- | ---------- | -------- | ------------------------ |
| `GRAVITY_M_S2`                | 9.80665    | m/s$^2$  | NIST CODATA 2018         |
| `AIR_DENSITY_SEA_LEVEL_KG_M3` | 1.225      | kg/m$^3$ | ISA Standard Atmosphere  |
| `GOLF_BALL_MASS_KG`           | 0.04593    | kg       | USGA Rule 5-1            |
| `GOLF_BALL_DIAMETER_M`        | 0.04267    | m        | USGA Rule 5-2            |
| `DRIVER_COR`                  | 0.78       | --       | USGA Rule Limit          |
| `TYPICAL_CONTACT_DURATION_S`  | 0.0005     | s        | High-speed video studies |
| `GRAPHITE_DENSITY_KG_M3`      | 1750       | kg/m$^3$ | Materials Handbook       |
| `STEEL_DENSITY_KG_M3`         | 7850       | kg/m$^3$ | Materials Handbook       |
| `TITANIUM_DENSITY_KG_M3`      | 4506       | kg/m$^3$ | Materials Handbook       |
| `MAGNUS_COEFFICIENT`          | 0.25       | --       | Bearman & Harvey         |
| `SPIN_DECAY_RATE_S`           | 0.05       | 1/s      | Trackman Data            |
| `EPSILON`                     | $10^{-15}$ | --       | Solver numerical floor   |
| `CONVERGENCE_TOLERANCE`       | $10^{-6}$  | --       | Default solver tolerance |
| `MAX_ITERATIONS`              | 10000      | --       | Default iteration cap    |
| `DEFAULT_RANDOM_SEED`         | 42         | --       | Reproducibility          |

---

## Appendix E: Error Codes Reference

UpstreamDrift uses a structured error code system defined in
`src/api/utils/error_codes.py`. All errors follow a consistent format and
include machine-readable codes for automated error handling.

---

### E.1 Error Code Format

```
GMS-{CATEGORY}-{NUMBER}
```

- **GMS**: Golf Modeling Suite (project prefix)
- **CATEGORY**: Three-letter category code identifying the subsystem
- **NUMBER**: Three-digit sequential error number within the category

#### E.1.1 Error Categories

| Code  | Category   | Description                                                |
| ----- | ---------- | ---------------------------------------------------------- |
| `GEN` | General    | Generic/uncategorized errors                               |
| `ENG` | Engine     | Physics engine loading, initialization, and runtime errors |
| `SIM` | Simulation | Simulation execution, parameter, and state errors          |
| `VID` | Video      | Video processing and pose estimation errors                |
| `ANL` | Analysis   | Swing analysis and computation errors                      |
| `AUT` | Auth       | Authentication, authorization, and quota errors            |
| `VAL` | Validation | Input validation and schema errors                         |
| `RES` | Resource   | Resource lifecycle errors (CRUD operations)                |
| `SYS` | System     | Infrastructure, database, and dependency errors            |

---

### E.2 Complete Error Code Table

#### E.2.1 General Errors (`GMS-GEN-xxx`)

| Code          | Name                | HTTP Status | Description                                  | Resolution                                                                         |
| ------------- | ------------------- | ----------- | -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `GMS-GEN-001` | INTERNAL_ERROR      | 500         | An unexpected internal server error occurred | Check server logs for stack trace. Report if persistent.                           |
| `GMS-GEN-002` | INVALID_REQUEST     | 400         | The request format or parameters are invalid | Review the API documentation for the correct request format.                       |
| `GMS-GEN-003` | RATE_LIMITED        | 429         | Rate limit exceeded                          | Wait and retry after the indicated cooldown period. Reduce request frequency.      |
| `GMS-GEN-004` | SERVICE_UNAVAILABLE | 503         | The service is temporarily unavailable       | The server may be restarting or under maintenance. Retry with exponential backoff. |

#### E.2.2 Engine Errors (`GMS-ENG-xxx`)

| Code          | Name                         | HTTP Status | Description                                          | Resolution                                                                           |
| ------------- | ---------------------------- | ----------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `GMS-ENG-001` | ENGINE_NOT_FOUND             | 404         | The specified physics engine was not found           | Verify the engine name. Available engines: mujoco, drake, pinocchio.                 |
| `GMS-ENG-002` | ENGINE_NOT_LOADED            | 400         | No physics engine is currently loaded                | Call the engine loading endpoint before attempting simulation.                       |
| `GMS-ENG-003` | ENGINE_LOAD_FAILED           | 500         | The physics engine failed to load                    | Check that engine dependencies are installed. Verify model file paths.               |
| `GMS-ENG-004` | ENGINE_NOT_AVAILABLE         | 400         | The requested engine is not installed on this system | Install the engine: `pip install -e ".[drake]"` or `pip install -e ".[pinocchio]"`.  |
| `GMS-ENG-005` | ENGINE_INITIALIZATION_FAILED | 500         | Engine initialization failed after loading           | Check model file validity (URDF/MJCF syntax). Verify system resources (GPU, memory). |
| `GMS-ENG-006` | ENGINE_INVALID_STATE         | 400         | The engine is in an invalid state for this operation | Reset the engine state or reload the model before retrying.                          |

#### E.2.3 Simulation Errors (`GMS-SIM-xxx`)

| Code          | Name                       | HTTP Status | Description                                               | Resolution                                                                                           |
| ------------- | -------------------------- | ----------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `GMS-SIM-001` | SIMULATION_FAILED          | 500         | The simulation execution failed                           | Check simulation parameters for physical validity (time step, joint limits). Review model stability. |
| `GMS-SIM-002` | SIMULATION_TIMEOUT         | 504         | The simulation exceeded the maximum allowed time          | Reduce simulation duration, increase time step, or simplify the model.                               |
| `GMS-SIM-003` | SIMULATION_INVALID_PARAMS  | 400         | One or more simulation parameters are invalid             | Verify parameter ranges: time_step > 0, duration > 0, joint angles within limits.                    |
| `GMS-SIM-004` | SIMULATION_MODEL_NOT_FOUND | 404         | The specified simulation model file was not found         | Verify the model path. Run `upstream-drift --setup-models` to download standard models.              |
| `GMS-SIM-005` | SIMULATION_STATE_ERROR     | 500         | The simulation reached an invalid state (NaN, divergence) | Reduce time step, check for unrealistic forces, verify contact parameters.                           |
| `GMS-SIM-006` | TASK_NOT_FOUND             | 404         | The specified async task ID was not found                 | The task may have expired or the ID is incorrect. Submit a new simulation request.                   |
| `GMS-SIM-007` | TASK_NOT_COMPLETED         | 400         | The task has not completed yet; results not available     | Poll the task status endpoint until completion, or use the WebSocket notification channel.           |

#### E.2.4 Video Processing Errors (`GMS-VID-xxx`)

| Code          | Name                           | HTTP Status | Description                                       | Resolution                                                                    |
| ------------- | ------------------------------ | ----------- | ------------------------------------------------- | ----------------------------------------------------------------------------- |
| `GMS-VID-001` | VIDEO_PIPELINE_NOT_INITIALIZED | 500         | The video processing pipeline is not initialized  | Ensure analysis dependencies are installed: `pip install -e ".[analysis]"`.   |
| `GMS-VID-002` | VIDEO_INVALID_FORMAT           | 400         | The uploaded file is not a supported video format | Supported formats: MP4, AVI, MOV, MKV. Maximum file size: 500 MB.             |
| `GMS-VID-003` | VIDEO_PROCESSING_FAILED        | 500         | Video processing failed during analysis           | Check video quality (resolution >= 720p recommended). Ensure adequate memory. |
| `GMS-VID-004` | VIDEO_ESTIMATOR_INVALID        | 400         | Invalid pose estimator type specified             | Available estimators: mediapipe, openpose, movenet.                           |
| `GMS-VID-005` | VIDEO_CONFIDENCE_INVALID       | 400         | Confidence threshold out of valid range           | Confidence must be in range [0.0, 1.0]. Default: 0.5.                         |

#### E.2.5 Analysis Errors (`GMS-ANL-xxx`)

| Code          | Name                             | HTTP Status | Description                                   | Resolution                                                                                          |
| ------------- | -------------------------------- | ----------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `GMS-ANL-001` | ANALYSIS_SERVICE_NOT_INITIALIZED | 500         | The analysis service has not been initialized | The server may need to be restarted. Check service health endpoint.                                 |
| `GMS-ANL-002` | ANALYSIS_FAILED                  | 500         | The analysis computation failed               | Check input data quality. Ensure sufficient trajectory data points (minimum 10).                    |
| `GMS-ANL-003` | ANALYSIS_INVALID_TYPE            | 400         | The requested analysis type is not recognized | Available types: swing_metrics, energy_analysis, kinematic_sequence, phase_detection, grf_analysis. |

#### E.2.6 Authentication Errors (`GMS-AUT-xxx`)

| Code          | Name                          | HTTP Status | Description                                                 | Resolution                                                     |
| ------------- | ----------------------------- | ----------- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| `GMS-AUT-001` | AUTH_TOKEN_INVALID            | 401         | The authentication token is invalid or malformed            | Ensure the Authorization header contains a valid Bearer token. |
| `GMS-AUT-002` | AUTH_TOKEN_EXPIRED            | 401         | The authentication token has expired                        | Request a new token via the `/api/auth/token` endpoint.        |
| `GMS-AUT-003` | AUTH_INSUFFICIENT_PERMISSIONS | 403         | The authenticated user lacks permissions for this operation | Contact the administrator to request elevated permissions.     |
| `GMS-AUT-004` | AUTH_QUOTA_EXCEEDED           | 429         | The usage quota for this billing period has been exceeded   | Upgrade your plan or wait until the next billing period.       |
| `GMS-AUT-005` | AUTH_USER_NOT_FOUND           | 404         | The specified user account was not found                    | Verify the username/email. Register a new account if needed.   |

#### E.2.7 Validation Errors (`GMS-VAL-xxx`)

| Code          | Name                     | HTTP Status | Description                                                  | Resolution                                                         |
| ------------- | ------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `GMS-VAL-001` | VALIDATION_FAILED        | 422         | General request validation failed                            | Review the error details for specific field violations.            |
| `GMS-VAL-002` | VALIDATION_MISSING_FIELD | 422         | A required field is missing from the request                 | Add the missing field as indicated in the error details.           |
| `GMS-VAL-003` | VALIDATION_INVALID_VALUE | 422         | A field value is outside its valid range or type             | Check the field type and range constraints in the API schema.      |
| `GMS-VAL-004` | VALIDATION_PATH_INVALID  | 400         | A file path is invalid, does not exist, or is not accessible | Verify the path exists and the server process has read permission. |

#### E.2.8 Resource Errors (`GMS-RES-xxx`)

| Code          | Name                    | HTTP Status | Description                                        | Resolution                                                                |
| ------------- | ----------------------- | ----------- | -------------------------------------------------- | ------------------------------------------------------------------------- |
| `GMS-RES-001` | RESOURCE_NOT_FOUND      | 404         | The requested resource was not found               | Verify the resource ID or path. The resource may have been deleted.       |
| `GMS-RES-002` | RESOURCE_ALREADY_EXISTS | 409         | A resource with the same identifier already exists | Use a different identifier, or update/delete the existing resource first. |
| `GMS-RES-003` | RESOURCE_ACCESS_DENIED  | 403         | Access to the resource is denied                   | Check resource ownership and sharing permissions.                         |

#### E.2.9 System Errors (`GMS-SYS-xxx`)

| Code          | Name                | HTTP Status | Description                                     | Resolution                                                               |
| ------------- | ------------------- | ----------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| `GMS-SYS-001` | DATABASE_ERROR      | 500         | A database operation failed                     | Check database connectivity and disk space. Retry the operation.         |
| `GMS-SYS-002` | CONFIGURATION_ERROR | 500         | The server configuration is invalid             | Review environment variables and configuration files.                    |
| `GMS-SYS-003` | DEPENDENCY_ERROR    | 500         | A required external dependency is not available | Install the missing dependency. Check `pip list` for installed packages. |

---

### E.3 Error Response Format

All API error responses follow a consistent JSON structure:

```json
{
  "error": {
    "code": "GMS-ENG-003",
    "message": "Failed to load physics engine",
    "details": {
      "engine_type": "drake",
      "reason": "ImportError: No module named 'pydrake'"
    },
    "request_id": "req_a1b2c3d4",
    "correlation_id": "corr_x7y8z9"
  }
}
```

| Field            | Type   | Description                                      |
| ---------------- | ------ | ------------------------------------------------ |
| `code`           | string | Machine-readable error code (GMS-XXX-YYY format) |
| `message`        | string | Human-readable error description                 |
| `details`        | object | Additional context (optional, varies by error)   |
| `request_id`     | string | Unique identifier for this request (for support) |
| `correlation_id` | string | Trace ID for cross-service correlation           |

---

### E.4 Exception Hierarchy

The internal Python exception hierarchy in `src/shared/python/error_utils.py`:

```
GolfSuiteError (base)
  |-- EngineNotAvailableError
  |-- ConfigurationError
  |     |-- EnvironmentError
  |-- ValidationError
  |     |-- PhysicalValidationError
  |-- ModelError
  |-- SimulationError
  |-- FileOperationError
  |-- IOError
  |     |-- FileNotFoundIOError
  |     |-- FileParseError
  |-- DataFormatError
  |-- TimeoutError
  |-- ResourceError
```

Additional domain-specific exceptions:

```
ContractViolationError (from contracts.py)
  |-- PreconditionError
  |-- PostconditionError
  |-- InvariantError
  |-- StateError

RoboticsError (from robotics/core/exceptions.py)
  |-- ContactError
  |-- ControlError
  |-- SolverError
  |-- LocomotionError
  |-- KinematicsError

AIError (from ai/exceptions.py)
  |-- AIProviderError
  |     |-- AIConnectionError
  |     |-- AIRateLimitError
  |     |-- AITimeoutError
  |-- ScientificValidationError
  |-- WorkflowError
  |-- ToolExecutionError
```

---

### E.5 Contract Violation Errors

Design by Contract violations (from `src/shared/python/contracts.py`) are
a special class of runtime errors that indicate programming logic errors
rather than user input errors.

| Exception            | Trigger                                                    | Example                                      |
| -------------------- | ---------------------------------------------------------- | -------------------------------------------- |
| `PreconditionError`  | Input constraint violated before method execution          | Calling `step()` on an uninitialized engine  |
| `PostconditionError` | Output constraint violated after method execution          | Computed acceleration contains NaN values    |
| `InvariantError`     | Class invariant violated after a state-modifying operation | Mass becomes negative after parameter update |
| `StateError`         | Operation attempted in an invalid state                    | Attempting simulation before model is loaded |

Contracts can be globally disabled for production performance:

```python
from src.shared.python.contracts import disable_contracts
disable_contracts()  # Skip all contract checks
```

---
