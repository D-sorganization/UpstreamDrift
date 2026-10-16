# Chapter 23: REST API Reference

The UpstreamDrift REST API is a FastAPI application that exposes the
simulation, analysis, and video processing capabilities over HTTP and
WebSocket protocols.

---

## 23.1 Server Architecture

### 23.1.1 Framework

The server is built on **FastAPI** (v2.1.0) with the following middleware
stack:

| Middleware              | Purpose                                |
| ----------------------- | -------------------------------------- |
| `TrustedHostMiddleware` | Reject requests from untrusted hosts   |
| `CORSMiddleware`        | Cross-Origin Resource Sharing control  |
| `slowapi` rate limiter  | Request rate limiting per IP           |
| Security headers        | HSTS, X-Content-Type-Options, etc.     |
| Upload size validation  | Reject oversized uploads               |
| Request tracing         | Attach correlation IDs for diagnostics |

CORS is configured with restricted origins (from `get_cors_origins()`)
and explicit allowed headers (`Content-Type`, `Authorization`,
`X-API-Key`). Wildcard headers are never used.

### 23.1.2 Database

An SQLite database (`golf_modeling_suite.db`) stores user accounts, API
keys, and usage tracking data. The database is initialised on startup
via `init_db()`.

### 23.1.3 Dependency Injection

Services are stored in `app.state` and injected into route handlers via
FastAPI's `Depends()` mechanism:

| Dependency          | Provider Function          |
| ------------------- | -------------------------- |
| `SimulationService` | `get_simulation_service()` |
| `AnalysisService`   | `get_analysis_service()`   |
| `EngineManager`     | `get_engine_manager()`     |
| `TaskManager`       | `get_task_manager()`       |
| Logger              | `get_logger()`             |

### 23.1.4 Task Manager

Background simulation tasks are tracked by `TaskManager`, a thread-safe
dictionary with automatic TTL-based cleanup:

| Setting       | Value | Description                                  |
| ------------- | ----- | -------------------------------------------- |
| `TTL_SECONDS` | 3600  | Tasks expire after 1 hour                    |
| `MAX_TASKS`   | 1000  | Maximum stored tasks; LRU eviction on excess |

### 23.1.5 Error Handling

Errors are returned as structured JSON with HTTP status codes.
Application-specific error codes follow the pattern `GMS-XXX-YYY`.
All requests are tagged with a correlation ID via the `RequestTracer`
middleware for end-to-end tracing.

### 23.1.6 Environment Configuration

| Environment Variable  | Description                                 |
| --------------------- | ------------------------------------------- |
| `ENVIRONMENT`         | `"development"` or `"production"`           |
| `GOLF_API_SECRET_KEY` | JWT signing secret (required in production) |
| `SECRET_KEY`          | Alias for the above                         |

In development mode, auto-reload is enabled. In production, the server
refuses to start if no secret key is configured.

---

## 23.2 Core Endpoints

### 23.2.1 Health and Information

#### `GET /`

Root endpoint returning API metadata.

**Response:**

```json
{
  "message": "Golf Modeling Suite API",
  "version": "1.0.0",
  "docs": "/docs",
  "status": "running"
}
```

#### `GET /health`

Health check endpoint returning server status and engine availability.

**Response:**

```json
{
  "status": "healthy",
  "engines_available": 3,
  "timestamp": "2026-02-05T12:00:00Z"
}
```

### 23.2.2 Engine Management

#### `GET /engines`

List all physics engines with their status and capabilities.

**Response model:** `EngineListResponse`

```json
{
  "engines": [
    {
      "name": "MUJOCO",
      "available": true,
      "loaded": true,
      "version": null,
      "capabilities": ["physics", "contacts", "muscles", "tendons"],
      "engine_type": "MUJOCO",
      "status": "loaded",
      "is_available": true,
      "description": "MUJOCO physics engine"
    }
  ],
  "mode": "local"
}
```

Each engine reports its capabilities:

| Engine    | Capabilities                        |
| --------- | ----------------------------------- |
| MUJOCO    | physics, contacts, muscles, tendons |
| DRAKE     | physics, optimization, control      |
| PINOCCHIO | kinematics, dynamics, collision     |
| OPENSIM   | musculoskeletal, biomechanics       |
| MYOSIM    | muscle, tendon, control             |
| MATLAB_2D | 2d-simulation, simscape             |
| MATLAB_3D | 3d-simulation, simscape             |
| PENDULUM  | pendulum, educational               |

#### `POST /engines/{engine_type}/load`

Load a specific engine with an optional model file path.

**Parameters:**

- `engine_type` (path) -- Engine type string (e.g., `"mujoco"`)
- `model_path` (query, optional) -- Path to a model file

Model paths undergo security validation to prevent path traversal
attacks.

### 23.2.3 Simulation

#### `POST /simulate`

Execute a synchronous simulation.

**Request model:** `SimulationRequest`

| Field             | Type        | Required | Description                   |
| ----------------- | ----------- | -------- | ----------------------------- | ---------------------------------- |
| `engine_type`     | `str`       | Yes      | Physics engine to use         |
| `model_path`      | `str        | None`    | No                            | Path to model file                 |
| `duration`        | `float`     | No (1.0) | Simulation duration (seconds) |
| `timestep`        | `float      | None`    | No                            | Simulation time step               |
| `initial_state`   | `dict       | None`    | No                            | Initial joint positions/velocities |
| `control_inputs`  | `list[dict] | None`    | No                            | Control sequence                   |
| `analysis_config` | `dict       | None`    | No                            | Analysis configuration             |

**Response model:** `SimulationResponse`

| Field              | Type       | Description                          |
| ------------------ | ---------- | ------------------------------------ | ----------------------------- |
| `success`          | `bool`     | Whether simulation completed         |
| `duration`         | `float`    | Actual simulation duration           |
| `frames`           | `int`      | Number of simulation frames          |
| `data`             | `dict`     | States, controls, derived quantities |
| `analysis_results` | `dict      | None`                                | Analysis results if requested |
| `export_paths`     | `list[str] | None`                                | Paths to exported files       |

**Error responses:**

| Status | Detail                    | Cause                     |
| ------ | ------------------------- | ------------------------- |
| 400    | Invalid parameters: ...   | Validation error          |
| 504    | Simulation timed out      | Duration exceeded timeout |
| 500    | Simulation failed: ...    | Runtime error             |
| 500    | Internal simulation error | Unexpected exception      |

#### `POST /simulate/async`

Start an asynchronous simulation. Returns immediately with a task ID.

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

#### `GET /simulate/status/{task_id}`

Poll the status of an asynchronous simulation.

**Response:**

```json
{
  "status": "completed",
  "result": { ... }
}
```

Returns 404 if the task ID is not found.

### 23.2.4 WebSocket Simulation Streaming

#### `WS /ws/simulate/{engine_type}`

Real-time bidirectional simulation streaming.

**Client sends:**

```json
{"action": "start", "config": {
    "duration": 3.0,
    "timestep": 0.002,
    "initial_state": { ... },
    "live_analysis": true
}}
```

**Server streams:**

```json
{
  "frame": 42,
  "time": 0.084,
  "state": { ... },
  "analysis": {
    "joint_angles": [...],
    "velocities": [...]
  }
}
```

**Control commands:**

| Action     | Effect                                |
| ---------- | ------------------------------------- |
| `"start"`  | Begin simulation with provided config |
| `"stop"`   | Terminate simulation immediately      |
| `"pause"`  | Pause simulation loop                 |
| `"resume"` | Resume from pause                     |

Frame throttling targets 60 FPS for UI responsiveness:
`frame_skip = max(1, int(steps_per_second / 60))`.

**Completion message:**

```json
{
  "status": "complete",
  "total_frames": 1500,
  "total_time": 3.0
}
```

### 23.2.5 Analysis

#### `POST /analysis`

Run biomechanical analysis on simulation data.

**Request model:** `AnalysisRequest`

| Field           | Type   | Description                              |
| --------------- | ------ | ---------------------------------------- |
| `analysis_type` | `str`  | Type: kinematics, kinetics, energy, etc. |
| `data_source`   | `str`  | Source: simulation, c3d, video           |
| `parameters`    | `dict` | Analysis-specific parameters             |
| `export_format` | `str`  | Output format (default `"json"`)         |

**Response model:** `AnalysisResponse`

| Field            | Type       | Description                        |
| ---------------- | ---------- | ---------------------------------- | ---------------------------------- |
| `analysis_type`  | `str`      | Type of analysis performed         |
| `success`        | `bool`     | Whether analysis completed         |
| `results`        | `dict`     | Analysis results (metrics, values) |
| `visualizations` | `list[str] | None`                              | Generated visualisation file paths |
| `export_path`    | `str       | None`                              | Path to exported results           |

### 23.2.6 Video Pose Estimation

#### `POST /video/pose`

Process a video file for pose estimation.

**Request model:** `VideoAnalysisRequest`

| Field                       | Type    | Default       | Description                  |
| --------------------------- | ------- | ------------- | ---------------------------- | ----------------------- |
| `estimator_type`            | `str`   | `"mediapipe"` | Pose estimator backend       |
| `min_confidence`            | `float` | 0.5           | Minimum detection confidence |
| `enable_temporal_smoothing` | `bool`  | `True`        | Smooth pose over time        |
| `max_frames`                | `int    | None`         | `None`                       | Limit frames to process |
| `export_keypoints`          | `bool`  | `True`        | Include raw keypoints        |
| `export_joint_angles`       | `bool`  | `True`        | Compute joint angles         |

**Response model:** `VideoAnalysisResponse`

| Field                | Type         | Description                       |
| -------------------- | ------------ | --------------------------------- |
| `filename`           | `str`        | Original video filename           |
| `total_frames`       | `int`        | Total frames in video             |
| `valid_frames`       | `int`        | Frames with valid pose detection  |
| `average_confidence` | `float`      | Mean confidence across frames     |
| `quality_metrics`    | `dict`       | Quality assessment metrics        |
| `pose_data`          | `list[dict]` | Per-frame pose estimation results |

### 23.2.7 Data Export

#### `GET /export/{task_id}`

Export results for a completed task.

**Query parameters:**

- `format` -- Export format (`"json"`, `"csv"`, etc.). Must be one of the
  valid export formats defined in `config.VALID_EXPORT_FORMATS`.

**Error responses:**

| Status | Detail               | Cause                        |
| ------ | -------------------- | ---------------------------- |
| 400    | Invalid format '...' | Unsupported export format    |
| 404    | Task not found       | Unknown task ID              |
| 400    | Task not completed   | Task still running or failed |

---

## 23.3 Authentication

The authentication system in `src/api/auth/` implements JWT-based access
control with role-based authorisation and API key support.

### 23.3.1 User Registration

#### `POST /auth/register`

**Rate limit:** 3 requests per hour per IP (protects against account
farming).

**Request model:** `UserCreate`

| Field          | Type  | Description              |
| -------------- | ----- | ------------------------ |
| `email`        | `str` | User email (unique)      |
| `password`     | `str` | Plain-text password      |
| `full_name`    | `str` | Display name             |
| `organization` | `str` | Organisation affiliation |

Passwords are hashed with **bcrypt** (cost factor 12). New users
receive the `FREE` role.

### 23.3.2 User Login

#### `POST /auth/login`

**Rate limit:** 5 requests per minute per IP (protects against brute
force).

**Request model:** `LoginRequest`

| Field      | Type  | Description      |
| ---------- | ----- | ---------------- |
| `email`    | `str` | Registered email |
| `password` | `str` | Account password |

**Response model:** `LoginResponse`

Contains `access_token` and `refresh_token` (both JWT strings).

### 23.3.3 JWT Tokens

The `SecurityManager` class handles token creation and verification.

**Access tokens:**

- Algorithm: HS256
- Default expiry: 30 minutes
- Payload includes `{"sub": user_id, "type": "access", "exp": ...}`

**Refresh tokens:**

- Default expiry: 30 days
- Payload includes `{"type": "refresh"}`

Token verification checks:

1. Signature validity (using the server secret key).
2. Expiration (`exp` claim).
3. Token type matches the expected type (`"access"` or `"refresh"`).

### 23.3.4 API Key Authentication

API keys are generated with a `gms_` prefix:

```
gms_<44-character-url-safe-random-string>
```

Keys are stored as bcrypt hashes (not raw SHA256) for brute-force
resistance. An `AuthCache` provides a 5-minute TTL cache of validated
keys to avoid repeated bcrypt computations.

For fast database lookup, a SHA256 hash of the first 8 characters of the
key is stored as an index. This prefix hash is distinct from the full
bcrypt-hashed key used for actual verification.

### 23.3.5 Role-Based Access Control

Four subscription tiers are defined:

| Role           | Level | Description               |
| -------------- | ----- | ------------------------- |
| `FREE`         | 0     | Basic access              |
| `PROFESSIONAL` | 1     | Extended quotas           |
| `ENTERPRISE`   | 2     | Full access               |
| `ADMIN`        | 3     | Administrative privileges |

The `RoleChecker` class enforces minimum-role requirements on endpoints.
The `UsageTracker` tracks and enforces per-month quotas for API calls,
video analyses, and simulations.

### 23.3.6 Authorised Requests

Protected endpoints require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

or an API key in the `X-API-Key` header:

```
X-API-Key: gms_<key>
```

In local mode (`is_local_mode()` returns `True`), authentication is
optional for most endpoints to support development workflows.

---

## 23.4 Request/Response Models

All request and response models are defined using **Pydantic** `BaseModel`
with `Field()` descriptors for validation and documentation.

### 23.4.1 SimulationRequest

```python
class SimulationRequest(BaseModel):
    engine_type: str = Field(..., description="Physics engine to use")
    model_path: str | None = Field(None, description="Path to model file")
    duration: float = Field(1.0, gt=0, description="Simulation duration")
    timestep: float | None = Field(None, gt=0, description="Time step")
    initial_state: dict | None = Field(None, description="Initial state")
    control_inputs: list[dict] | None = Field(None, description="Controls")
    analysis_config: dict | None = Field(None, description="Analysis config")
```

The `engine_type` field is required (`...` sentinel). The `duration`
and `timestep` fields enforce `gt=0` (greater than zero).

### 23.4.2 SimulationResponse

```python
class SimulationResponse(BaseModel):
    success: bool
    duration: float
    frames: int
    data: dict
    analysis_results: dict | None = None
    export_paths: list[str] | None = None
```

### 23.4.3 AnalysisRequest

```python
class AnalysisRequest(BaseModel):
    analysis_type: str
    data_source: str
    parameters: dict = Field(default_factory=dict)
    export_format: str = "json"
```

### 23.4.4 AnalysisResponse

```python
class AnalysisResponse(BaseModel):
    analysis_type: str
    success: bool
    results: dict
    visualizations: list[str] | None = None
    export_path: str | None = None
```

### 23.4.5 VideoAnalysisRequest and Response

See Section 23.2.6 for field details.

### 23.4.6 TaskStatusResponse

```python
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float | None = None
    result: dict | None = None
    error: str | None = None
```

### 23.4.7 Model Fitting Request

```python
class ModelFittingRequest(BaseModel):
    model_path: str
    data_path: str
    fitting_method: str = "least_squares"
    parameters: dict = Field(default_factory=dict)
```

Used for fitting biomechanical models to experimental data (C3D, etc.).

---
