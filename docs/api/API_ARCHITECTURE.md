# Golf Modeling Suite - API Architecture

> **Last Updated**: January 2026
> **Status**: Production-Ready Local-First API

## Overview

The Golf Modeling Suite uses a **local-first API architecture** built on FastAPI. This design prioritizes:

1. **Local Development Experience** - No cloud dependencies for local use
2. **Optional Cloud Integration** - Scale to cloud when needed
3. **Multi-Engine Support** - Unified interface across physics engines
4. **Professional Security** - JWT auth, rate limiting, CORS

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Golf Modeling Suite                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌──────────────────────────────────────┐  │
│  │   Web UI    │────▶│         FastAPI Server               │  │
│  │  (React)    │     │        (src/api/server.py)           │  │
│  └─────────────┘     │                                      │  │
│                      │  ┌────────────────────────────────┐  │  │
│  ┌─────────────┐     │  │         Middleware Stack       │  │  │
│  │  Desktop UI │────▶│  │  • Security Headers            │  │  │
│  │   (PyQt6)   │     │  │  • Request Tracing             │  │  │
│  └─────────────┘     │  │  • Upload Validation           │  │  │
│                      │  │  • Rate Limiting               │  │  │
│  ┌─────────────┐     │  └────────────────────────────────┘  │  │
│  │   CLI /     │────▶│                                      │  │
│  │  Scripts    │     │  ┌────────────────────────────────┐  │  │
│  └─────────────┘     │  │         API Routes             │  │  │
│                      │  │  /auth     - Authentication    │  │  │
│                      │  │  /engines  - Engine Management │  │  │
│                      │  │  /simulate - Physics Sim       │  │  │
│                      │  │  /analyze  - Biomechanics      │  │  │
│                      │  │  /video    - Pose Estimation   │  │  │
│                      │  │  /export   - Data Export       │  │  │
│                      │  └────────────────────────────────┘  │  │
│                      │                                      │  │
│                      │  ┌────────────────────────────────┐  │  │
│                      │  │       Services Layer           │  │  │
│                      │  │  • SimulationService           │  │  │
│                      │  │  • AnalysisService             │  │  │
│                      │  └────────────────────────────────┘  │  │
│                      └──────────────────────────────────────┘  │
│                                        │                        │
│                      ┌─────────────────┴──────────────────┐    │
│                      ▼                                    ▼    │
│  ┌──────────────────────────────┐  ┌─────────────────────────┐│
│  │      Engine Manager          │  │     Video Pipeline      ││
│  │  (src/shared/python/)        │  │   (Pose Estimation)     ││
│  │                              │  │                         ││
│  │  ┌────────┐ ┌────────┐      │  │  • MediaPipe            ││
│  │  │ MuJoCo │ │ Drake  │      │  │  • OpenPose             ││
│  │  └────────┘ └────────┘      │  │  • MoveNet              ││
│  │  ┌────────┐ ┌────────┐      │  │                         ││
│  │  │Pinocchio│ │OpenSim│      │  └─────────────────────────┘│
│  │  └────────┘ └────────┘      │                             │
│  │  ┌────────┐ ┌────────┐      │                             │
│  │  │MyoSuite│ │MATLAB  │      │                             │
│  │  └────────┘ └────────┘      │                             │
│  └──────────────────────────────┘                             │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. DRY (Don't Repeat Yourself)

All shared utilities are centralized in `src/api/utils/`:

| Module | Purpose |
|--------|---------|
| `datetime_compat.py` | Timezone-aware datetime utilities |
| `error_codes.py` | Structured error codes (GMS-XXX-YYY) |
| `tracing.py` | Request correlation and tracing |
| `path_validation.py` | Secure file path validation |

### 2. Orthogonality

Components are decoupled and independently replaceable:

- **Routes** only handle HTTP concerns
- **Services** contain business logic
- **Engine Manager** abstracts physics engines
- **Middleware** handles cross-cutting concerns

### 3. Design by Contract

The `src/shared/python/contracts.py` module provides:

```python
from src.shared.python.contracts import (
    precondition,      # Validate inputs
    postcondition,     # Validate outputs
    require_state,     # State requirements
    ContractChecker,   # Class invariants
)

@precondition(lambda self: self._is_initialized, "Engine must be initialized")
@postcondition(lambda result: result is not None, "Must return result")
def simulate(self) -> SimulationResult:
    ...
```

## API Routes

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info and version |
| GET | `/health` | Health check with engine status |
| GET | `/docs` | OpenAPI documentation |
| GET | `/redoc` | ReDoc documentation |

### Authentication (`/auth`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | Obtain JWT tokens |
| POST | `/auth/register` | Create account (cloud mode) |
| POST | `/auth/refresh` | Refresh access token |

**Note**: Authentication is optional in local mode (`GOLF_SUITE_MODE=local`).

### Engine Management (`/engines`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/engines` | List all engines with status |
| POST | `/engines/{type}/load` | Load a specific engine |
| POST | `/engines/{type}/unload` | Unload an engine |

### Simulation (`/simulate`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/simulate` | Run synchronous simulation |
| POST | `/simulate/async` | Start async simulation |
| GET | `/simulate/status/{id}` | Check async task status |

### Analysis (`/analyze`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze/biomechanics` | Run biomechanical analysis |
| POST | `/analyze/video` | Analyze video pose |
| POST | `/analyze/video/async` | Async video analysis |

### Export (`/export`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/export/{task_id}` | Export completed task results |

## Error Handling

All errors use structured error codes for traceability:

```json
{
  "error": {
    "code": "GMS-ENG-003",
    "message": "Failed to load physics engine",
    "details": {
      "engine_type": "drake",
      "reason": "Drake not installed"
    },
    "request_id": "req_a1b2c3d4e5f6",
    "correlation_id": "cor_xyz789"
  }
}
```

### Error Code Categories

| Prefix | Category | Example |
|--------|----------|---------|
| GMS-GEN | General | Internal errors, rate limits |
| GMS-ENG | Engine | Load failures, invalid state |
| GMS-SIM | Simulation | Timeout, invalid params |
| GMS-VID | Video | Invalid format, processing fail |
| GMS-ANL | Analysis | Service not ready |
| GMS-AUT | Auth | Token expired, quota exceeded |
| GMS-VAL | Validation | Missing field, invalid value |
| GMS-RES | Resource | Not found, access denied |
| GMS-SYS | System | Database, configuration |

## Request Tracing

Every request is assigned:

- **Request ID** (`X-Request-ID`): Unique per request
- **Correlation ID** (`X-Correlation-ID`): For cross-service tracing

These IDs appear in:
- Response headers
- Error responses
- Log entries

## Security

### Middleware Stack

1. **TrustedHostMiddleware** - Validates request hosts
2. **CORSMiddleware** - Cross-origin resource sharing
3. **Security Headers** - HSTS, X-Content-Type, etc.
4. **Upload Validation** - 10MB file size limit
5. **Rate Limiting** - Prevents abuse
6. **Request Tracing** - Correlation IDs

### Authentication Modes

| Mode | Auth Required | Use Case |
|------|---------------|----------|
| `local` | No | Local development |
| `cloud` | Yes | Production deployment |

Set via environment: `GOLF_SUITE_MODE=local`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOLF_SUITE_MODE` | `local` | Authentication mode |
| `GOLF_AUTH_DISABLED` | `false` | Force disable auth |
| `GOLF_PORT` | `8000` | Server port |
| `CORS_ORIGINS` | localhost | Allowed CORS origins |
| `GOLF_API_SECRET_KEY` | - | JWT signing key |
| `DATABASE_URL` | sqlite:///.db | Database connection |

### Config Files

- `src/api/config.py` - API constants and defaults
- `src/api/database.py` - Database configuration
- `.env` - Environment overrides (not in git)

## Diagnostics

### Built-in Diagnostics

Access diagnostics at runtime:

```python
from src.api.diagnostics import APIDiagnostics

diag = APIDiagnostics(app)
report = diag.run_all_checks()
# Returns: { summary, checks, recommendations }
```

### Diagnostic Checks

- Python environment
- Static file serving
- UI build status
- API route registration
- CORS configuration
- Dependencies
- Engine manager

## Performance

### Task Manager

Background tasks use TTL-based cleanup:
- `TTL_SECONDS = 3600` (1 hour)
- `MAX_TASKS = 1000`
- LRU eviction when over limit

### Auth Cache

API authentication results cached:
- `TTL_SECONDS = 300` (5 minutes)
- Avoids repeated bcrypt hashing

## File Structure

```
src/api/
├── server.py           # FastAPI app initialization
├── config.py           # Configuration constants
├── database.py         # SQLAlchemy setup
├── diagnostics.py      # Diagnostic utilities
├── dependencies.py     # Dependency injection
│
├── routes/            # API route handlers
│   ├── core.py        # Health, root
│   ├── auth.py        # Authentication
│   ├── engines.py     # Engine management
│   ├── simulation.py  # Physics simulation
│   ├── analysis.py    # Biomechanics
│   ├── video.py       # Pose estimation
│   └── export.py      # Data export
│
├── services/          # Business logic
│   ├── simulation_service.py
│   └── analysis_service.py
│
├── models/            # Pydantic schemas
│   ├── requests.py
│   └── responses.py
│
├── auth/              # Authentication
│   ├── models.py      # User models
│   ├── security.py    # JWT, bcrypt
│   ├── dependencies.py
│   └── middleware.py
│
├── middleware/        # HTTP middleware
│   ├── security_headers.py
│   └── upload_limits.py
│
└── utils/             # Utilities (DRY)
    ├── datetime_compat.py
    ├── error_codes.py
    ├── tracing.py
    └── path_validation.py
```

## Extending the API

### Adding a New Route

1. Create route file in `src/api/routes/`
2. Define router: `router = APIRouter()`
3. Add configure function for dependencies
4. Include in `server.py`: `app.include_router(your_routes.router)`
5. Configure in startup: `your_routes.configure(...)`

### Adding a New Service

1. Create service class in `src/api/services/`
2. Inject dependencies via constructor
3. Use contracts for validation
4. Initialize in `server.py` startup

### Adding a New Engine

1. Implement `PhysicsEngine` protocol
2. Add to `EngineType` enum
3. Create probe class for detection
4. Add loader to `LOADER_MAP`
5. Register in `EngineManager.__init__`

## See Also

- [Quick Start Guide](../user_guide/getting_started.md)
- [Engine Selection Guide](../engine_selection_guide.md)
- [Development Guide](DEVELOPMENT.md)
- [Contracts Guide](../development/design_by_contract.md)
