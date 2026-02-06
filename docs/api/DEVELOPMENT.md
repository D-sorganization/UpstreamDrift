# API Development Guide

> **For Developers and AI Agents**

This guide explains how to develop and troubleshoot the Golf Modeling Suite API.

## Quick Start for Developers

```bash
# 1. Ensure you're in the project root
cd /home/user/Golf_Modeling_Suite

# 2. Start the API server
python start_api_server.py

# 3. Access endpoints
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

## Development Principles

### 1. DRY - Don't Repeat Yourself

All shared utilities live in `src/api/utils/`:

```python
# BAD: Duplicated datetime handling
from datetime import UTC, datetime, timezone
try:
    from datetime import timezone
except ImportError:
    timezone.utc = timezone.utc

# GOOD: Use centralized utilities
from src.api.utils import UTC, utc_now
timestamp = utc_now()
```

### 2. Orthogonality

Components should be independent and replaceable:

```python
# Routes handle HTTP only
@router.post("/simulate")
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    return await simulation_service.run_simulation(request)

# Services handle business logic
class SimulationService:
    async def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        # Business logic here
        pass
```

### 3. Design by Contract

Use contracts for validation:

```python
from src.shared.python.contracts import (
    precondition,
    postcondition,
    require_state,
)

@precondition(lambda request: request.duration > 0, "Duration must be positive")
@postcondition(lambda result: result.success or result.error, "Must have result or error")
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    ...
```

## Diagnostics and Traceability

### Error Codes

Use structured error codes for debugging:

```python
from src.api.utils import ErrorCode, raise_api_error

# Instead of generic errors
raise HTTPException(status_code=500, detail="Engine failed")

# Use structured codes
raise_api_error(
    ErrorCode.ENGINE_LOAD_FAILED,
    message="Failed to load MuJoCo",
    engine_type="mujoco",
    reason="Library not found"
)
```

### Error Code Reference

| Code        | Category   | HTTP | Description              |
| ----------- | ---------- | ---- | ------------------------ |
| GMS-GEN-001 | General    | 500  | Internal server error    |
| GMS-GEN-002 | General    | 400  | Invalid request          |
| GMS-GEN-003 | General    | 429  | Rate limited             |
| GMS-ENG-001 | Engine     | 404  | Engine not found         |
| GMS-ENG-002 | Engine     | 400  | Engine not loaded        |
| GMS-ENG-003 | Engine     | 500  | Engine load failed       |
| GMS-SIM-001 | Simulation | 500  | Simulation failed        |
| GMS-SIM-006 | Simulation | 404  | Task not found           |
| GMS-VID-001 | Video      | 500  | Pipeline not initialized |
| GMS-VID-002 | Video      | 400  | Invalid video format     |
| GMS-AUT-001 | Auth       | 401  | Invalid token            |
| GMS-AUT-002 | Auth       | 401  | Token expired            |

### Request Tracing

Every request gets automatic tracing:

```python
from src.api.utils import get_request_id, traced_log

# In any handler or service
request_id = get_request_id()
traced_log("info", "Processing simulation", engine="mujoco")
# Logs: {"message": "Processing simulation", "engine": "mujoco", "request_id": "req_abc123"}
```

Response headers include:

- `X-Request-ID`: Unique request identifier
- `X-Correlation-ID`: Cross-service correlation
- `X-Response-Time-Ms`: Processing time

### Running Diagnostics

```python
from src.api.diagnostics import APIDiagnostics

# Create diagnostics with app reference
diag = APIDiagnostics(app)

# Run all checks
report = diag.run_all_checks()
print(report["summary"])
# {"total_checks": 7, "passed": 6, "failed": 1, "warnings": 0, "status": "degraded"}

# Individual checks
diag.check_python_environment()
diag.check_static_files()
diag.check_api_routes()
diag.check_cors_config()
diag.check_dependencies()
diag.check_engine_manager()
```

## Adding New Features

### New Route

1. Create `src/api/routes/myroute.py`:

```python
from fastapi import APIRouter
from ..utils import ErrorCode, raise_api_error

router = APIRouter()

_my_service = None

def configure(my_service) -> None:
    global _my_service
    _my_service = my_service

@router.get("/myendpoint")
async def my_endpoint():
    if not _my_service:
        raise_api_error(ErrorCode.SERVICE_UNAVAILABLE)
    return {"status": "ok"}
```

2. Register in `server.py`:

```python
from .routes import myroute as my_routes

# In include_router section
app.include_router(my_routes.router)

# In startup_event
my_routes.configure(my_service)
```

### New Error Code

Add to `src/api/utils/error_codes.py`:

```python
class ErrorCode(str, Enum):
    # ... existing codes ...
    MY_NEW_ERROR = "GMS-MYC-001"

ERROR_METADATA[ErrorCode.MY_NEW_ERROR] = {
    "status_code": 400,
    "message": "Description of error",
    "category": ErrorCategory.GENERAL,
}
```

### New Service

```python
# src/api/services/my_service.py
from src.shared.python.logging_config import get_logger
from src.shared.python.contracts import precondition, postcondition

logger = get_logger(__name__)

class MyService:
    def __init__(self, engine_manager):
        self.engine_manager = engine_manager

    @precondition(lambda self, x: x > 0, "Input must be positive")
    @postcondition(lambda result: result is not None, "Must return result")
    async def process(self, x: int) -> dict:
        logger.info("Processing", extra={"input": x})
        return {"result": x * 2}
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# API tests only
pytest tests/unit/test_api/

# With coverage
pytest tests/ --cov=src/api --cov-report=html
```

### Test Example

```python
import pytest
from fastapi.testclient import TestClient
from src.api.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_error_response_format(client):
    response = client.get("/engines/invalid/load", method="POST")
    assert response.status_code == 400
    error = response.json()["error"]
    assert "code" in error  # GMS-XXX-YYY format
    assert "message" in error
```

## Troubleshooting

### Common Issues

#### Engine Not Loading

```
Error: GMS-ENG-003 - Failed to load physics engine
```

**Solution**:

1. Check if engine is installed: `python -c "import mujoco"`
2. Run diagnostics: `diag.check_engine_manager()`
3. Check engine paths in `EngineManager.engine_paths`

#### API Routes Not Found

```
Error: 404 Not Found
```

**Solution**:

1. Check route registration in `server.py`
2. Verify router prefix matches expected path
3. Run `diag.check_api_routes()` to list all routes

#### Database Errors

```
Error: GMS-SYS-001 - Database operation failed
```

**Solution**:

1. Check `DATABASE_URL` environment variable
2. Ensure database file is writable
3. Run `init_db()` to create tables

### Debug Logging

Enable debug logging:

```python
from src.shared.python.logging_config import setup_logging, LogLevel

setup_logging(level=LogLevel.DEBUG, use_detailed_format=True)
```

Or via environment:

```bash
GOLF_LOG_LEVEL=DEBUG python start_api_server.py
```

## Code Quality

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Type Checking

```bash
mypy src/api/ --ignore-missing-imports
```

### Linting

```bash
ruff check src/api/
ruff format src/api/
```

## File Reference

| File              | Purpose                       |
| ----------------- | ----------------------------- |
| `server.py`       | FastAPI app setup, middleware |
| `config.py`       | Constants, defaults           |
| `database.py`     | SQLAlchemy config             |
| `diagnostics.py`  | Health checks                 |
| `dependencies.py` | DI helpers                    |
| `routes/*.py`     | HTTP handlers                 |
| `services/*.py`   | Business logic                |
| `models/*.py`     | Request/response schemas      |
| `auth/*.py`       | Authentication                |
| `middleware/*.py` | HTTP middleware               |
| `utils/*.py`      | Shared utilities              |

## See Also

- [API Architecture](API_ARCHITECTURE.md) - Full architecture docs
- [Contracts Guide](../development/design_by_contract.md) - DbC patterns
- [Engine Docs](../engines/) - Physics engine integration
