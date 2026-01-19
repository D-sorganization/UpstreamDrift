# Known Issues - Golf Modeling Suite

Last Updated: 2026-01-19

## Test Infrastructure

### [HIGH] Module Reload Corruption
- **Status**: Open
- **Component**: `tests/integration/test_physics_engines_strict.py:152-210`
- **Description**: Drake and Pinocchio strict tests permanently skipped because `reload()` causes numpy corruption when mocking physics engines
- **Suggested Fix**: Refactor to use separate Python processes for engine isolation
- **GitHub Issue**: #496

### [MEDIUM] Coverage Threshold Reduced
- **Status**: Open
- **Component**: `pyproject.toml`
- **Description**: Main repo coverage reduced from 60% to 10% while recovery is in progress
- **Suggested Fix**: Incrementally add tests and restore threshold
- **GitHub Issue**: #533

### [LOW] Inconsistent Coverage Requirements
- **Status**: Open
- **Component**: Engine-specific pyproject.toml files
- **Description**: Different engines have different coverage requirements (Main: 10%, MuJoCo: 60%, Drake: 90%, Pinocchio: 90%)
- **Suggested Fix**: Standardize coverage requirements across all engines
- **GitHub Issue**: #534

## Missing Test Coverage

### [HIGH] AI/Workflow Engine Untested
- **Status**: Open
- **Component**: `shared/python/ai/`
- **Description**: Critical AI components have no tests:
  - `workflow_engine.py` - NO TESTS
  - `tool_registry.py` - NO TESTS
  - `adapters/anthropic_adapter.py` - NO TESTS
  - `adapters/openai_adapter.py` - NO TESTS
- **Suggested Fix**: Add unit tests for workflow engine and adapters
- **GitHub Issue**: #494

### [HIGH] Injury Analysis Module Untested
- **Status**: Open
- **Component**: `shared/python/injury/`
- **Description**: Safety-critical injury analysis has no tests:
  - `injury_risk.py` - NO TESTS
  - `joint_stress.py` - NO TESTS
  - `spinal_load_analysis.py` - NO TESTS
- **Suggested Fix**: Add unit tests for all injury analysis functions
- **GitHub Issue**: #495

### [MEDIUM] Dashboard Components Untested
- **Status**: Open
- **Component**: `shared/python/dashboard/`
- **Description**: All dashboard widgets have only mocked coverage tests, no functional tests
- **Suggested Fix**: Add integration tests for dashboard components
- **GitHub Issue**: #535

### [MEDIUM] Swing Optimizer Untested
- **Status**: Open
- **Component**: `shared/python/optimization/swing_optimizer.py`
- **Description**: No tests for swing optimization module
- **Suggested Fix**: Add unit tests with known optimization scenarios
- **GitHub Issue**: #536

### [MEDIUM] MediaPipe Estimator Untested
- **Status**: Open
- **Component**: `shared/python/pose/mediapipe_estimator.py`
- **Description**: No tests for MediaPipe pose estimation
- **Suggested Fix**: Add tests with sample images/video
- **GitHub Issue**: #537

## Platform Issues

### [MEDIUM] Windows DLL Initialization Issues
- **Status**: Open
- **Component**: Platform-specific tests
- **Description**: Some tests fail on Windows due to DLL initialization issues
- **Suggested Fix**: Add proper DLL path configuration or skip on Windows
- **GitHub Issue**: #538

### [LOW] Linux Requires Xvfb for GUI Tests
- **Status**: Open
- **Component**: CI configuration
- **Description**: GUI tests on Linux require Xvfb wrapper
- **Suggested Fix**: Document requirement and ensure CI uses xvfb-run
- **GitHub Issue**: #539

## API Security & Operations

### [CRITICAL] API Endpoints Unauthenticated
- **Status**: Open
- **Component**: `api/server.py`, `api/routes/auth.py`
- **Description**: Auth router is not mounted and core endpoints do not require authentication or API keys
- **Suggested Fix**: Include auth router and apply `RequireAuth` dependencies to protected endpoints
- **GitHub Issue**: #543

### [HIGH] Database Initialization Not Invoked
- **Status**: Open
- **Component**: `api/database.py`, `start_api_server.py`
- **Description**: Database initialization exists but is not called during API startup
- **Suggested Fix**: Invoke `init_db()` on startup or before server run
- **GitHub Issue**: #544

### [HIGH] Upload Size Limit Not Enforced
- **Status**: Open
- **Component**: `api/server.py`
- **Description**: Video upload endpoints read full payloads without enforcing MAX_UPLOAD_SIZE_BYTES
- **Suggested Fix**: Enforce size limit via Content-Length checks or streaming guards
- **GitHub Issue**: #545

## CI/CD

### [LOW] Weekly CI Failure Digests Accumulating
- **Status**: Resolved
- **Component**: GitHub Issues
- **Description**: Multiple weekly digest issues (#378, #270, #170, #75) were open
- **Suggested Fix**: Close old digests, fix underlying CI issues
- **GitHub Issue**: #378 (open), #270, #170, #75 (closed)
