# ADR-0001: FastAPI for Local-First API Design

- Status: Accepted
- Date: 2026-02-18
- Decision Makers: Core maintainers
- Related Issues/PRs: #1486

## Context

The golf modeling suite needs a local API server for communication between the UI
frontend and physics engine backends. The server must handle both traditional
request/response patterns (model loading, configuration) and real-time streaming
(simulation state updates, sensor data). Options considered included Flask,
Django, FastAPI, and a raw WebSocket server.

Key constraints:

- Must support real-time simulation data streaming with low latency
- Must provide type-safe request/response handling for complex physics payloads
- Must generate API documentation automatically (multiple engines = large surface area)
- Must run locally with minimal overhead (not a cloud-deployed service)

## Decision

Use FastAPI with Uvicorn for the local API server because:

- Native async/await support for real-time simulation streaming
- Automatic OpenAPI documentation generation
- Pydantic model validation for request/response schemas
- WebSocket support for real-time data streaming
- Excellent performance with minimal overhead

## Alternatives Considered

1. **Flask**: Mature ecosystem but lacks native async support. Would require
   Flask-SocketIO for WebSocket handling and additional libraries for validation.
2. **Django**: Too heavyweight for a local-first application. ORM and admin
   features add unnecessary complexity.
3. **Raw WebSocket server**: Maximum flexibility but requires building request
   routing, validation, and documentation from scratch.

## Consequences

- **Positive**: Auto-generated API docs at `/docs` and `/redoc` reduce documentation burden
- **Positive**: Pydantic models enforce type safety at API boundaries, catching errors early
- **Positive**: Strong middleware ecosystem (CORS, rate limiting, security headers)
- **Positive**: WebSocket support is first-class, no adapter libraries needed
- **Negative**: Requires understanding of async programming patterns
- **Negative**: Some optional dependencies (uvloop) not available on all platforms
- **Follow-ups**: Establish API versioning strategy as engine capabilities grow

## Validation

- API contract tests in `tests/api/` verify endpoint schemas against Pydantic models
- Health endpoint (`/api/health`) confirms server is running in CI
- OpenAPI spec is exported and diffed in CI to detect unintended API changes
