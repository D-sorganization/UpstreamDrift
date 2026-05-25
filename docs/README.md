# UpstreamDrift Documentation

Welcome to UpstreamDrift — a unified platform for golf swing analysis across
multiple physics engines and biomechanical modeling approaches.

## Quick Navigation

| I want to...            | Go to...                                            |
| ----------------------- | --------------------------------------------------- |
| Get started quickly     | [Quick Start](#quick-start)                         |
| Understand the API      | [API Architecture](api/API_ARCHITECTURE.md)         |
| Develop new features    | [Development Guide](api/DEVELOPMENT.md)             |
| Choose a physics engine | [Engine Selection Guide](engine_selection_guide.md) |
| Troubleshoot issues     | [Troubleshooting](troubleshooting/)                 |

---

## Quick Start

The recommended entry point is the **web UI**:

```bash
python launch_golf_suite.py
```

This starts the local API server (port 8000) and opens the React UI in your default browser.

### Other entry points

| Command | What it launches |
|---------|------------------|
| `python launch_golf_suite.py` | Web UI (recommended) |
| `python launch_golf_suite.py --classic` | Classic PyQt6 desktop launcher |
| `python launch_golf_suite.py --api-only` | API server without auto-opening a UI |
| `python launch_golf_suite.py --engine <name>` | Legacy direct engine launch |
| Pose Studio standalone | `python -m src.tools.pose_studio` |

### Run a Simulation via API

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"engine_type": "mujoco", "duration": 1.0}'
```

The classic PyQt6 launcher remains supported as a fallback and for users who prefer a desktop window.
---

## Documentation Structure

`docs/` contains the full documentation tree. The most commonly used
subdirectories are listed below; many additional topic-specific directories
(`adr/`, `assessments/`, `motion_pipeline/`, `governance/`, etc.) live
alongside them.

```
docs/
├── README.md              ← You are here
│
├── api/                   # API reference
│   ├── API_ARCHITECTURE.md   # Complete API architecture
│   ├── DEVELOPMENT.md        # Developer guide
│   ├── engines.md            # Engine APIs
│   └── shared.md             # Shared utilities
│
├── user_guide/            # End-user documentation
│   ├── installation.md
│   ├── getting_started.md
│   └── launchers.md
│
├── engines/               # Physics engine docs
│   ├── mujoco.md
│   ├── drake.md
│   ├── pinocchio.md
│   ├── opensim.md
│   └── simscape.md
│
├── development/           # Developer resources
│   ├── architecture.md
│   ├── design_by_contract.md
│   ├── contributing.md
│   └── agent_templates/
│
├── architecture/          # Technical architecture
│   ├── system_overview.md
│   ├── engine_loading_flow.md
│   └── data_pipeline.md
│
├── troubleshooting/       # Problem solving
│   ├── common-issues.md
│   ├── installation.md
│   └── FAQ.md
│
├── adr/                   # Architecture decision records
├── assessments/           # Project reviews and assessments
├── motion_pipeline/       # Motion capture and tracking pipeline
├── plans/                 # Implementation plans
└── technical/             # Engine reports and control strategies
```

---

## Core Concepts

### Multi-Engine Support

Choose from multiple physics engines:

| Engine        | Best For                         |
| ------------- | -------------------------------- |
| **MuJoCo**    | Full musculoskeletal simulation  |
| **Drake**     | Trajectory optimization, control |
| **Pinocchio** | Fast rigid body dynamics         |
| **OpenSim**   | Biomechanical validation         |
| **MyoSuite**  | 290-muscle body models           |
| **MATLAB**    | Simscape Multibody models        |

See [Engine Selection Guide](engine_selection_guide.md) for details.

### Design Principles

The codebase follows three key principles:

1. **DRY** - Shared utilities in `src/api/utils/`
2. **Orthogonality** - Decoupled, replaceable components
3. **Design by Contract** - Formal validation with contracts

See [Design by Contract Guide](development/design_by_contract.md).

---

## Key Features

### Physics Simulation

- Multi-engine physics with unified interface
- Real-time and batch simulation modes
- Async task support for long simulations

### Video Analysis

- Pose estimation (MediaPipe, OpenPose, MoveNet)
- Swing sequence detection
- Biomechanical analysis

### Diagnostics

- Structured error codes (GMS-XXX-YYY)
- Request tracing (correlation IDs)
- Built-in health checks

### Security

- JWT authentication (cloud mode)
- Rate limiting
- CORS and security headers

---

## API Overview

### Endpoints

| Route                        | Purpose                |
| ---------------------------- | ---------------------- |
| `GET /health`                | System health check    |
| `GET /engines`               | List available engines |
| `POST /engines/{type}/load`  | Load an engine         |
| `POST /simulate`             | Run simulation         |
| `POST /analyze/biomechanics` | Biomechanical analysis |
| `POST /analyze/video`        | Video pose analysis    |
| `GET /export/{task_id}`      | Export results         |

### Error Handling

All errors include:

- **Error code**: `GMS-ENG-003`
- **Message**: Human-readable description
- **Request ID**: For log correlation
- **Details**: Additional context

Example:

```json
{
  "error": {
    "code": "GMS-ENG-003",
    "message": "Failed to load physics engine",
    "request_id": "req_abc123",
    "details": { "engine": "drake" }
  }
}
```

---

## For Developers

### Getting Started

1. Read [API Architecture](api/API_ARCHITECTURE.md)
2. Follow [Development Guide](api/DEVELOPMENT.md)
3. Understand [Design by Contract](development/design_by_contract.md)

### Key Files

| File                                  | Purpose              |
| ------------------------------------- | -------------------- |
| `src/api/server.py`                   | FastAPI application  |
| `src/api/utils/`                      | Shared utilities     |
| `src/shared/python/contracts.py`      | DbC decorators       |
| `src/shared/python/engine_manager.py` | Engine orchestration |

### Running Tests

```bash
pytest tests/
pytest tests/unit/test_api/ --cov=src/api
```

---

## Development Operations

See [AGENTS.md](../AGENTS.md) in the project root for internal automation and
repository maintenance guidance.

---

## Detailed Documentation

### [User Guide](user_guide/README.md)

- [Installation](user_guide/installation.md) - Setup instructions
- [Getting Started](user_guide/getting_started.md) - First simulation
- [Launchers](user_guide/launchers.md) - GUI options

### [Engines](engines/README.md)

- [MuJoCo](engines/mujoco.md) - High-performance physics
- [Drake](engines/drake.md) - Model-based design
- [Pinocchio](engines/pinocchio.md) - Rigid body algorithms
- [OpenSim](engines/opensim.md) - Biomechanical validation
- [Engine Capabilities](engines/engine_capabilities.md) - Feature comparison

### [Development](development/README.md)

- [Architecture](development/architecture.md) - System design
- [Contributing](development/contributing.md) - Contribution guide
- [Design by Contract](development/design_by_contract.md) - DbC patterns
- [Maintenance Guidance](../AGENTS.md) - Automation and repository operations

### [Technical](technical/README.md)

- [Control Strategies](technical/control-strategies-summary.md)
- Engine reports and assessments

### [Integration Guides]

- [MyoSuite Integration](development/MYOSUITE_INTEGRATION.md) - 290-muscle models
- [OpenSim Integration](development/OPENSIM_INTEGRATION.md) - Musculoskeletal

---

## Getting Help

- **API Docs**: http://localhost:8000/docs (when API server is running)
- **GitHub Issues**: Report bugs and request features
- **Troubleshooting**: See [troubleshooting/](troubleshooting/)

---

## License

MIT License - See [LICENSE](../LICENSE)
