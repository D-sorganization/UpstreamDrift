# Golf Modeling Suite Documentation

> **January 2026** | Local-First API Architecture

Welcome to the Golf Modeling Suite - a professional biomechanical analysis and physics simulation platform.

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Quick Start](#quick-start) |
| Understand the API | [API Architecture](api/API_ARCHITECTURE.md) |
| Develop new features | [Development Guide](api/DEVELOPMENT.md) |
| Choose a physics engine | [Engine Selection Guide](engine_selection_guide.md) |
| Troubleshoot issues | [Troubleshooting](troubleshooting/) |

---

## Quick Start

### 1. Start the API Server

```bash
cd /home/user/Golf_Modeling_Suite
python start_api_server.py
```

### 2. Access the API

- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Run a Simulation

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"engine_type": "mujoco", "duration": 1.0}'
```

---

## Documentation Structure

```
docs/
├── README.md              ← You are here
│
├── api/                   # API Reference
│   ├── API_ARCHITECTURE.md   # Complete API architecture
│   ├── DEVELOPMENT.md        # Developer guide
│   ├── engines.md            # Engine APIs
│   └── shared.md             # Shared utilities
│
├── user_guide/            # End User Documentation
│   ├── installation.md       # Setup instructions
│   ├── getting_started.md    # First steps
│   └── launchers.md          # GUI launchers
│
├── engines/               # Physics Engine Docs
│   ├── mujoco.md            # MuJoCo integration
│   ├── drake.md             # Drake integration
│   ├── pinocchio.md         # Pinocchio integration
│   ├── opensim.md           # OpenSim integration
│   └── simscape.md          # MATLAB Simscape
│
├── development/           # Developer Resources
│   ├── architecture.md      # System design
│   ├── design_by_contract.md # DbC patterns
│   ├── contributing.md      # Contribution guide
│   └── agent_templates/     # AI agent templates
│
├── architecture/          # Technical Architecture
│   ├── system_overview.md
│   ├── engine_loading_flow.md
│   └── data_pipeline.md
│
├── troubleshooting/       # Problem Solving
│   └── (troubleshooting guides)
│
└── archive/               # Historical Documentation
    ├── assessments_jan2026/
    ├── phase_plans/
    └── historical/
```

---

## Core Concepts

### Local-First API

The Golf Modeling Suite uses a **local-first architecture**:

- **No cloud required** for local development
- **Optional cloud mode** for production scaling
- **Same API** whether local or cloud

### Multi-Engine Support

Choose from 6+ physics engines:

| Engine | Best For |
|--------|----------|
| **MuJoCo** | Full musculoskeletal simulation |
| **Drake** | Trajectory optimization, control |
| **Pinocchio** | Fast rigid body dynamics |
| **OpenSim** | Biomechanical validation |
| **MyoSuite** | 290-muscle body models |
| **MATLAB** | Simscape Multibody models |

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

| Route | Purpose |
|-------|---------|
| `GET /health` | System health check |
| `GET /engines` | List available engines |
| `POST /engines/{type}/load` | Load an engine |
| `POST /simulate` | Run simulation |
| `POST /analyze/biomechanics` | Biomechanical analysis |
| `POST /analyze/video` | Video pose analysis |
| `GET /export/{task_id}` | Export results |

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
    "details": {"engine": "drake"}
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

| File | Purpose |
|------|---------|
| `src/api/server.py` | FastAPI application |
| `src/api/utils/` | Shared utilities |
| `src/shared/python/contracts.py` | DbC decorators |
| `src/shared/python/engine_manager.py` | Engine orchestration |

### Running Tests

```bash
pytest tests/
pytest tests/unit/test_api/ --cov=src/api
```

---

## For AI Agents

See [AGENTS.md](../AGENTS.md) in the project root for:
- Agent coding guidelines
- Important files reference
- Testing requirements
- PR workflow

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
- [Engine Capabilities](engine_capabilities.md) - Feature comparison

### [Development](development/README.md)
- [Architecture](development/architecture.md) - System design
- [Contributing](development/contributing.md) - Contribution guide
- [Design by Contract](development/design_by_contract.md) - DbC patterns
- [AI Agents](development/AGENTS.md) - Agent guidelines

### [Technical](technical/README.md)
- [Control Strategies](technical/control-strategies-summary.md)
- Engine reports and assessments

### [Integration Guides]
- [MyoSuite Integration](MYOSUITE_INTEGRATION.md) - 290-muscle models
- [OpenSim Integration](OPENSIM_INTEGRATION.md) - Musculoskeletal

---

## Recent Updates (January 2026)

- **API Architecture Upgrade** - Local-first FastAPI implementation
- **Diagnostics Enhancement** - Structured error codes, request tracing
- **Design by Contract** - Comprehensive contract infrastructure
- **Documentation Reorganization** - Archived old docs, new clear structure

---

## Archived Documentation

Historical assessments, phase plans, and old implementation reports have been moved to [archive/](archive/).

---

## Getting Help

- **API Docs**: http://localhost:8000/docs
- **GitHub Issues**: Report bugs and request features
- **Troubleshooting**: See [troubleshooting/](troubleshooting/)

---

## License

MIT License - See [LICENSE](../LICENSE)
