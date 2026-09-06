# UpstreamDrift Documentation

> Last reviewed: 2026-09-03

Welcome to UpstreamDrift - a professional biomechanical analysis and physics simulation platform.

## Quick Navigation

| I want to...             | Go to...                                                                 |
| ------------------------ | ------------------------------------------------------------------------ |
| Get started quickly      | [Quick Start](#quick-start)                                              |
| Understand the API       | [API Architecture](api/API_ARCHITECTURE.md)                              |
| Develop new features     | [Development Guide](api/DEVELOPMENT.md)                                  |
| Add a physics engine     | [Adapter Authoring Guide](adapters/authoring_guide.md)                   |
| Choose a physics engine  | [Engine Selection Guide](engines/engine_selection_guide.md)              |
| Track motion capture     | [Motion Pipeline](motion_pipeline/README.md)                             |
| Review architecture      | [ADRs](adr/)                                                             |
| Read the specification   | [SPEC](../SPEC.md)                                                       |
| Find any document        | [Documentation catalog](index.md)                                        |
| Read a calculation sheet | [Calculation references](index.md#calculation-and-derivation-references) |
| Troubleshoot issues      | [Troubleshooting](troubleshooting/)                                      |

---

## Quick Start

### 1. Launch UpstreamDrift (Web UI)

```bash
cd /home/user/UpstreamDrift
python launch_upstream_drift.py
```

By default this starts the local API server and opens the React web UI.
Other modes (also available via the `upstream-drift` console script):

```bash
python launch_upstream_drift.py --classic        # Classic PyQt6 desktop launcher
python launch_upstream_drift.py --api-only       # API server without auto-opening a UI
python launch_upstream_drift.py --engine <name>  # Legacy direct engine launch
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

<!-- BEGIN GENERATED: docs-structure (scripts/generate_docs_map.py) -->

```text
docs/
|-- README.md   <- you are here (task-oriented hub)
|-- index.md    <- catalog: owner, stability, and full map
|-- adapters/                 # 1 page
|-- adr/                      # 51 pages
|-- ai_implementation/        # 5 pages
|-- api/                      # 7 pages
|-- architecture/             # 19 pages
|-- assessments/              # 294 pages
|-- audit_reports/            # 1 page
|-- audits/                   # 7 pages
|-- bunkershot3d/             # 8 pages
|-- code-quality/             # 1 page
|-- codemap/                  # 3 pages
|-- competitive_analysis/     # 1 page
|-- config/                   # 1 page
|-- conformance/              # no Markdown pages
|-- conventions/              # 3 pages
|-- deployment/               # 1 page
|-- design/                   # 2 pages
|-- development/              # 118 pages
|-- engineering/              # 2 pages
|-- engines/                  # 15 pages
|-- estimation/               # 1 page
|-- examples/                 # no Markdown pages
|-- golf-model/               # 1 page
|-- governance/               # 6 pages
|-- help/                     # 5 pages
|-- historical/               # 4 pages
|-- installation/             # 2 pages
|-- issues/                   # 56 pages
|-- legal/                    # 1 page
|-- model_explorer/           # 1 page
|-- motion_capture/           # 4 pages
|-- motion_matching/          # 2 pages
|-- motion_pipeline/          # 6 pages
|-- motion_training/          # 1 page
|-- operations/               # 17 pages
|-- physics/                  # 3 pages
|-- plans/                    # 17 pages
|-- portfolio/                # 1 page
|-- proposals/                # 1 page
|-- references/               # 1 page
|-- research/                 # 32 pages
|-- review_archive/           # 24 pages
|-- reviews/                  # 3 pages
|-- sg_optimizer/             # 3 pages
|-- sidekick/                 # 3 pages
|-- simulation_backends/      # 4 pages
|-- specs/                    # 4 pages
|-- sphinx/                   # 1 page
|-- status/                   # no Markdown pages
|-- status_quo_analysis/      # 1 page
|-- technical/                # 7 pages
|-- technical_debt/           # 1 page
|-- testing/                  # 8 pages
|-- troubleshooting/          # 6 pages
|-- tutorials/                # 6 pages
|-- ui/                       # 1 page
|-- user_guide/               # 30 pages
|-- ux/                       # 1 page
|-- validation/               # 1 page
|-- workflows/                # 2 pages
```

That is 60 top-level directories. The tree above is
generated by `scripts/generate_docs_map.py` from the real filesystem;
do not edit it by hand. For owners, stability tags, and per-directory
descriptions see [the documentation catalog](index.md).

<!-- END GENERATED: docs-structure -->

---

## Core Concepts

### Architecture Overview

UpstreamDrift uses a **web-first, local-first architecture**:

- **No cloud required** for local development
- **Optional cloud mode** for production scaling
- **Same API** whether local or cloud

### Multi-Engine Support

Choose from 6+ physics engines:

| Engine        | Best For                         |
| ------------- | -------------------------------- |
| **MuJoCo**    | Full musculoskeletal simulation  |
| **Drake**     | Trajectory optimization, control |
| **Pinocchio** | Fast rigid body dynamics         |
| **OpenSim**   | Biomechanical validation         |
| **MyoSuite**  | 290-muscle body models           |
| **MATLAB**    | Simscape Multibody models        |

See [Engine Selection Guide](engines/engine_selection_guide.md) for details.

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

## Recent Updates

- **Web UI (primary)** - `python launch_upstream_drift.py` (or the `upstream-drift` console script) opens the React-based web interface
- **Classic PyQt6 launcher** - available via `python launch_upstream_drift.py --classic`
- **Motion Pipeline** - From video to tracked motion in 5 commands (see `docs/motion_pipeline/`)
- **Multi-engine support** - MuJoCo (default), Drake, Pinocchio, OpenSim, MyoSuite, MATLAB Simscape

---

## Archived Documentation

Historical assessments, phase plans, and old implementation reports live in
[historical/](historical/), [assessments/](assessments/), and
[review_archive/](review_archive/). All three are tagged `archived` in the
[documentation catalog](index.md); see its consolidation decisions section for
why they remain at the top level.

---

## Getting Help

- **API Docs**: http://localhost:8000/docs
- **GitHub Issues**: Report bugs and request features
- **Troubleshooting**: See [troubleshooting/](troubleshooting/)

---

## License

MIT License - See [LICENSE](../LICENSE)
