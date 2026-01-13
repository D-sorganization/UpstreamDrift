# Golf Modeling Suite - System Architecture

## Overview

The Golf Modeling Suite is a professional-grade biomechanical analysis and physics simulation platform designed for golf performance analysis. It provides a unified interface over multiple independent physics backends, enabling researchers and developers to compare simulation results across different engines.

**Key Design Principles:**

- **Engine Agnosticism**: Core logic is independent of any specific physics engine
- **Protocol-Based Interfaces**: All engines implement a common `PhysicsEngine` protocol
- **Layered Architecture**: Clear separation between UI, business logic, and physics computation
- **Scientific Rigor**: Validated coefficients from peer-reviewed research

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Golf Suite  │  │ Shot Tracer │  │ Drake/MuJoCo        │ │
│  │ Launcher    │  │ GUI         │  │ Dashboards          │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      ORCHESTRATION LAYER                     │
│  ┌─────────────────────┐  ┌────────────────────────────────┐│
│  │ Engine Manager      │  │ Video Pose Pipeline            ││
│  │ - Engine discovery  │  │ - MediaPipe integration        ││
│  │ - Lifecycle mgmt    │  │ - Pose estimation              ││
│  │ - State tracking    │  │ - Temporal smoothing           ││
│  └─────────────────────┘  └────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      PHYSICS LAYER                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │ MuJoCo   │ │ Drake    │ │Pinocchio │ │ Ball Flight     │ │
│  │ Engine   │ │ Engine   │ │ Engine   │ │ Models (7)      │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      SHARED SERVICES LAYER                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ Ball Flight │ │ Impact      │ │ Biomechanics           ││
│  │ Physics     │ │ Model       │ │ Analysis               ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      DATA LAYER                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ C3D Files   │ │ URDF/MJCF   │ │ Kaggle Trajectory      ││
│  │ Processing  │ │ Models      │ │ Validation Data        ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Golf_Modeling_Suite/
├── api/                          # REST API (FastAPI)
│   ├── auth/                     # Authentication & authorization
│   │   ├── security.py           # JWT, password hashing
│   │   ├── models.py             # User, Role models
│   │   └── dependencies.py       # Auth middleware
│   ├── models/                   # Request/Response models
│   ├── routes/                   # API endpoints
│   ├── services/                 # Business logic
│   └── server.py                 # Main API application
│
├── config/                       # Configuration files
│   └── interim_config.yaml       # Server/auth/quota settings
│
├── data/                         # Validation datasets
│   └── golf_trajectory.csv       # Kaggle launch monitor data (832 shots)
│
├── docs/                         # Documentation
│   ├── assessments/              # Quality assessments
│   ├── development/              # Developer guides
│   ├── physics/                  # Scientific documentation
│   └── references/               # Academic references
│
├── engines/                      # Physics engine implementations
│   └── physics_engines/
│       ├── mujoco_humanoid_golf/ # MuJoCo-based golfer model
│       ├── drake_golf/           # Drake-based simulation
│       └── pinocchio_golf/       # Pinocchio rigid body dynamics
│
├── launchers/                    # GUI applications
│   ├── golf_suite_launcher.py    # Main entry point
│   ├── shot_tracer.py            # Ball flight visualization
│   ├── drake_dashboard.py        # Drake-specific UI
│   └── mujoco_dashboard.py       # MuJoCo-specific UI
│
├── shared/                       # Shared libraries
│   ├── models/                   # URDF/MJCF model files
│   └── python/                   # Python modules
│       ├── ball_flight_physics.py    # Primary ball flight model
│       ├── flight_models.py          # Multi-model framework (7 models)
│       ├── impact_model.py           # Club-ball impact physics
│       ├── engine_manager.py         # Engine orchestration
│       ├── engine_registry.py        # Engine discovery
│       ├── video_pose_pipeline.py    # Video analysis
│       ├── kaggle_validation.py      # Model validation
│       └── validation_data.py        # TrackMan reference data
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests (92+ files)
│   ├── integration/              # Integration tests
│   ├── physics_validation/       # Scientific validation
│   └── fixtures/                 # Test data
│
└── tools/                        # Development utilities
    ├── urdf_generator/           # Robot model generation
    └── c3d_processor/            # Motion capture processing
```

## Core Abstractions

### PhysicsEngine Protocol

All physics engines implement this protocol for consistent interaction:

```python
from typing import Protocol

class PhysicsEngine(Protocol):
    """Protocol that all physics engines must implement."""

    def load_model(self, model_path: str) -> None:
        """Load a robot model (URDF/MJCF)."""
        ...

    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state."""
        ...

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """Set joint positions."""
        ...

    def apply_torque(self, joint_name: str, torque: float) -> None:
        """Apply torque to a joint."""
        ...
```

### BallFlightModel Protocol

Ball flight models implement this protocol for trajectory simulation:

```python
class BallFlightModel(ABC):
    """Abstract base class for ball flight models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        ...

    @abstractmethod
    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
    ) -> FlightResult:
        """Simulate trajectory from launch conditions."""
        ...
```

### Available Ball Flight Models

| Model             | Source                     | Physics Approach               |
| ----------------- | -------------------------- | ------------------------------ |
| Waterloo/Penner   | McPhee et al.              | Quadratic Cd/Cl vs spin ratio  |
| MacDonald-Hanzely | Am. J. Phys. 1991          | ODE with spin decay            |
| Nathan            | Prof. Nathan, UIUC         | Reynolds-dependent drag        |
| Ballantyne        | sb362/golf-ball-simulator  | Constant Cd physics            |
| JCole             | jcole/golf-shot-simulation | TrackMan-validated             |
| Rospie-DL         | asrospie/golf-flight-model | Physics + Deep Learning        |
| Charry-L3         | angelocharry/Projet-L3     | French physics with spin decay |

## Data Flow

### Ball Flight Simulation

```
User Input (GUI/API)
        │
        ▼
┌─────────────────────┐
│ UnifiedLaunchConditions │
│ - ball_speed        │
│ - launch_angle      │
│ - spin_rate         │
│ - wind (optional)   │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ FlightModelRegistry │
│ - Select model(s)   │
│ - Compare results   │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ BallFlightModel     │
│ - ODE integration   │
│ - Drag/Magnus calc  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ FlightResult        │
│ - trajectory[]      │
│ - carry_distance    │
│ - max_height        │
│ - landing_angle     │
└─────────────────────┘
        │
        ▼
Visualization / Export
```

### Video Analysis Pipeline

```
Video Upload
     │
     ▼
┌─────────────────────┐
│ VideoPosePipeline   │
│ - Frame extraction  │
│ - MediaPipe inference│
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Pose Results        │
│ - keypoints[]       │
│ - joint_angles{}    │
│ - confidence        │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Biomechanics Analysis│
│ - Swing phases      │
│ - Velocity curves   │
│ - Energy transfer   │
└─────────────────────┘
```

## Security Architecture

### Authentication Flow

1. **JWT-based authentication** for API access
2. **Environment variable** configuration for secrets (`GOLF_API_SECRET_KEY`)
3. **bcrypt** password hashing
4. **Rate limiting** via slowapi
5. **Path validation** to prevent traversal attacks

### Security Boundaries

```
┌─────────────────────────────────────────────────┐
│                 Public Internet                  │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            Rate Limiter (slowapi)               │
│            CORS Policy (restricted)             │
│            Trusted Host Middleware              │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│               JWT Authentication                 │
│               API Key Authentication            │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              Path Validation                    │
│              Input Sanitization                 │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              Application Logic                  │
└─────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable              | Required   | Description                            |
| --------------------- | ---------- | -------------------------------------- |
| `GOLF_API_SECRET_KEY` | Yes (prod) | JWT signing secret (min 32 chars)      |
| `GOLF_ADMIN_PASSWORD` | Yes (prod) | Initial admin password                 |
| `DATABASE_URL`        | No         | Database connection string             |
| `API_HOST`            | No         | Server bind address (default: 0.0.0.0) |
| `API_PORT`            | No         | Server port (default: 8000)            |

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`): 92+ test files
   - Individual function/class testing
   - Mocked dependencies
2. **Integration Tests** (`tests/integration/`): 24 test files
   - Cross-component testing
   - Database integration
3. **Physics Validation** (`tests/physics_validation/`): 6 test files
   - Comparison against reference data
   - Coefficient verification
4. **API Tests** (`tests/unit/test_api_server.py`): 15 tests
   - Endpoint coverage
   - Security validation
   - Error handling

### Validation Data Sources

- **Kaggle Dataset**: 832 shots from Garmin R50 launch monitor
- **PGA Tour TrackMan**: Aggregate statistics for benchmarking
- **Academic Papers**: MacDonald & Hanzely (1991), Penner (2003)

## How to Add a New Engine

1. Create engine directory: `engines/physics_engines/new_engine/`
2. Implement `PhysicsEngine` protocol
3. Register in `shared/python/engine_registry.py`
4. Add to `EngineManager` discovery
5. Create dashboard in `launchers/` (optional)
6. Add tests in `tests/unit/`

## How to Add a New Ball Flight Model

1. Create class inheriting from `BallFlightModel`
2. Add to `FlightModelType` enum in `flight_models.py`
3. Register in `FlightModelRegistry._models`
4. Add unit tests validating against known data
5. Document physics equations in model docstring

## References

See `docs/references/` for academic sources and `docs/physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md` for detailed physics documentation.
