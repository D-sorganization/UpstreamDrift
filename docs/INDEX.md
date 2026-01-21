# Golf Modeling Suite Documentation Index

Welcome to the Golf Modeling Suite documentation. This index provides quick navigation to all documentation organized by role and topic.

---

## Quick Start by Role

### I'm a New User
1. [Installation Guide](user_guide/installation.md) - Set up your environment
2. [Getting Started](user_guide/getting_started.md) - Run your first simulation
3. [Launchers Guide](user_guide/launchers.md) - Understand available launchers
4. [Engine Selection Guide](engine_selection_guide.md) - Choose the right engine

### I'm a Developer
1. [Development Guide](development/README.md) - Development overview
2. [Architecture](development/architecture.md) - System design
3. [Contributing](development/contributing.md) - How to contribute
4. [Testing Guide](testing-guide.md) - Testing practices
5. [AI Agents](development/AGENTS.md) - Agent guidelines

### I'm an Integrator
1. [Engine APIs](api/engines.md) - Interface specifications
2. [Shared Utilities](api/shared.md) - Common utilities
3. [Engine Capabilities](engine_capabilities.md) - Feature comparison

---

## Documentation Map

```
docs/
├── Getting Started
│   ├── user_guide/           # User documentation
│   ├── installation/         # Installation guides
│   └── troubleshooting/      # Common issues
│
├── Engines
│   ├── engines/              # Engine-specific docs
│   ├── engine_selection_guide.md
│   └── engine_capabilities.md
│
├── Development
│   ├── development/          # Dev guides & PRs
│   ├── api/                  # API reference
│   ├── architecture/         # System design
│   └── testing-guide.md
│
├── Project Management
│   ├── plans/                # Roadmap & plans
│   ├── releases/             # Changelog & versions
│   └── reports/              # Technical reports
│
├── Technical
│   ├── technical/            # Technical docs
│   └── physics/              # Physics modeling
│
└── Internal (for maintainers)
    ├── internal/assessments/ # Quality audits
    └── internal/historical/  # Development history
```

---

## Core Documentation

### User Guide
| Document | Description |
|----------|-------------|
| [User Guide](user_guide/README.md) | Complete user documentation |
| [Installation](user_guide/installation.md) | Environment setup |
| [Getting Started](user_guide/getting_started.md) | First simulation tutorial |
| [Launchers](user_guide/launchers.md) | Available launchers |
| [Configuration](user_guide/configuration.md) | Configuration options |

### Physics Engines
| Engine | Description | Best For |
|--------|-------------|----------|
| [MuJoCo](engines/mujoco.md) | High-performance physics | Biomechanics, muscle models |
| [Drake](engines/drake.md) | Model-based design | Trajectory optimization |
| [Pinocchio](engines/pinocchio.md) | Fast rigid body algorithms | Analytical solutions |
| [OpenSim](engines/opensim.md) | Biomechanical validation | Model validation |
| [MyoSuite](engines/myosuite.md) | Muscle modeling | Musculoskeletal research |
| [Simscape](engines/simscape.md) | MATLAB/Simulink | Control system design |
| [Pendulum](engines/pendulum.md) | Simplified models | Education, fundamentals |

### Development
| Document | Description |
|----------|-------------|
| [Development Guide](development/README.md) | Development overview |
| [Architecture](development/architecture.md) | System design |
| [Contributing](development/contributing.md) | Contribution guidelines |
| [Agent Templates](development/agent_templates/README.md) | AI agent templates |
| [PR Documentation](development/prs/README.md) | Pull request docs |

### API Reference
| Document | Description |
|----------|-------------|
| [API Overview](api/README.md) | API documentation hub |
| [Shared Utilities](api/shared.md) | Common Python utilities |
| [Engine APIs](api/engines.md) | Engine interfaces |

---

## Project Status

### Plans & Roadmap
| Document | Description |
|----------|-------------|
| [Plans Overview](plans/README.md) | Planning hub |
| [Migration Status](plans/migration_status.md) | Codebase migration status |
| [Implementation Roadmap](plans/implementation_roadmap.md) | Future development |
| [Priority Improvements](plans/priority_improvements_jan_2026.md) | Current priorities |

### Reports
| Document | Description |
|----------|-------------|
| [Reports Overview](reports/README.md) | Project reports hub |
| [Performance Analysis](reports/PERFORMANCE_ANALYSIS.md) | Performance metrics |
| [Test Coverage](reports/TEST_COVERAGE_REPORT.md) | Test coverage data |
| [Security Fixes](reports/SECURITY_FIXES_SUMMARY.md) | Security improvements |

### Releases
| Document | Description |
|----------|-------------|
| [Changelog](releases/CHANGELOG.md) | Version history |
| [PR Summary](releases/PR_SUMMARY.md) | Notable PRs |

---

## Technical Documentation

### Control & Physics
| Document | Description |
|----------|-------------|
| [Technical Overview](technical/README.md) | Technical documentation hub |
| [Control Strategies](technical/control-strategies-summary.md) | Control approaches |
| [Physics Models](physics/README.md) | Physics modeling details |

### System Architecture
| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture/README.md) | Architecture documentation |
| [Engine Loading](architecture/engine_loading_flow.md) | Engine initialization |

---

## Troubleshooting

| Document | Description |
|----------|-------------|
| [Troubleshooting Hub](troubleshooting/README.md) | Common issues |
| [Installation Issues](troubleshooting/installation.md) | Setup problems |
| [Engine Issues](troubleshooting/engines.md) | Engine-specific issues |

---

## Internal Documentation

> Note: Internal documentation is primarily for project maintainers.

| Document | Description |
|----------|-------------|
| [Internal Overview](internal/README.md) | Internal docs hub |
| [Assessments](internal/assessments/README.md) | Quality audits |
| [Historical](internal/historical/README.md) | Development history |

---

## External Resources

- [GitHub Repository](https://github.com/D-sorganization/Golf_Modeling_Suite)
- [Issue Tracker](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
- [MuJoCo Documentation](https://mujoco.org/)
- [Drake Documentation](https://drake.mit.edu/)
- [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)
- [MyoSuite Documentation](https://github.com/MyoHub/myosuite)
- [OpenSim Documentation](https://opensim.stanford.edu/)
