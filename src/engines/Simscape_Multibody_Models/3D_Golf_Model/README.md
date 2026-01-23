# Golf Model - Biomechanical Golf Swing Analysis

A comprehensive MATLAB-based biomechanical modeling and simulation system for analyzing golf swing dynamics,
trajectory optimization, and performance metrics.

## Overview

This repository contains tools for:

- **Golf swing biomechanical modeling** using MATLAB/Simulink Simscape Multibody
- **Motion capture data analysis** and visualization
- **Dataset generation** for swing parameter variations
- **Interactive GUI applications** for visualization and analysis
- **Performance tracking** and optimization tools
- **Python-based utilities** for data processing and visualization

## Repository Structure

```text
Golf_Model/
├── matlab/                    # MATLAB source code and models
│   ├── Model/                 # Simscape Multibody models
│   ├── Scripts/               # Analysis and GUI scripts
│   │   ├── Golf_GUI/          # Interactive visualization applications
│   │   ├── Dataset Generator/ # Batch simulation tools
│   │   └── Functions/         # Utility functions
│   ├── Data/                  # Input data files
│   └── tests/                 # MATLAB test suite
├── python/                    # Python utilities
├── docs/                      # Documentation
├── scripts/                   # Automation scripts
├── output/                    # Generated results
└── examples/                  # Example implementations
```

## Quick Start

### Prerequisites

- MATLAB R2023a or later with Simulink and Simscape Multibody
- Python 3.11+ (for Python utilities)
- Git with Git LFS installed

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/D-sorganization/Golf_Model.git
   cd Golf_Model
   git lfs install
   git lfs pull
   ```

2. **Set up MATLAB environment**

   ```matlab
   cd matlab
   setup_matlab_environment
   ```

   This configures Simulink cache directories and MATLAB paths.

3. **Install Python dependencies** (optional)

   ```bash
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks** (for developers)

   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Running the Model

**Run all tests and examples:**

```matlab
cd matlab
run_all
```

**Open the main GUI:**

```matlab
cd matlab/Scripts/Golf_GUI
% Follow specific GUI documentation in the subdirectories
```

## Development Workflow

### Daily Safety Practices

- Commit frequently (every ~30 minutes)
- Use `wip:` prefix for work-in-progress commits
- Create backup branches before major refactors: `git checkout -b backup/before-<description>`
- Run tests before pushing: `cd matlab && run_matlab_tests`

### Code Quality

All code changes are automatically checked with:

- **Ruff** (Python linting and formatting)
- **mypy** (Python type checking)
- **pre-commit hooks** (automated checks before commit)
- **CI/CD pipeline** (GitHub Actions on push/PR)

Run quality checks manually:

```bash
pre-commit run --all-files
```

## Documentation

- **[Development Guidelines](docs/GUARDRAILS_GUIDELINES.md)** - Repository safety and quality standards
- **[MATLAB Setup](matlab/SETUP_INSTRUCTIONS.md)** - Detailed MATLAB environment configuration
- **[MATLAB Quality Controls](docs/MATLAB_QUALITY_CONTROLS.md)** - Code quality tools for MATLAB
- **[Performance Tracking](docs/PERFORMANCE_TRACKING_GUIDE.md)** - Monitoring and optimization

Additional documentation available in the `docs/` directory.

## Key Features

### Interactive GUIs

- **2D Skeleton Plotter** - Real-time visualization of golf swing motion
- **3D Motion Capture Plotter** - Advanced 3D visualization with playback controls
- **Dataset Generator GUI** - Batch parameter sweep and simulation management
- **Code Analysis GUI** - Static analysis and quality metrics for MATLAB code

### Analysis Tools

- Joint angle and angular velocity calculations
- Club head speed and trajectory analysis
- Ground reaction force modeling
- Energy transfer and efficiency metrics
- Parameter sensitivity analysis

### Dataset Generation

- Automated batch simulations
- Parameter space exploration
- Parallel processing support (14-core configuration available)
- Data export in multiple formats

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and commit frequently
3. Run quality checks: `pre-commit run --all-files`
4. Run tests: `cd matlab && run_matlab_tests`
5. Push and create a pull request

See [Development Guidelines](docs/GUARDRAILS_GUIDELINES.md) for detailed contribution standards.

## Testing

**MATLAB tests:**

```matlab
cd matlab
run_matlab_tests
```

**Python tests:**

```bash
pytest
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Support

For issues, questions, or contributions, please open an issue on GitHub.
