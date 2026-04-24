# Examples

This directory contains executable scripts demonstrating the core functionality of the Golf Modeling Suite.

## Basics

- **01_basic_simulation.py**: Demonstrates initializing the engine manager, loading MuJoCo (gracefully handling absence), running a mock loop, and saving results.
- **02_parameter_sweeps.py**: Demonstrates accessing the Physics Parameter Registry, running a parameter sweep, and exporting analysis reports.

## Running Examples

Ensure your environment is set up (see `docs/development/contributing.md`), then run from the repository root:

```bash
python examples/01_basic_simulation.py
python examples/02_parameter_sweeps.py
python examples/03_injury_risk_tutorial.py
```

Results are saved to `output/`.

**Note:** These examples are designed to run from the repository root. The examples use dynamic path setup to work without requiring the package to be installed in editable mode (`pip install -e .`).
