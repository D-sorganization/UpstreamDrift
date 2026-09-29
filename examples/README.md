# Examples

This directory contains executable scripts demonstrating the core functionality of the Golf Modeling Suite.

## Basics

- **01_basic_simulation.py**: Demonstrates initializing the engine manager, loading MuJoCo (gracefully handling absence), running a mock loop, and saving results.
- **02_parameter_sweeps.py**: Demonstrates accessing the Physics Parameter Registry, running a parameter sweep, and exporting analysis reports.

## Running Examples

Ensure your environment is set up (see `docs/development/contributing.md`), then run:

```bash
python examples/01_basic_simulation.py
python examples/02_parameter_sweeps.py
```

Results are saved to `output/`.
