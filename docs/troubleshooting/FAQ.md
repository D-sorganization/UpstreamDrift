# Frequently Asked Questions

Quick answers to the most common questions. For detailed troubleshooting, see
[common-issues.md](common-issues.md).

## Setup and Installation

### Q: Which Python version do I need?

Python 3.11 or newer. Python 3.13 is recommended. Check with:

```bash
python3 --version
```

### Q: Do I need all 5 physics engines installed?

No. You can install only the engines you need. MuJoCo is the recommended
default. The suite degrades gracefully when engines are missing. See the
[Engine Selection Guide](../engine_selection_guide.md) to choose.

### Q: How do I install in development mode?

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

## Engines

### Q: I get `EngineNotAvailableError: Engine 'mujoco' not found`

The engine is not installed in your current environment. Solutions:

1. Check that the engine is installed: `python3 -c "import mujoco; print(mujoco.__version__)"`
2. Verify your virtual environment is active: `which python3`
3. Reinstall: `pip install -e ".[mujoco]"`

### Q: Which engine should I start with?

MuJoCo. It has the easiest installation, best documentation, and covers the
widest range of use cases. See the [Engine Selection Guide](../engine_selection_guide.md).

### Q: Can I run simulations without any physics engine?

Yes, for UI development. Set `GOLF_USE_MOCK_ENGINE=1` to use the mock engine:

```bash
export GOLF_USE_MOCK_ENGINE=1
python3 launch_golf_suite.py
```

## API and WebSocket

### Q: The simulation stream disconnects after ~30 seconds

This is usually a WebSocket timeout. Solutions:

1. Check the server is running: `curl http://localhost:8000/api/health`
2. Increase timeout in client configuration
3. Check that your firewall is not blocking WebSocket upgrades

### Q: Where are the API docs?

When the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

### Q: Pre-commit hooks reject my commit

Run hooks manually to see the full output:

```bash
pre-commit run --all-files
```

Common fixes:

- Formatting: `ruff format src/`
- Lint errors: `ruff check src/ --fix`

### Q: Tests fail locally but pass in CI (or vice versa)

Common causes:

1. Missing optional dependencies: `pip install -e ".[all-engines]"`
2. Stale pytest cache: `pytest --cache-clear`
3. Order-dependent tests: `pytest --randomly-seed=last`
4. Platform differences (e.g., headless Linux needs `xvfb-run pytest`)

### Q: I get `ModuleNotFoundError` when importing project modules

Ensure the package is installed in development mode:

```bash
pip install -e ".[dev]"
```

Check that you are using the correct Python:

```bash
which python3
```

### Q: How do I run only a subset of tests?

```bash
# Unit tests only (fast)
pytest tests/unit -x -q

# Specific test file
pytest tests/unit/test_engine_manager.py -v

# Tests matching a keyword
pytest -k "mujoco" -v
```

## Performance

### Q: Simulation is running slowly

1. Use headless mode for batch runs (no rendering overhead)
2. Reduce visualization frequency if rendering every frame
3. Check that you are not running inside a debugger

### Q: High memory usage (>4GB)

1. Load models on-demand instead of all at startup
2. Clear cached models when switching between simulations
3. Check for large data arrays being kept in memory

## Still Stuck?

1. Search [GitHub Issues](https://github.com/D-sorganization/UpstreamDrift/issues)
2. Read the detailed [Common Issues Guide](common-issues.md)
3. Check [Cross-Engine Deviations](cross_engine_deviations.md) for engine-specific quirks
4. Open a new issue with your Python version, OS, and full error traceback
