# Launchers

UpstreamDrift provides unified entry points that wrap the simulation web UI, PyQt6 desktop Control Tower, Sidekick assistant, and headless batch runner.

## 1. Primary Entry Point (`launch_upstream_drift.py` and `upstream-drift`)

This is the recommended way to launch the application. It provides a simple CLI that routes to the web UI, classic PyQt6 desktop launcher, local API server, or an individual engine.

**Usage:**

```bash
# Launch web UI (default, opens browser on port 8000)
upstream-drift
# or: python launch_upstream_drift.py

# Launch the classic desktop GUI (PyQt6-based UpstreamDriftLauncher)
python launch_upstream_drift.py --classic

# Start the local API server only (custom port, no browser)
python launch_upstream_drift.py --api-only --port 8080 --no-browser

# Launch a specific engine directly
python launch_upstream_drift.py --engine mujoco

# Install desktop application shortcuts
python launch_upstream_drift.py --install-shortcuts
```

Installed console scripts in `pyproject.toml`:

- `upstream-drift`: Same CLI as `python launch_upstream_drift.py`
- `sidekick`: Standalone Sidekick assistant (`sidekick.__main__:main`)
- `ud-sim`: Headless simulation runner (`src.sim_run.cli:main`)
- `ud-sim-mcp`: Stdio MCP server for `sim_run` (`src.sim_run.mcp:main`)

_Note: `launch_golf_suite.py` is retained only as a deprecated compatibility shim for older scripts._

## 2. Desktop GUI (`src/launchers/upstream_drift_launcher.py`)

The modernized desktop launcher acts as the "Control Tower". It manages execution environments and provides a robust, cross-platform interface.

**Features:**

- Native Windows and Docker/WSL runtime modes, persistent across sessions.
- Tabbed workspace architecture for hosting specialized analysis tools.
- Intelligent module probes that check dependency availability without blocking the UI.
- Aesthetic, dynamic tile grid with real-time process monitoring and status chips.

You can launch it directly as a module (e.g. for debugging):

```bash
python -m src.launchers.upstream_drift_launcher
```

## 3. Direct Engine Scripts

For isolated debugging, each engine or tool retains its own entry script within its package structure (e.g., `python -m src.shared.python.pendulum_simulator`).
