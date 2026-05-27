# Launchers

The suite provides a unified entry point that gracefully wraps the entire UpstreamDrift ecosystem, with an intelligent router and desktop GUI.

## 1. Primary Entry Point (`launch_golf_suite.py`)

This is the recommended way to launch the application. It provides a simple CLI that delegates to the proper internal module based on your arguments.

**Usage:**

```bash
# Launch web UI (default, recommended for web-capable environments)
python launch_golf_suite.py

# Launch the classic desktop GUI (PyQt6-based UpstreamDriftLauncher)
python launch_golf_suite.py --classic

# Start the local API server only
python launch_golf_suite.py --api-only
```

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
