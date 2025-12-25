# Launchers

The suite provides multiple entry points depending on your needs.

## 1. Unified Launcher (`launchers/golf_launcher.py`)
This is the main entry point. It provides a GUI to select and configure simulations across all supported Python engines.

**Features:**
- Engine Selection (MuJoCo, Drake, Pinocchio).
- Model Selection.
- Environment Configuration.

## 2. Suite Launcher (`launchers/golf_suite_launcher.py`)
A legacy or alternative launcher that may offer different configuration options or debugging paths.

## 3. Direct Engine Scripts
For advanced users or debugging, each engine has its own `main.py` or entry script within its directory structure. See [Getting Started](getting_started.md) for details.
