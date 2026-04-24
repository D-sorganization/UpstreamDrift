# Launcher Systems Overview

This document catalogues the launcher systems present in the repository, their purpose,
and the recommended entry point for new users and integrations.

## Recommended Primary Entry Point

**`launch_golf_suite.py` (repo root)** is the canonical entry point for the UpstreamDrift suite.
It supports three modes:

```
upstream-drift              # Launch web UI (default, recommended)
upstream-drift --classic    # Launch classic PyQt6 launcher
upstream-drift --api-only   # Launch API server only (for development)
upstream-drift --engine X   # Launch specific engine directly
```

Use `launch_golf_suite.py` for all new integrations and user-facing invocations.

---

## Launcher Inventory

### Root-level

| File                   | Purpose                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `launch_golf_suite.py` | **Canonical entry point.** Unified launcher supporting web UI, classic PyQt6, API-only, and engine-specific modes. |

### `src/launchers/`

| File                                 | Purpose                                                                                                                                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unified_launcher.py`                | Wraps the PyQt6 `GolfLauncher` with async startup, background worker thread, lazy loading of heavy modules (MuJoCo, Drake, etc.), and a splash-screen progress display. Used by `launch_golf_suite.py --classic`. |
| `mujoco_unified_launcher.py`         | Hub launcher for MuJoCo Humanoid Simulation and Analysis Dashboard. Extends `BaseLauncher`.                                                                                                                       |
| `golf_launcher.py`                   | Full-featured PyQt6 `GolfLauncher`. Composes UI, theme, simulation, dialog, and process-manager mixins. Supports both local and Docker modes. Supersedes `golf_suite_launcher.py`.                                |
| `golf_suite_launcher.py`             | **Deprecated.** Legacy local-Python-only golf suite launcher. Use `golf_launcher.py` instead. Retained for backward compatibility.                                                                                |
| `motion_capture_launcher.py`         | Central hub for C3D visualisation and Markerless Pose Estimation. Extends `BaseLauncher`.                                                                                                                         |
| `launcher_constants.py`              | Shared constants (paths, identifiers) used across launcher modules.                                                                                                                                               |
| `launcher_diagnostics.py`            | Runtime diagnostics and health checks invoked at launcher startup.                                                                                                                                                |
| `launcher_dialogs.py`                | Mixin: settings dialog, keyboard-shortcut dialog, toast notifications.                                                                                                                                            |
| `launcher_layout_manager.py`         | Mixin: grid layout and main-window geometry management.                                                                                                                                                           |
| `launcher_model_handlers.py`         | Mixin: model loading and reload event handlers.                                                                                                                                                                   |
| `launcher_model_sources.py`          | Mixin: resolving model file sources (local vs. bundled vs. Docker).                                                                                                                                               |
| `launcher_process_manager.py`        | Mixin: spawning, tracking, and terminating sub-processes.                                                                                                                                                         |
| `launcher_provider_compatibility.py` | Compatibility shims for different physics-engine providers.                                                                                                                                                       |
| `launcher_simulation.py`             | Mixin: simulation launching, engine selection, dependency checking.                                                                                                                                               |
| `launcher_theme.py`                  | Mixin: theme application, theme menus, plot theming.                                                                                                                                                              |
| `launcher_ui_setup.py`               | Mixin: menu bar, top bar, grid area, bottom bar, search, console setup.                                                                                                                                           |

### `src/engines/physics_engines/mujoco/python/`

| File                     | Purpose                                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `humanoid_launcher.py`   | Entry point for the MuJoCo humanoid-golf simulation window (interactive or headless trajectory generation). Lower-level than the unified launchers. |
| `golf_suite_launcher.py` | MuJoCo-specific golf suite launcher (local Python). Prefer `src/launchers/golf_launcher.py` for new work.                                           |

### `src/shared/python/`

| File                                       | Purpose                                                                                   |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `launcher_factory.py`                      | Factory that constructs the appropriate launcher instance based on runtime configuration. |
| `dashboard/launcher.py`                    | Common launcher utilities for the unified dashboard (wraps `UnifiedDashboardWindow`).     |
| `gui_launcher/launcher.py`                 | Core launcher implementation for GUI applications; base for mixins.                       |
| `gui_pkg/launcher_utils.py`                | Low-level GUI utilities shared across launcher modules.                                   |
| `upstream_drift_tools/launcher_factory.py` | Alternative launcher factory used by the `upstream_drift_tools` package.                  |

### `src/api/`

| File                           | Purpose                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| `routes/launcher.py`           | FastAPI routes that expose launcher actions over HTTP (used with `--api-only` mode). |
| `services/launcher_service.py` | Business-logic layer backing the launcher API routes.                                |

### `src/config/`

| File                          | Purpose                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `launcher_manifest_loader.py` | Loads and validates the launcher manifest (list of available engines/tools). |

---

## Consolidation Notes (Issue #3058)

There are 5+ launcher systems because the suite evolved to support multiple deployment
targets (local Python, Docker, web UI, API, engine-specific). The architecture is:

```
launch_golf_suite.py          ← user-facing canonical entry point
  └─ unified_launcher.py      ← async PyQt6 wrapper (--classic mode)
       └─ golf_launcher.py    ← full-featured GolfLauncher (composed from mixins)
  └─ start_api_server.py      ← FastAPI backend (--api-only mode)
```

The `mujoco_unified_launcher.py` and `motion_capture_launcher.py` are domain-specific
hubs that can be invoked directly for specialised workflows, but should be considered
internal utilities rather than primary entry points.

For a future consolidation sprint, consider:

1. Merging `golf_suite_launcher.py` (deprecated) removal.
2. Unifying `launcher_factory.py` and `upstream_drift_tools/launcher_factory.py`.
3. Making `BaseLauncher` the single extension point for new domain launchers.
