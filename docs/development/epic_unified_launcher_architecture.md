# Epic: Unified Agentic Launcher Architecture

## Goal
Transition the UpstreamDrift launcher from a "collection of disjoint programs" (spawning separate sub-processes) to a single unified application. This unified application will be capable of hosting individual biomechanical engines (like MuJoCo) as docked tabs or pop-up windows within the same Qt event loop. Crucially, all loaded engines and tools will share a single, central Sidekick AI assistant session.

## Background
Currently, clicking a tile in the launcher spawns a new process via `subprocess.Popen`. This disjoint architecture has several critical drawbacks:
1.  **Orphaned Agent Context**: Each engine spawns its own isolated, deprecated chat interface, ignoring the primary launcher's Sidekick session context.
2.  **Resource Overhead**: Multiple heavy Qt loops and physics engines load duplicate resources.
3.  **Fragmented Workflows**: Cross-engine analysis is impossible because engines do not share memory or a unified workspace state.

## Subtasks

### 1. Resolve MuJoCo Engine Startup Crash
- **Task**: Investigate and fix the startup crash occurring when the MuJoCo engine is launched.
- **Details**: Add comprehensive diagnostics and error handling to the startup routine to prevent silent crashes and ensure stable engine initialization.

### 2. Fix Ollama Sidekick Connection
- **Task**: Restore the local LLM connection for the Sidekick agent.
- **Details**: The Sidekick attempts to initialize `llama3.1:8b`, which is not present in the local Ollama registry. We must either pull the correct model or dynamically fallback to an available model (e.g., `qwen3.5:cloud`).

### 3. Refactor Launcher Process Architecture (The Core Migration)
- **Task**: Deprecate `launcher_process_manager.py`'s `subprocess.Popen` strategy.
- **Details**: 
    - Migrate engine entry points to expose a `get_main_widget()` or `get_dockable_ui()` method instead of a `main()` blocking execution loop.
    - Implement a central `WorkspaceWindow` in the launcher capable of dynamically adding these widgets as DockWidgets or MDIArea sub-windows.

### 4. Implement Unified Sidekick Session Context
- **Task**: Pipe the central Sidekick instance into the newly docked engine windows.
- **Details**:
    - Remove the old/deprecated chat UI codebase isolated within the MuJoCo engine.
    - Ensure the unified launcher's `ThemeManager` and `SidekickStore` singletons are passed down into the engine widgets.
    - Add Sidekick tooling capabilities allowing the agent to read state from any active docked engine.

## Execution Status
- [ ] Investigate MuJoCo Crash
- [ ] Fix Ollama Connection
- [ ] Create Unified `WorkspaceWindow` Shell
- [ ] Refactor MuJoCo entry point to be dockable
- [ ] Pipe Shared Sidekick
