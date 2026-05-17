# UpstreamDrift Sidekick & UI Modernization Roadmap

## Phase 1: Core Layout & Frameless Interactivity (Completed)
- **Frameless Window Resizing**: Deployed a global `QApplication` event filter to seamlessly capture mouse events and enable robust resizing across all corners/edges, overriding the native minimum size constraints that caused the "bloated screen" behavior.
- **Top Bar Styling**: Normalized CustomTitleBar text coloring and removed excess drop shadowing for a modern flat aesthetic.
- **Sidekick Installation**: Re-architected sidebar installation into `QSplitter` panes to prevent native `QDockWidget` floating issues in frameless scenarios.

## Phase 2: Feature Parity & Tab Enhancements (Completed)
- **Terminal Reliability**: Configured Windows PowerShell fallback to prevent `pwsh` loading errors due to missing PyWinPty paths.
- **Data Explorer & Files Tab**: 
  - Added "Show in OS Explorer" to quickly open files with default programs.
  - Implemented the "Quick Access" label and context menu option to add directories locally.
- **Settings Connections**: 
  - Wired the left sidebar Settings button to properly invoke `_show_preferences`.
  - Refactored Sidekick Tab Settings dialog to trigger as a modal popup.
- **Pop-out System**: Engineered a custom `QDialog`-based detachment for Sidekick to bypass frameless constraints, featuring a dynamic "Re-dock" operation.

## Phase 3: Future UI Polish & Theming (Planned)
- **Fluid Layouts**: Enforce strict minimum/maximum stretch factors across `QSplitter` so elements collapse gracefully on smaller resolutions.
- **Dynamic Color Profiles**: Add accent hover states on active tab headers using generic design tokens.
- **Animation Enhancements**: Deploy `QPropertyAnimation` sweeps when expanding/collapsing Sidekick and the Left Panel for a smoother transition.

## Phase 4: Automated Test Coverage (Planned)
- **Frameless Event Verification**: Use `pytest-qt` simulated mouse movements on `FramelessResizeFilter` to ensure continuous coverage.
- **Pop-out State Management**: Assert parent inheritance swaps and geometry restoration of the detached Sidekick Dialog.
- **Shell Discovery Metrics**: Assert the prioritized discovery tuple ensures `powershell.exe` falls back gracefully across virtual environments.
