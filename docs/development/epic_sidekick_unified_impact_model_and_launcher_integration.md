# Epic: Sidekick Unified Integration (Impact & Ball Flight Model & UpstreamDrift Fleet)

## Overview

This epic establishes and hardens the **Unified Sidekick Sidebar** across the **Rate of Closure (Impact & Ball Flight Simulator)** application and all host applications within the **UpstreamDrift Launcher**.

The Sidekick provides a multi-tab engineering companion (AI Chat Assistant, Scientific Calculators, Python REPL, Live Workspace Variable Inspector, Jupyter Notebooks, Data Processor, and File Navigation). This epic ensures:

1. **Impact & Ball Flight Simulator Parity**: The RateOfClosureMainWindow incorporates the full Sidekick dock widget with bidirectional state binding (live club model, impact trajectory, fitting document/report, and landing scatter available inside the REPL and Chat).
2. **Dual-Mode Launch Reliability**: The Impact Explorer functions identically with full Sidekick availability when launched **in isolation (standalone CLI/script)** and when launched **from within the UpstreamDrift Launcher**.
3. **Fleet-Wide Launcher Parity**: All GUI applications registered in launcher_manifest.json reliably attach the Sidekick sidebar with consistent Catppuccin design tokens, robust fallback handling, and zero token-discard regressions.

---

## Architecture & Design Principles

`mermaid
graph TD
    A[UpstreamDrift Launcher] -->|Launches Tile| B[Rate of Closure MainWindow]
    C[Standalone CLI / launch_pyqt6.py] -->|Direct Launch| B
    B -->|Installs Dock Widget| D[UnifiedToolsSidebar]
    D --> E[AI Chat Assistant]
    D --> F[Python REPL & Workspace]
    D --> G[Engineering Calculators]
    D --> H[Jupyter Notebook / Data Explorer]
    B -->|Seeds Simulation State| F
`

1. **Shared-First Contract**: The sidebar implementation resides in src/shared/python/sidekick/ (in Tools, consumed downstream by UpstreamDrift via endor/ud-tools or sibling checkout).
2. **Non-Invasive Embedding**: Host applications connect via shared.python.gui_launcher.tools_sidebar_integration:install_tools_sidebar or explicit mixin, ensuring the host runs even if the Sidekick module is unavailable.
3. **Domain State Seeding**: Host applications publish their authoritative models to sidebar.registry.set_variable(...) so the user can interactively script against real live runtime objects.

---

## Work Breakdown & Tasks

### Phase S1: Architecture & Host Integration Contract

- [ ] Define SidekickHostContract in src/shared/python/gui_launcher/tools_sidebar_integration.py specifying workspace variable exports, project root propagation, and file open handlers.
- [ ] Ensure sidekick_tokens mapping correctly binds to RateOfClosureMainWindow and Catppuccin design token themes.
- [ ] Standardize the Sidekick toggle button in ApplicationToolstrip (RateOfClosureMainWindow) and shortcut (Ctrl+B).

### Phase S2: Rate of Closure (Impact & Ball Flight) Sidekick Integration

- [ ] Add SidekickIntegrationMixin (or install_tools_sidebar wiring) to RateOfClosureMainWindow in src/rate_of_closure/ui/pyqt6/main_window.py.
- [ ] Implement \_seed_impact_workspace():
  - ctive_club: Current ClubSpecification / ClubFittingDocument.
  - simulation_result: Latest 6-DOF swing / interval / flight trajectory.
  - itting_report: Deterministic JSON fitting and coupling analysis report.
  - ariation_dataset: Active Monte Carlo landing scatter dataset.
- [ ] Connect derivation and simulation update events to live workspace variable updates.

### Phase S3: Standalone Launch Hardening

- [ ] Update src/rate_of_closure/launch_pyqt6.py and src/rate_of_closure/ui/pyqt6/launcher.py to ensure launch_rate_pyqt6() starts the Sidekick background API worker (or gracefully falls back to local tools) in standalone mode.
- [ ] Add tests in ests/rate_of_closure/test_sidekick_integration.py verifying standalone sidebar instantiation and variable seeding.

### Phase S4: UpstreamDrift Launcher Embedding Parity

- [ ] Verify launcher_manifest.json tile
      ate_of_closure boots cleanly from the UpstreamDrift PyQt6 launcher with the active host Sidekick sidebar.
- [ ] Test cross-engine context handoff: dragging/opening club models or swing profiles from the Sidekick file explorer into the Rate of Closure workspace.
- [ ] Ensure graceful teardown on window close (cleaning up REPL workers, Morris processes, and API monitors).

### Phase S5: Fleet-Wide Verification & Test Suite

- [ ] Run ests/unit/test_gui_launcher_manifest_targets.py across all tiles in UpstreamDrift.
- [ ] Add contract tests verifying Sidekick initialization across all launcher tiles (mujoco, drake, pinocchio, opensim, myosim, putting_green, c3d_viewer,
      ate_of_closure).
- [ ] Verify accessibility audit (udit_visible_focusable_controls) passes with Sidekick docked.

---

## Associated Repositories

- **Tools** (src/shared/python/sidekick/, src/shared/python/gui_launcher/, src/rate_of_closure/)
- **UpstreamDrift** (src/launchers/, src/config/launcher_manifest.json, src/shared/python/gui_launcher/)
- **AffineDrift** (Documentation and technical monographs)
