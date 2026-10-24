# Grip Wrench Plots

Desktop tool (tile `grip_wrench_plots`, run as `python -m src.tools.grip_wrench_plots`) that shows the hand wrenches on the club and, in its later tabs, the closed-loop null-space analysis.

User help: [Grip Wrench Plots](../../../docs/help/grip_wrench_plots.md). Interpretation of the null-space tabs: [Loop Null-Space Exploration](../../../docs/help/loop_nullspace.md) and [Variability Sensitivity Interpretation Guide](../../../docs/help/loop_nullspace_variability.md). Press F1 in the window to open the tile help.

## Layout

| Module                        | Role                                                                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `window.py`, `gui.py`         | Main window; `GripWrenchTabsWidget` keeps the grip view as the first tab.                                                                     |
| `nsx_tabs.py`                 | Invariants, Envelopes, Hand Split and Diagnostics tabs, plus the Variability & Consistency tab (NSX-100).                                     |
| `nsx_view.py`                 | Derives each panel from an `NSXPlotModel` and nothing else.                                                                                   |
| `nsx_draw.py`                 | Matplotlib drawing: solid fill for a determined sign, shading for indeterminate, hatching for unbounded, dashed overlay for a representative. |
| `nsx_job.py`, `nsx_worker.py` | The analysis job, run on a `QThreadPool` worker with progress and cancel.                                                                     |
| `variability_*.py`            | Request controls and rendering for the variability tab.                                                                                       |
| `_embed_adapter.py`           | Lazy embed adapter for the launcher host.                                                                                                     |

## Rules the Tool Follows

- The tool computes nothing and gates nothing itself. Every shown, withheld, band or reason state comes from the NSX plot model, which records the `display_decision` outcome (`src/shared/python/loop_nullspace/output_policy_integration.py`).
- A quantity the model does not mark as shown is listed with its reason and never plotted with a number.
- A representative is drawn only as an overlay on its band, never alone.
- The window runs headless tests with `QT_QPA_PLATFORM=offscreen`; do not launch it from automation.
