## 2024-05-22 - Simulation Control State
**Learning:** Users confuse "Start" (reset & run) with "Resume" (continue) in physics simulations.
**Action:** Implement explicit "Resume" state in UI logic when `time > 0` to prevent accidental state loss.

## 2024-06-01 - Canvas Simulation Keyboard Accessibility
**Learning:** Canvas-based physics simulations often lack focusable elements, making standard keyboard navigation insufficient. Users expect global shortcuts (Space/R) for playback control.
**Action:** Implement global `keydown` listeners for Start/Pause/Reset shortcuts, while ensuring they don't interfere with form inputs (check `e.target.tagName`).

## 2024-06-15 - Shortcut Discoverability in Simulations
**Learning:** Global keyboard shortcuts (like Space/R) are powerful but invisible. Users often miss them unless they read documentation or hover over buttons.
**Action:** Always include a persistent, visible legend for essential shortcuts near the controls, not just in tooltips.

## 2024-10-24 - Legacy Web Visualization Accessibility
**Learning:** Legacy vanilla JS/HTML visualizations in this repo often lack basic accessibility (ARIA labels, focus states) and visual polish (icons) compared to modern frameworks.
**Action:** Systematically upgrade legacy control panels with SVG icons and ARIA attributes to match modern standards without rewriting the underlying engine logic.

## 2024-10-25 - Live Parameter Tuning in Physics Simulations
**Learning:** Users expect immediate feedback when adjusting simulation parameters (mass, length) without restarting the simulation loop. This encourages "playful exploration".
**Action:** Implement `input` listeners on parameter fields to update the simulation model in real-time, while carefully separating state variables (integrator state) from model parameters.

## 2024-05-23 - Standard Icons in PyQt6
**Learning:** PyQt6's `QStyle.StandardPixmap` provides a reliable way to add semantic icons (like Play, Clear, Trash) without managing external assets. This ensures visual consistency with the OS and reduces bundle size.
**Action:** Always check `QStyle.StandardPixmap` before adding custom icon files for generic actions like Launch, Clear, or Save.
