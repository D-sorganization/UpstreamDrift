## 2024-05-22 - Simulation Control State
**Learning:** Users confuse "Start" (reset & run) with "Resume" (continue) in physics simulations.
**Action:** Implement explicit "Resume" state in UI logic when `time > 0` to prevent accidental state loss.

## 2024-06-01 - Canvas Simulation Keyboard Accessibility
**Learning:** Canvas-based physics simulations often lack focusable elements, making standard keyboard navigation insufficient. Users expect global shortcuts (Space/R) for playback control.
**Action:** Implement global `keydown` listeners for Start/Pause/Reset shortcuts, while ensuring they don't interfere with form inputs (check `e.target.tagName`).
