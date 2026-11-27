---
issue: 15522
summary: "Exercise the Simulation Backends launcher with headless QTest and stop active physics workers on window close."
---

Real Qt mouse and keyboard events drive ODE rollouts and mass sweeps, checking
physics results, reports, plots and worker teardown. Refs #15522.
