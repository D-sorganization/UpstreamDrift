## 2025-02-12 - MuJoCo Python Loop Optimization
**Learning:** Manual iteration over MuJoCo bodies in Python to compute system-wide quantities (like COM velocity) is extremely slow due to Python overhead and repeated C-API calls (`mj_jacBodyCom`).
**Action:** Always check for native MuJoCo functions (like `mj_subtreeVel`) which perform these calculations in C. The speedup can be massive (>50x).

## 2025-12-17 - Bi-directional Signal Loops in UI
**Learning:** In Qt/PyQt, connecting `valueChanged` signals between two controls (slider & spinbox) bi-directionally creates infinite recursion risks and data precision loss (float -> int -> float) if signals are not blocked during updates.
**Action:** Always use `blockSignals(True)` before programmatically updating a coupled UI control to prevent redundant signal emission and data corruption.
