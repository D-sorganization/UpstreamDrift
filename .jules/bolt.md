## 2025-12-17 - Bi-directional Signal Loops in UI
**Learning:** In Qt/PyQt, connecting `valueChanged` signals between two controls (slider & spinbox) bi-directionally creates infinite recursion risks and data precision loss (float -> int -> float) if signals are not blocked during updates.
**Action:** Always use `blockSignals(True)` before programmatically updating a coupled UI control to prevent redundant signal emission and data corruption.
