# Bolt's Journal

## 2025-02-23 - PyQt Signal Recursion
**Learning:** Bi-directional synchronization between QSlider (int) and QDoubleSpinBox (float) creates a recursive signal loop that degrades performance (3x calls) and causes precision loss due to int casting.
**Action:** Always use `widget.blockSignals(True)` before programmatically updating a synchronized widget to break the feedback loop.
