# Bolt's Journal

## 2025-02-23 - PyQt Signal Recursion
**Learning:** Bi-directional synchronization between QSlider (int) and QDoubleSpinBox (float) creates a recursive signal loop that degrades performance (3x calls) and causes precision loss due to int casting.
**Action:** Always use `widget.blockSignals(True)` before programmatically updating a synchronized widget to break the feedback loop.

## 2025-02-23 - MuJoCo Python Bindings API overhead
**Learning:** The `mj_jacBody` function has different signatures in different MuJoCo versions. Using `try-except` to detect this on every frame adds significant overhead (~30% in tight loops) due to exception handling and allocation. Checking the API once at initialization and pre-allocating buffers (where safe) provides a measurable speedup.
**Action:** Detect MuJoCo API capabilities in `__init__` and use the optimal path without runtime checks. Pre-allocate NumPy arrays for MuJoCo C-API calls when thread safety allows.
