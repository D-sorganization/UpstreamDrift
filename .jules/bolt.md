# Bolt's Journal

## 2025-02-23 - PyQt Signal Recursion
**Learning:** Bi-directional synchronization between QSlider (int) and QDoubleSpinBox (float) creates a recursive signal loop that degrades performance (3x calls) and causes precision loss due to int casting.
**Action:** Always use `widget.blockSignals(True)` before programmatically updating a synchronized widget to break the feedback loop.

## 2025-05-23 - MuJoCo API Overhead
**Learning:** MuJoCo Python bindings (mj_jacBody) may have version-dependent signatures (shaped vs flat arrays). Using try-except in a hot loop to handle this adds significant overhead (~5%), plus allocation overhead.
**Action:** Detect API capability once in `__init__`, pre-allocate buffers, and use a conditional branch instead of try-except in per-frame methods.
