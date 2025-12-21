# Bolt's Journal

## 2025-02-23 - PyQt Signal Recursion
**Learning:** Bi-directional synchronization between QSlider (int) and QDoubleSpinBox (float) creates a recursive signal loop that degrades performance (3x calls) and causes precision loss due to int casting.
**Action:** Always use `widget.blockSignals(True)` before programmatically updating a synchronized widget to break the feedback loop.

## 2025-02-23 - MuJoCo Python Bindings API overhead
**Learning:** The `mj_jacBody` function has different signatures in different MuJoCo versions. Using `try-except` to detect this on every frame adds significant overhead (~30% in tight loops) due to exception handling and allocation. Checking the API once at initialization and pre-allocating buffers (where safe) provides a measurable speedup.
**Action:** Detect MuJoCo API capabilities in `__init__` and use the optimal path without runtime checks. Pre-allocate NumPy arrays for MuJoCo C-API calls when thread safety allows.

## 2025-02-23 - Numpy Flatten Copy Overhead
**Learning:** `np.flatten()` always returns a copy, even for contiguous arrays. In high-frequency spatial algebra operations (like RNEA), copying small 6-vectors adds measurable overhead (~10%). Strided views are faster for element-wise access than copying to contiguous memory.
**Action:** Avoid `flatten()` when input shape is already correct. Use `np.asarray()` to get a view and check shape/strides before flattening.

## 2025-05-21 - Statistical Analysis Redundant Computations
**Learning:** `StatisticalAnalyzer` was re-calculating `rad2deg` and `min`/`max` multiple times per joint (once for ROM, once for stats). Optimizing `compute_summary_stats` to reuse indices and `generate_comprehensive_report` to reuse arrays yielded a ~24% speedup.
**Action:** In data-heavy loops, hoist conversions (like `rad2deg`) out of inner calls and ensure statistical functions return all necessary derived metrics to avoid re-computation.
