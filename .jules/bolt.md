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

## 2025-05-21 - MuJoCo MjData Allocation Cost
**Learning:** Creating `mujoco.MjData(model)` is an expensive operation as it allocates memory for the entire simulation state. In trajectory analysis loops where perturbed state is needed (e.g., for finite differences), re-allocating `MjData` at each step kills performance.
**Action:** Pre-allocate a scratch `MjData` object in `__init__` and reuse it for temporary calculations (like finite difference perturbations), ensuring proper state resetting or overwriting.

## 2025-05-21 - Python Buffer Reuse Overhead
**Learning:** In the Recursive Newton-Euler Algorithm (RNEA), pre-allocating buffers to avoid allocating ~50 small (6x6) matrices per step reduced GC pressure but did not improve execution speed (measured ~2% slowdown). The overhead of Python function calls and `np.matmul(..., out=...)` argument processing outweighed the cost of small allocations, which NumPy handles very efficiently.
**Action:** Prioritize buffer reuse for large arrays or where GC pause is critical. For small, short-lived arrays (like 6x6 spatial transforms), simple allocation might be faster due to lower Python interpreter overhead.
