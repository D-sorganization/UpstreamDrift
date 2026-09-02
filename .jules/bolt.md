## YYYY-MM-DD - [Optimize Python List and Array Reduction Overheads]
**Learning:** Using `np.sum()` on a standard Python list introduces significant overhead because NumPy must first implicitly convert the list into a temporary ndarray. This overhead is heavily pronounced when the list contains NumPy scalars, as opposed to raw Python floats. Python's built-in `sum()` is ~10x faster for such lists. Similarly, evaluating `np.count_nonzero()` on boolean arrays is ~30% faster than `np.sum()` because it counts directly at the C-level without engaging the full summation machinery.
**Action:** Replace `np.sum()` with the built-in `sum()` for python lists, and use `np.count_nonzero()` for boolean arrays. Additionally, replace module-level `np.sum(array)` with the array method `.sum()` when applicable.

## 2026-08-25 - [Optimize Argmax of Vector Magnitude]
**Learning:** Calculating the `argmax` (or `argmin`) of vector magnitudes (e.g., `np.argmax(np.linalg.norm(arr, axis=1))`) incurs significant overhead due to intermediate array creations in `np.linalg.norm` and unnecessary square root calculations. Since square root is monotonically increasing, the index of the maximum magnitude is strictly the same as the index of the maximum squared magnitude. Using `np.einsum('ij,ij->i', arr, arr)` to directly compute the array of squared magnitudes yields the exact same index without any temporary allocations or root evaluations, which provides measurable speedup.
**Action:** Replace `np.argmax(np.linalg.norm(arr, axis=1))` with `np.argmax(np.einsum('ij,ij->i', arr, arr))` to safely and efficiently optimize. Coerce `arr` with `np.asarray` first if it might not natively be a NumPy ndarray.
## 2024-05-19 - Fast Multidimensional Array Magnitude
**Learning:** `np.linalg.norm(..., axis=1)` is known to be relatively slow due to internal overhead and intermediate array allocations. Replacing it with `np.sqrt(np.einsum('ij,ij->i', ...))` is a highly effective optimization that provides a significant speedup (often 2x-4x faster for small-to-medium arrays) while keeping the code readable.
**Action:** When computing vector norms along an axis (other than small 2D vectors where `np.hypot` is best), use `np.sqrt(np.einsum)` instead of `np.linalg.norm` to avoid intermediate allocations and speed up the computation.
## 2024-05-19 - Fast Small Array Magnitude
**Learning:** `np.linalg.norm()` is known to be relatively slow for small arrays (1D arrays with 2 to 6 elements) due to internal overhead and instance checks. Built-in `math.hypot()` is much faster, providing a ~2x to ~5x speedup. For arrays larger than that, `math.sqrt(np.vdot(arr, arr))` provides a ~2x speedup by bypassing `np.linalg.norm` overhead while leveraging the fast C-level `np.vdot`.
**Action:** When computing vector norms for small 1D arrays, replace `np.linalg.norm(v)` with `math.hypot(*v)` for small arrays (length <= 6) or `math.sqrt(np.vdot(v, v))` for other 1D cases. For simple checks like `np.linalg.norm(v) > 0.0`, `np.vdot(v, v) > 0.0` completely skips the square root.
## 2024-05-19 - Limit Micro-Optimizations for Array Summation
**Learning:** While replacing `np.sum(array)` with `array.sum()` does avoid NumPy's internal function dispatch overhead (~1 microsecond), it is an extreme micro-optimization. In heavy computational contexts (such as Principal Component Analysis involving SVD), this change has absolutely no measurable impact on overall application performance and is not worth the noise of inline comments or PR churn.
**Action:** Do not perform this `.sum()` replacement in standard calculations unless it is inside a provably hot loop where the microsecond overhead is a true bottleneck.

## 2026-08-29 - [Optimize Square Array Summation]
**Learning:** Computing the sum of squares of an array (e.g., `np.square(arr).sum()` or `np.sum(arr**2)`) incurs unnecessary overhead due to the intermediate array created by `np.square()` or `**2`. By using `np.vdot(arr, arr)`, we skip this temporary array allocation and speed up the computation directly at the C-level (often ~2x faster).
**Action:** When computing the sum of squared elements for real floating-point arrays, replace `np.square(arr).sum()` or `np.sum(arr**2)` with `np.vdot(arr, arr)`. Do not apply this to complex arrays (`np.vdot` conjugates its first argument, giving `sum(|arr|**2)` rather than `sum(arr**2)`) or to narrow integer/boolean arrays (`np.vdot` keeps the narrow dtype instead of `np.sum`'s promoted accumulator, so it can overflow or change a boolean result).

## 2024-05-24 - API Array Summation Optimization
**Learning:** Replaced `np.sum()` with `.sum()` for small array math in `physics.py` logic. This avoids numpy dispatch and yields measurable speedup.
**Action:** When working in hotpath math like loop array modifications, always look to use direct methods rather than NumPy's wrapped equivalents.

## 2026-09-01 - [Optimize Norm Calculation in Morris Design]
**Learning:** Using `np.sqrt(np.einsum("ij,ij->i", diff, diff))` is significantly faster (~30%) than `np.linalg.norm(diff, axis=1)` for 2D differences, avoiding intermediate array allocation in the inner loops of combinatorial design sampling algorithms.
**Action:** Replace `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum("ij,ij->i", diff, diff))` in `src/bunkershot3d/study/morris.py` to optimize trajectory distance calculations.

## 2026-09-01 - [Optimize Norm Calculation in Morris Design]
**Learning:** Using `np.sqrt(np.einsum("ij,ij->i", diff, diff))` is significantly faster (~30%) than `np.linalg.norm(diff, axis=1)` for 2D differences, avoiding intermediate array allocation in the inner loops of combinatorial design sampling algorithms.
**Action:** Replace `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum("ij,ij->i", diff, diff))` in `src/bunkershot3d/study/morris.py` to optimize trajectory distance calculations.

## 2024-05-19 - [Optimize Sum of Squares]
