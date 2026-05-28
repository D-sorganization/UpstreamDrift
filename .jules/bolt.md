## 2025-05-25 - np.sum(x * x) Bottleneck with temporary arrays

**Learning:** When dealing with multi-dimensional numpy arrays in performance-critical code loops (like motion matching evaluations), expressions like `np.sum(x * x, axis=...)` are problematic because they force numpy to allocate temporary intermediate arrays in memory before summing them, which is slow.
**Action:** Replace `np.sum(x * x, axis=2)` or `np.sum(x * x, axis=1)` with the `np.einsum` equivalent, e.g., `np.einsum("ijk,ijk->ij", db, db)`. This operates directly at the C-level avoiding temporary array allocations, giving a ~2-3x speedup.

## 2025-05-25 - np.sum(x * x) Bottleneck with temporary arrays

**Learning:** When dealing with multi-dimensional numpy arrays in performance-critical code loops (like motion matching evaluations), expressions like `np.sum(x * x, axis=...)` are problematic because they force numpy to allocate temporary intermediate arrays in memory before summing them, which is slow.
**Action:** Replace `np.sum(x * x, axis=2)` or `np.sum(x * x, axis=1)` with the `np.einsum` equivalent, e.g., `np.einsum("ijk,ijk->ij", db, db)`. This operates directly at the C-level avoiding temporary array allocations, giving a ~2-3x speedup.

## 2025-05-25 - np.sum(x * x) Bottleneck with temporary arrays

**Learning:** When dealing with multi-dimensional numpy arrays in performance-critical code loops (like motion matching evaluations), expressions like `np.sum(x * x, axis=...)` are problematic because they force numpy to allocate temporary intermediate arrays in memory before summing them, which is slow.
**Action:** Replace `np.sum(x * x, axis=2)` or `np.sum(x * x, axis=1)` with the `np.einsum` equivalent, e.g., `np.einsum("ijk,ijk->ij", db, db)`. This operates directly at the C-level avoiding temporary array allocations, giving a ~2-3x speedup.
## 2024-05-28 - [Drake accuracy_cost Optimization]
**Learning:** For small 1D arrays, `np.linalg.norm` is much slower (~6.4s for 1M calls) than simply computing the sum of squares with `np.dot` and taking the square root (~4.3s for 1M calls) because `np.linalg.norm` creates intermediate array allocations and handles complexities not needed for small, 1D vector distances.
**Action:** Replace `np.linalg.norm` with `np.dot` in tight loop calculation to avoid power overhead and array allocations when the vector dimensionality is very small.
