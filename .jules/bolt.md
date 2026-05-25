## 2025-05-25 - np.sum(x * x) Bottleneck with temporary arrays

**Learning:** When dealing with multi-dimensional numpy arrays in performance-critical code loops (like motion matching evaluations), expressions like `np.sum(x * x, axis=...)` are problematic because they force numpy to allocate temporary intermediate arrays in memory before summing them, which is slow.
**Action:** Replace `np.sum(x * x, axis=2)` or `np.sum(x * x, axis=1)` with the `np.einsum` equivalent, e.g., `np.einsum("ijk,ijk->ij", db, db)`. This operates directly at the C-level avoiding temporary array allocations, giving a ~2-3x speedup.

## 2025-05-25 - np.sum(x * x) Bottleneck with temporary arrays

**Learning:** When dealing with multi-dimensional numpy arrays in performance-critical code loops (like motion matching evaluations), expressions like `np.sum(x * x, axis=...)` are problematic because they force numpy to allocate temporary intermediate arrays in memory before summing them, which is slow.
**Action:** Replace `np.sum(x * x, axis=2)` or `np.sum(x * x, axis=1)` with the `np.einsum` equivalent, e.g., `np.einsum("ijk,ijk->ij", db, db)`. This operates directly at the C-level avoiding temporary array allocations, giving a ~2-3x speedup.

## 2025-05-25 - np.sum(x * x) Bottleneck with temporary arrays

**Learning:** When dealing with multi-dimensional numpy arrays in performance-critical code loops (like motion matching evaluations), expressions like `np.sum(x * x, axis=...)` are problematic because they force numpy to allocate temporary intermediate arrays in memory before summing them, which is slow.
**Action:** Replace `np.sum(x * x, axis=2)` or `np.sum(x * x, axis=1)` with the `np.einsum` equivalent, e.g., `np.einsum("ijk,ijk->ij", db, db)`. This operates directly at the C-level avoiding temporary array allocations, giving a ~2-3x speedup.
