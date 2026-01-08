## 2025-02-23 - F-Order Arrays for Spatial Dynamics
**Learning:** Switching to Fortran-contiguous (column-major) arrays (`order='F'`) for spatial vector collections (like velocity `v` and acceleration `a` with shape `(6, nb)`) enables efficient use of columns as `out` parameters in NumPy operations (like `matmul`). This avoids implicit temporary buffering and copying that occurs when writing to non-contiguous slices of C-ordered arrays, yielding a ~16% speedup in rigid body dynamics loops.

**Action:** When implementing spatial algebra algorithms (RNEA, ABA) where vectors are stored in columns `(6, N)`, initialize arrays with `order='F'` and verify that `out` arguments point to contiguous memory segments.

## 2025-02-24 - NumPy Norm Overhead
**Learning:** `np.linalg.norm(axis=1)` has significant overhead for small fixed dimensions (like 2D/3D vectors) compared to explicit calculation `np.sqrt(x**2 + y**2)`. In `StatisticalAnalyzer.compute_grf_metrics`, replacing `norm` with explicit calculation yielded a ~27% speedup.

**Action:** For performance-critical code involving norms of 2D/3D vectors, prefer explicit `sqrt(sum(sq))` over `np.linalg.norm`.
