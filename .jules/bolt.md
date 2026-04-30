## 2026-04-26 - Optimizing Sum of Squares in Trajectory Benchmark
**Learning:** `np.sum(error**2)` allocates a temporary array in memory to store the squared values before summing them, which is slow.
**Action:** Replace `np.sum(error**2)` with `np.vdot(error, error)` on flat/real-numbered arrays. It computes the dot product directly at the C level, yielding up to a ~3-4x performance speedup by avoiding intermediate array allocations.

## 2024-04-26 - Optimize Mean Squared Error calculations
**Learning:** When computing Mean Squared Error (MSE) across arrays, `np.vdot(diff, diff) / diff.size` is significantly faster (~2x) than `np.mean(diff**2)` because it leverages optimized C code and completely bypasses the memory allocation overhead of creating a temporary squared array.
**Action:** Use `np.vdot(diff, diff) / diff.size` instead of `np.mean(diff**2)` when calculating MSE in hot paths.## 2025-04-27 - [Optimize norm calculation for collision checking]
**Learning:** Element-wise norm computations or generic `np.linalg.norm(..., axis=None)` applied to 3D arrays are slower than leveraging `math.hypot(*v)`. Since robotics frequently computes distances between points, optimizing Euclidean distance computation brings measurable speedups.
**Action:** Replace `np.linalg.norm(v)` with `math.hypot(*v)` where `v` is a small fixed-length vector (e.g., 3D point) in high-frequency distance queries like collision checks.
## 2025-02-27 - Optimize sum of squares using einsum
**Learning:** `np.sum(x**2, axis=0)` allocates intermediate memory to store squared values. Replacing it with `np.einsum('i...,i...->...', x, x)` sidesteps intermediate temporary arrays allocation, reducing memory pressure.
**Action:** When computing vector lengths or magnitudes, use `np.einsum` or `np.vdot` to prevent temporary array allocations to improve performance.

## 2025-05-18 - Optimize sum of squares using einsum
**Learning:** `np.linalg.norm(..., axis=1)` is relatively slow for small inner dimensions because of internal overhead in NumPy and intermediate allocations. Replacing it with `np.sqrt(np.einsum('ij,ij->i', x, x))` computes the identical L2 norm while avoiding the overhead and allocations, yielding significant performance speedups (e.g. ~35% for small 3D vectors).
**Action:** When computing vector magnitudes or Euclidean norms along an axis, use `np.sqrt(np.einsum('ij,ij->i', x, x))` instead of `np.linalg.norm(x, axis=1)` to improve performance.
