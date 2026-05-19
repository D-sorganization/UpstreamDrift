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
## 2026-05-01 - [Optimize UI re-rendering and data resorting during filtering]
**Learning:** In React, typing rapidly into an input field (like a data filter) that triggers state updates at the root component level can cause severe performance lag if expensive operations like array sorting (`[...rows].sort()`) or rendering large child components (like a `DataTable`) are executed synchronously on every single keystroke render cycle.
**Action:** Always wrap expensive derived computations in `useMemo()` with appropriate dependency arrays so they only re-compute when their specific inputs change, and wrap large, purely presentational child components in `React.memo()` so they don't blindly re-render when a parent's unrelated state (like the filter input text) changes.

## 2025-05-18 - Optimize bounding sphere radius computation in mesh primitive fitting
**Learning:** `np.linalg.norm` evaluates element-wise square roots and allocates intermediate temporary arrays. Since `max` and `sqrt` are commutative for positive numbers, computing the maximum sum-of-squares first using `np.einsum`, then applying `sqrt` avoids memory allocations and performs exactly 1 square root instead of N square roots.
**Action:** Replace `np.max(np.linalg.norm(vertices, axis=1))` with `np.sqrt(np.max(np.einsum('ij,ij->i', vertices, vertices)))` when calculating bounding sphere radii from mesh vertices to improve performance.
## 2026-05-01 - Optimize clubhead speed computation using einsum
**Learning:** `np.linalg.norm(..., axis=1)` on multi-dimensional arrays evaluates element-wise square roots and allocates intermediate temporary arrays, making it relatively slow.
**Action:** Replace `np.linalg.norm(x, axis=1)` with `np.sqrt(np.einsum("ij,ij->i", x, x))` to calculate magnitudes. This avoids temporary array allocations and is ~35% faster.
## 2026-05-18 - Optimize norm calculation combined with argmax
**Learning:** `np.linalg.norm(..., axis=1)` creates intermediate memory allocations and has overhead when used with `np.argmax`. Since `argmax` is invariant to monotonic transformations like `sqrt`, the `sqrt` can be completely omitted.
**Action:** Replace `np.argmax(np.linalg.norm(x, axis=1))` with `np.argmax(np.einsum('ij,ij->i', x, x))` to find the index of the maximum magnitude vector without calculating the full norm. This yields significant speedup by avoiding both intermediate allocations and square root computation.

## 2026-05-18 - Optimize sum of squares along axis
**Learning:** `np.sum(diff ** 2, axis=1)` evaluates element-wise square and sum operations along an axis, creating intermediate memory allocations and has overhead.
**Action:** Replace `np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))` with `np.sqrt(np.vdot(diff, diff) / diff.shape[0])` when evaluating the RMSE on an array of coordinates over N frames, by vectorizing the sum of squares across all the matrix coordinates at once. This avoids intermediate allocations and accelerates the calculations significantly.
## 2026-05-18 - Optimize norm calculation combined with argmax
**Learning:** `np.linalg.norm(..., axis=1)` creates intermediate memory allocations and has overhead when used with `np.argmax`. Since `argmax` is invariant to monotonic transformations like `sqrt`, the `sqrt` can be completely omitted.
**Action:** Replace `np.argmax(np.linalg.norm(x, axis=1))` with `np.argmax(np.einsum('ij,ij->i', x, x))` to find the index of the maximum magnitude vector without calculating the full norm. This yields significant speedup by avoiding both intermediate allocations and square root computation.
## 2026-05-12 - Optimize sum of squares along axis using einsum and vdot
**Learning:** Computing sum of squares over a dimension, or MSE with `np.mean(x**2)` causes a large intermediate allocation in `numpy` before it gets reduced/averaged. For frequently called hot paths, avoiding these allocations matters.
**Action:** Use `np.vdot(x, x)` or `np.vdot(x, x)/x.size` to compute total squared norm or MSE. Use `np.einsum('ij,ij->i', x, x)` to compute row-wise squared norms without allocating a fully squared temporary matrix. Make sure the type is appropriately float.
## 2025-05-18 - Optimize norm calculation along axis using einsum
**Learning:** `np.linalg.norm(..., axis=1)` is relatively slow for small inner dimensions because of internal overhead in NumPy and intermediate allocations. Replacing it with `np.sqrt(np.einsum('ij,ij->i', x, x))` computes the identical L2 norm while avoiding the overhead and allocations, yielding significant performance speedups.
**Action:** When computing vector magnitudes or Euclidean norms along an axis, use `np.sqrt(np.einsum('ij,ij->i', x, x))` instead of `np.linalg.norm(x, axis=1)` to improve performance. For scenarios where `keepdims=True` was used, append `[:, np.newaxis]` to the einsum result.

## 2026-05-18 - Optimize norm calculation along axis using einsum for variables
**Learning:** `np.linalg.norm(..., axis=1)` and `np.linalg.norm(..., axis=-1)` are relatively slow because of internal overhead in NumPy and intermediate allocations. `np.sqrt(np.einsum('ij,ij->i', x, x))` or `np.sqrt(np.einsum('...i,...i->...', x, x))` computes the identical L2 norm while avoiding the overhead and allocations, yielding ~1.7x to ~2.2x performance speedups for N-dimensional arrays.
**Action:** When computing vector magnitudes or Euclidean norms along an axis, use `np.sqrt(np.einsum('ij,ij->i', x, x))` instead of `np.linalg.norm(x, axis=1)` to improve performance. For scenarios where `keepdims=True` was used, append `[:, np.newaxis]` or `[..., np.newaxis]` to the einsum result.

## 2026-05-18 - Cast integer vectors before einsum norm
**Learning:** `np.einsum` operations on integer vectors can silently overflow before the `np.sqrt` calculation when performing operations like `np.einsum("...i,...i->...", x, x)`, resulting in negative values which produce `NaN` or incorrect magnitudes. The old `np.linalg.norm` handled this correctly by returning float results.
**Action:** Always ensure numeric arrays that might be integers are explicitly cast or promoted to float using `np.asarray(vector, dtype=np.float64)` before attempting `np.einsum` calculations for magnitudes.
## 2026-05-18 - Optimize norm calculations in UI/Viz adapters
**Learning:** `np.linalg.norm(..., axis=1)` creates an intermediate memory allocation and has significant internal overhead when used on multi-dimensional numpy arrays inside tight loops, leading to suboptimal performance, particularly when parsing and calculating distances in data visualizers.
**Action:** Replace `np.linalg.norm(diff, axis=1)` with `np.sqrt(np.einsum("ij,ij->i", diff, diff))` for all generic vector distance calculations that map down dimensions. Remember to retain any shape alterations such as `[:, np.newaxis]` when performing element-wise broadcasting on multi-dimensional arrays, so the shapes do not mismatch.
## 2026-05-19 - Optimize generic element-wise norm for small vectors
**Learning:** Element-wise norm computation or generic `np.linalg.norm(..., axis=None)` creates temporary arrays and runs via python layer handling. For very small native tuples or 1D arrays like normal vectors, standard `math.hypot(*v)` is substantially faster than `math.hypot(*np.ravel(v))` and extremely faster than `np.linalg.norm`.
**Action:** Always prefer `math.hypot(*v)` directly rather than applying `np.ravel()` first on 1D flat structures when optimizing tiny vectors for normalisations.
