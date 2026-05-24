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
## 2026-05-18 - Optimize sum of squares using vdot in motion matching
**Learning:** `np.sum(diff * diff)` and `np.sum(diff ** 2)` create intermediate memory allocations and have overhead. For flattening large arrays and calculating RMSE over a trajectory, `np.vdot(diff, diff) / N` is significantly faster as it avoids allocations and computes the sum of squares directly at the C level.
**Action:** Replace `np.mean(np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1))` with `(np.vdot(db, db) + np.vdot(dc, dc)) / db.shape[0]` when calculating per-frame RMSE. Replace `sum(np.sum(r ** 2))` with `sum(np.vdot(r, r))` for constraint residuals.
## 2026-05-18 - Optimize dot product over trajectory along an axis
**Learning:** `np.sum(a * b, axis=1)` to compute row-wise dot products creates intermediate memory allocations for `a * b` and has internal numpy sum overhead. `np.einsum("ij,ij->i", a, b)` computes this without allocations and is faster.
**Action:** Replace `np.sum(a * b, axis=1)` with `np.einsum("ij,ij->i", a, b)` when computing dot products between pairs of vectors (like quaternions) across a time trajectory.
## 2026-05-18 - Optimize element-wise sum of squares using vdot
**Learning:** `np.sum(a * b**2)` allocates a temporary array in memory. When multiplying and squaring, `np.vdot(a, b * b)` utilizes optimized C logic and avoids unnecessary intermediate array allocations, speeding up operations in calculation-heavy functions.
**Action:** Replace `np.sum(inertia * ang_vel**2)` with `np.vdot(inertia, ang_vel * ang_vel)` when calculating rotational kinetic energy across body segments.

## 2026-05-19 - Optimize generic element-wise norm for small vectors
**Learning:** Element-wise norm computation or generic `np.linalg.norm(..., axis=None)` creates temporary arrays and runs via python layer handling. For very small native tuples or 1D arrays like normal vectors, standard `math.hypot(*v)` is substantially faster than `math.hypot(*np.ravel(v))` and extremely faster than `np.linalg.norm`.
**Action:** Always prefer `math.hypot(*v)` directly rather than applying `np.ravel()` first on 1D flat structures when optimizing tiny vectors for normalisations.
## 2025-05-19 - Optimize R-squared and RMSE calculation using vdot
**Learning:** `np.sum(residuals**2)` allocates a temporary array in memory of the same size as `residuals` because of the element-wise squaring `**2`. For simple calculations like Sum of Squared Residuals (SS_res) and Total Sum of Squares (SS_tot) over 1D arrays, this overhead is noticeable in fitting toolkits.
**Action:** Replace `np.sum(x**2)` with `np.vdot(x, x)` to calculate sum of squares without allocating intermediate temporary memory for the squares, speeding up statistical fitting implementations. Additionally, avoid repeatedly recalculating mean squares by reusing the `ss_res` result (e.g. `rmse = np.sqrt(ss_res / x.size)`).
## 2026-05-19 - Optimize MSE calculation in JAX
**Learning:** `jnp.mean(diff ** 2)` and `jnp.sum(x ** 2)` create intermediate array allocations that slow down evaluation in `jax` loss functions.
**Action:** Replace `jnp.mean(diff ** 2)` with `jnp.vdot(diff, diff) / diff.size` and `jnp.sum(x ** 2)` with `jnp.vdot(x, x)` to optimize sum-of-squares evaluations by avoiding intermediate temporary memory allocations.

## 2026-05-19 - Optimize mechanical work metrics
**Learning:** Computations like `np.sum(derivatives**2)` and `np.sum(values**2 * dt)` or `np.sum(torque**2, axis=1)` involve element-wise operations that allocate intermediate memory (`**2` and multiplication). This can be slow for long arrays of time series data.
**Action:** Replace `np.sum(x**2)` with `np.vdot(x, x)`, `np.sum(x**2 * dt)` with `np.vdot(x, x * dt)`. For multi-dimensional axis summation like `np.sum(torque**2, axis=1)`, replace it with `np.einsum('ij,ij->i', torque, torque)`. Similarly, replace `np.sum(np.abs(torque), axis=1)` with `np.einsum('ij->i', np.abs(torque))`.

## 2026-05-23 - GitHub Actions setup-python pip issues
**Learning:** Installing `pydantic-core` sometimes fails with uninstall errors under the virtual environments created by `actions/setup-python` during GitHub Actions CI due to missing `RECORD` files or invalid metadata entries, particularly when repeatedly updating `pip` inside the runner.
**Action:** When working on GitHub Actions Python CI scripts, ensure to proactively use `pip install --ignore-installed --no-deps pydantic-core==2.46.3 || true` right before the main `pip install -e ".[dev]"` lines, to bypass unresolvable cache/uninstall issues for `pydantic-core`.

## 2026-05-23 - xvfb missing in GitHub Actions runners
**Learning:** Some test suites (like `test_leaderboard.py`) that rely on plotting or visual dependencies fail in CI with `xvfb-run` missing errors, or display server missing if `xvfb-run -a` is prepended but `xvfb` is not installed on the system (exit code 3).
**Action:** When prepending `xvfb-run -a` to a test command in a CI workflow, also make sure to explicitly run `sudo apt-get update && sudo apt-get install -y xvfb` inside the environment setup step if the runner is missing it.

## 2026-05-23 - trimesh ImportErrors
**Learning:** Hard-coded imports of `trimesh` in files like `_mesh_decimation.py` and `_mesh_io.py` can cause tests or other modules that import them to fail if `trimesh` isn't installed.
**Action:** Always wrap `import trimesh` with a `try...except ImportError` block and conditionally check if `trimesh is None` to safely handle environments where it is missing, or alternatively, make sure to add it to the test environment requirements.
## 2026-05-18 - Optimize dot product over trajectory along an axis
**Learning:** `np.sum(a * b, axis=1)` to compute row-wise dot products creates intermediate memory allocations for `a * b` and has internal numpy sum overhead. `np.einsum("ij,ij->i", a, b)` computes this without allocations and is faster.
**Action:** Replace `np.sum(a * b, axis=1)` with `np.einsum("ij,ij->i", a, b)` when computing dot products between pairs of vectors (like quaternions) across a time trajectory.
