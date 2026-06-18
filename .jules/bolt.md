## 2026-06-14 - math.hypot for 3D vector magnitudes
**Learning:** For small fixed-size numpy arrays (like 3D velocity vectors), extracting individual components and using `math.hypot(x, y, z)` avoids NumPy's type checking, dispatching, and temporary array allocation overhead.
**Action:** Use explicit component extraction and `math.hypot` instead of `np.linalg.norm` in tight inner loops processing very small arrays, but ensure the array size is rigidly fixed to avoid IndexError.

## 2024-06-15 - [NumPy L2 Norm Calculation Optimization]
**Learning:** For calculating the Euclidean norm along an axis for 2D arrays (like forces or coordinates) in NumPy, `np.linalg.norm(array, axis=1)` incurs significant performance overhead due to intermediate memory allocations and function dispatch routing.
**Action:** Replace `np.linalg.norm(array, axis=1)` with `np.sqrt(np.einsum('ij,ij->i', array, array))` in hot paths or large data transformations. This avoids temporary allocations and provides a measured speedup of ~2.4x.

## 2026-06-16 - np.einsum for L2 norms over axes
**Learning:** For multi-dimensional NumPy arrays where you compute an L2 norm along a specific axis (like calculating RMSE distance over an array of 3D coordinates using `np.mean(np.sum(diff**2, axis=1))`), `np.sum(...**2, axis=...)` allocates a temporary array for the intermediate squares. `np.einsum` avoids this.
**Action:** Replace `np.sum(diff**2, axis=1)` with `np.einsum('ij,ij->i', diff, diff)` or `np.sum(diff*diff, axis=-1)` with `np.einsum('...i,...i->...', diff, diff)` to achieve a 2-3x speedup.
## 2026-06-17 - np.einsum for sum over axis
**Learning:** For multi-dimensional NumPy arrays where you compute a sum along a specific axis (like calculating `np.sum(H, axis=2)`), `np.sum(..., axis=...)` has more overhead. `np.einsum` avoids this.
**Action:** For fixed-shape FEM tensors, consider replacing `np.sum(H, axis=2)` with `np.einsum('ijk->ij', H)` only after parity coverage locks the batched reduction shape.

## 2024-05-14 - Replace np.linalg.norm with math.hypot for 2D/3D vectors
**Learning:** `math.hypot(x, y)` is significantly faster (~5x) than `np.linalg.norm` for explicitly unpacked 2D arrays, and `math.hypot(x, y, z)` is similarly faster for 3D arrays, but array dimension assumptions should be verified carefully, or explicit bounds checks (`len()`) applied to avoid indexing errors when vectors might be 2D or 3D.
**Action:** Replace `np.linalg.norm(v)` with `math.hypot(v[0], v[1])` or `math.hypot(v[0], v[1], v[2])` when vector sizes are known to be 2D or 3D to improve performance. Use length checks if the size varies.
## 2026-06-18 - [NumPy vs Math Overhead in Robotics Simulation Loops]
**Learning:**  carries a significant amount of Python-level overhead (input validation, general-purpose shape handling) which makes it considerably slower than  for small 1D vectors, and slower than  for small statically sized vectors (like 3D coordinates). These changes yield genuine speedups in tight loops (e.g. simulation environments). When optimizing mock assertions in related tests, carefully track test assertions that expect specific mock call counts, as optimizations might indirectly affect or expose pre-existing mock count bugs.
**Action:** Use  and  instead of  for small array magnitude operations in critical paths.
## 2026-06-18 - [NumPy vs Math Overhead in Robotics Simulation Loops]
**Learning:** `np.linalg.norm()` carries a significant amount of Python-level overhead (input validation, general-purpose shape handling) which makes it considerably slower than `math.sqrt(np.dot(v, v))` for small 1D vectors, and slower than `math.hypot()` for small statically sized vectors (like 3D coordinates). These changes yield genuine speedups in tight loops (e.g. simulation environments). When optimizing mock assertions in related tests, carefully track test assertions that expect specific mock call counts, as optimizations might indirectly affect or expose pre-existing mock count bugs.
**Action:** Use `math.sqrt(np.dot(x, x))` and `math.hypot` instead of `np.linalg.norm` for small array magnitude operations in critical paths.

## 2026-06-19 - Unmeasurable micro-optimizations
**Learning:** Replacing `np.linalg.norm` with `math.hypot` inside camera controllers (like `golf_camera_system.py`) to avoid numpy array allocation overhead only saves a few microseconds per call. This has no measurable impact on the overall application performance.
**Action:** Do not apply micro-optimizations (like converting from `np.linalg.norm` to `math.hypot`) in paths that do not have a measurable performance impact (like UI or camera controllers). This sacrifices code readability for unmeasurable gains.
