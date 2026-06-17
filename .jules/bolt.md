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
