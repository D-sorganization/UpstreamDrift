
## 2026-04-15 - np.sum(diff**2) Performance Optimization

**Learning:** When optimizing multidimensional array operations (like distances in `NonlinearDynamicsMixin`), replacing explicit sum of squares (`np.sum(diff**2, axis=-1)`) with `np.einsum('...i,...i->...', diff, diff)` yields a ~3x speedup by reducing intermediate memory allocations for the squared differences. However, the data should not be forced into a float cast (like `astype(float)`) if it's already floating point, as this introduces an unnecessary copy and memory allocation. Further, using `astype(float, copy=False)` is fragile and raises a `ValueError` in NumPy 2.0+.

**Action:** Prefer `np.einsum` for computing sums of squares along an axis. Ensure the arrays are natively floats (like most physics/simulation states) before applying `np.einsum` directly, to avoid integer overflows and avoid explicit `astype` casts.

## 2026-04-16 - Optimizing np.linalg.norm axis reductions

**Learning:** When using `np.sum(diff**2, axis=-1)` or `np.linalg.norm(..., axis=1/2)` on multi-dimensional array operations (such as computing segment lengths or marker error metrics), using `np.einsum('...i,...i->...', diff, diff)` provides a ~2x speedup and avoids temporary array allocations. For Euclidean distances, using `np.sqrt(np.einsum(...))` is the fastest method while remaining dimension-agnostic, reducing memory pressure.

**Action:** Consistently replace `np.sum((a - b)**2, axis=-1)` and `np.linalg.norm(..., axis=X)` with `np.sqrt(np.einsum('...i,...i->...', diff, diff))` when performing reductions across small inner dimensions (like 3D coordinates).

## 2026-04-20 - Optimization of Numba JIT compiled distance functions
**Learning:** When optimizing functions compiled with Numba (`@njit`), using NumPy utility functions like `np.linalg.norm` and `np.allclose` for small fixed-size vectors (e.g., 3D coordinates) causes significant abstraction overhead.
**Action:** Replacing them with explicit element-wise arithmetic (e.g., `math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])`) and manual dot products avoids this overhead and yields over a 2x performance improvement.
## 2024-05-18 - [math.hypot over generators for distance]
**Learning:** When calculating Euclidean distance from a list of deltas in Python, using a generator expression like `math.sqrt(sum(d * d for d in delta))` adds significant Python overhead compared to passing the elements directly to C-optimized functions.
**Action:** Use `math.hypot(*delta)` for arbitrary dimension iterables, as it avoids generator overhead and evaluates entirely in C, yielding a ~4x speedup.
## 2024-05-18 - [math.dist over math.hypot and list comprehension]
**Learning:** When calculating Euclidean distance between two points, computing an intermediate delta array via a list comprehension and applying `math.hypot(*delta)` is still slower than using `math.dist(pos_a, pos_b)` directly, which completely avoids creating the intermediate lists in python and computes the distance in C.
**Action:** Use `math.dist(pos_a, pos_b)` to calculate the distance between iterables, and then only compute `delta` if it is strictly necessary to return it.

## 2024-05-18 - Replacing `np.linalg.norm` with `math.hypot`
**Learning:** For small vectors (2D or 3D) like physical velocities or forces, calculating magnitudes using `math.hypot(*velocity)` or `math.hypot(v[0], v[1])` is up to ~5x faster than using `np.linalg.norm(velocity)`. This avoids the substantial overhead of NumPy's generalized, dimension-agnostic reduction functions, memory allocations, and Python-to-C API calls for tiny arrays.
**Action:** When working with 2D or 3D physics vectors where you need magnitudes (e.g., speed, spin magnitude, 2D horizontal velocities) on a hot path (like force calculators and integration loops), use the built-in `math.hypot` with unpacked elements (e.g., `math.hypot(*vec)`) instead of `np.linalg.norm`.
