## 2026-09-09 - [Optimize Python List and Array Reduction Overheads]
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

## 2024-05-19 - [Optimize Sum of Squares]

## 2026-09-05 - Optimize np.linalg.norm for multidimensional arrays
**Learning:** Using np.linalg.norm(..., axis=1) in NumPy forces multiple internal dispatch checks and temporary array allocations, which become a bottleneck in tight loops or large sweeps. Replacing it with np.sqrt(np.einsum('ij,ij->i', arr, arr)) bypasses this overhead and is ~2.4x faster for medium-sized multidimensional arrays.
**Action:** Always prefer np.sqrt(np.einsum(...)) or math.hypot (for small slices) over np.linalg.norm when calculating magnitude along an axis in high-performance or simulation modules.

## 2026-09-05 - Optimizing `np.mean` for Python Lists
**Learning:** Calling `np.mean()` on a standard Python list (e.g., `np.mean([d.duration for d in self.demonstrations])`) forces an expensive implicit conversion to a temporary NumPy array.
**Action:** Replace `np.mean(list)` with built-in `sum(list) / len(list)` (handling zero-division if the list can be empty) to avoid allocation overhead, which is significantly faster.

## 2024-05-19 - Fast Multidimensional Array Magnitude
**Learning:** `np.linalg.norm(..., axis=1)` and `np.linalg.norm(..., axis=2)` is known to be relatively slow due to internal overhead and intermediate array allocations. Replacing it with `np.sqrt(np.einsum('ij,ij->i', ...))` or `np.sqrt(np.einsum('ijk,ijk->ij', ...))` is a highly effective optimization that provides a significant speedup (often 2x-4x faster for small-to-medium arrays) while keeping the code readable.
**Action:** When computing vector norms along an axis (other than small 2D vectors where `np.hypot` is best), use `np.sqrt(np.einsum)` instead of `np.linalg.norm` to avoid intermediate allocations and speed up the computation.
## 2024-05-18 - Optimized Sphere Collision Radius Calculation
**Learning:** `np.sqrt(np.max(np.einsum("ij,ij->i", vertices, vertices)))` is about ~2x faster than `np.max(np.linalg.norm(vertices, axis=1))` when computing the maximum distance to vertex for bounding sphere collision.
**Action:** Use `einsum` to square the components to avoid temporary allocations for multidimensional arrays.
## 2024-05-19 - Optimize Norm Calculation in Analytics
**Learning:** Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` instead of `np.linalg.norm(..., axis=1)` is ~2.4x faster for medium-sized multidimensional arrays because it avoids multiple internal dispatch checks and temporary array allocations within NumPy's linear algebra engine. It's particularly useful in data-heavy analysis loops like motion capture reconstruction.
**Action:** Replace `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum('ij,ij->i', arr, arr))` in data-heavy analysis pipelines to optimize array magnitude calculations.

## 2026-09-09 - [Optimize Multidimensional Norm Calculations in Motion Capture]
**Learning:** In the motion capture analysis and reconstruction pipelines (e.g., `src/motion_capture/compare_variants.py`), computing the Euclidean distance along the inner-most axis using `np.linalg.norm(..., axis=2)` incurs significant overhead due to NumPy's internal dispatching and temporary array allocations. Profiling reveals that computing the difference first and then evaluating `np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))` is ~10-20x faster.
**Action:** Replace `np.linalg.norm(..., axis=2)` with `np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))` where multidimensional array magnitude operations are performed in loops or heavily executed evaluation methods. Ensure `diff` is explicitly calculated once to prevent duplicate temporary allocations inside the `einsum` call.

## 2026-09-09 - [Optimize Multidimensional Norm Calculations in Motion Capture]
**Learning:** In the motion capture analysis and reconstruction pipelines (e.g., `src/motion_capture/compare_variants.py`), computing the Euclidean distance along the inner-most axis using `np.linalg.norm(..., axis=2)` incurs significant overhead due to NumPy's internal dispatching and temporary array allocations. Profiling reveals that computing the difference first and then evaluating `np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))` is ~10-20x faster.
**Action:** Replace `np.linalg.norm(..., axis=2)` with `np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))` where multidimensional array magnitude operations are performed in loops or heavily executed evaluation methods. Ensure `diff` is explicitly calculated once to prevent duplicate temporary allocations inside the `einsum` call.
## 2026-09-09 - [Optimize Root Mean Square Calculation]
**Learning:** Computing the RMS of a multi-dimensional error array (e.g., `np.sqrt(np.mean(np.sum(error**2, axis=0)))`) incurs overhead due to the intermediate array created by squaring and summing over an axis. By replacing `np.sum(error**2, axis=0)` with `np.einsum("i...,i...->...", error, error)`, we skip temporary array allocation and achieve a measurable speedup (around ~25% faster for typical OCP tracking sizes), while leaving the overall logic intact.
**Action:** Replace `np.sum(error**2, axis=0)` with `np.einsum("i...,i...->...", error, error)` in tight computation paths where a 2D array is reduced over its first axis.
## 2024-05-18 - Optimized Bounding Sphere Radius Calculation
**Learning:** In 3D rendering and physical modeling tools (e.g., `src/tools/capture_rig/model_frame_source.py`), determining the maximum bounding radius typically involves computing Euclidean distance of all points to a center. Calling `np.linalg.norm(..., axis=1).max()` performs intermediate square-root operations and temporary array allocations. Precomputing the distance vectors and extracting the max norm directly using `np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff)))` results in an efficient, ~2.7x faster computation.
**Action:** Replace `np.linalg.norm(diff, axis=1).max()` with `np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff)))` whenever maximum point distances are required.
## 2026-09-10 - Bioptim BoundsList Tuple Assignment Creates 3-Element Lists
**Learning:** Assigning a tuple to `bioptim.BoundsList` via `parameter_bounds[name] = (np.array([min]), np.array([max]))` (as opposed to using the `.add()` method with interpolation types) inadvertently creates bounds arrays of shape `(1, 3)` due to bioptim's default 3-node interpolation. This causes broadcast errors during OCP initialization when bounds vectors are collapsed into a single column array `(N, 1)`.
**Action:** Use `parameter_bounds.add(name, min_bound=np.array([[val]]), max_bound=np.array([[val]]), interpolation=biopt.InterpolationType.CONSTANT)` and explicitly provide 2D column arrays (e.g., `[[val]]`) for parameters instead of tuple assignment.
## 2024-05-19 - Optimize Norm Calculation in JCS
**Learning:** Using `np.linalg.norm(..., axis=-1)` to calculate vector magnitudes in multidimensional joint coordinate system arrays incurs internal dispatch overhead and temporary allocations. By replacing it with `np.sqrt(np.einsum("...i,...i->...", cross, cross))`, we skip the temporary array allocation and achieve a ~2.7x speedup for typical array sizes used in biomechanics processing pipelines.
**Action:** Replace `np.linalg.norm(cross, axis=-1)` with `np.sqrt(np.einsum("...i,...i->...", cross, cross))` in `src/shared/python/biomechanics/joint_conventions.py`.
## 2024-05-19 - Fast Iterables Summation over Axis
**Learning:** Using `np.sum()` on a sequence of NumPy arrays created via functions like `np.stack` or directly as a list introduces overhead because NumPy must first implicitly or explicitly allocate a new array. When reducing a sequence of arrays along an axis (e.g., `np.sum(np.stack(tuple(masks.values())), axis=0)`), explicitly converting the sequence to an array first before calling `.sum(axis=0)` (e.g., `np.asarray(list(masks.values())).sum(axis=0)`) is much faster (~2.2x). It avoids the extra overhead of `np.stack`.
**Action:** Replace `np.sum(np.stack(...), axis=0)` with `np.asarray(...).sum(axis=0)` when summing a collection of arrays along an axis.

## 2024-05-20 - [Optimize RMS and Max Norm Calculation]
**Learning:** Using `np.linalg.norm(..., axis=1)` to compute array magnitudes followed by another power operation like `distances**2` creates unnecessary intermediate arrays and performs redundant square root operations. By replacing it directly with `sq_distances = np.einsum('ij,ij->i', diff, diff)`, we avoid `np.linalg.norm` dispatch overhead and temporary allocations. We can then compute both the RMS and max directly from the squared distances (`np.sqrt(np.mean(sq_distances))` and `np.sqrt(np.max(sq_distances))`), which is ~10-15% faster for typical trajectory sizes.
**Action:** Replace `distances = np.linalg.norm(prediction - target, axis=1)` with `sq_distances = np.einsum('ij,ij->i', diff, diff)` when only scalar reductions (like mean or max of the norms) are needed, to optimize computation in validation loops.
## 2026-09-11 - [Optimize Euclidean Distance in 2D Array]
**Learning:** Using `np.sqrt(np.einsum('ij,ij->i', diff, diff))` is significantly faster than `np.linalg.norm(..., axis=1)` for multi-dimensional distance calculations since it bypasses the overhead of np.linalg.norm which internally does checks and allocations.
**Action:** Replace `np.linalg.norm(selected - reference.position_m, axis=1)` with the precalculated diff array and einsum in `src/motion_capture/coaching/measurements.py`.

## 2024-03-22 - [Optimization: Small 1D Array Norm Calculation]
**Learning:** For small 1D NumPy arrays (like 3D vectors), `np.sqrt(ndarray.dot(ndarray))` is ~2x faster than `np.linalg.norm(ndarray)`. It is also faster than `np.sqrt(np.einsum('i,i->', arr, arr))` which is optimized for multidimensional arrays.
**Action:** Replace `np.linalg.norm()` with `np.sqrt(ndarray.dot(ndarray))` when calculating the magnitude of single 1D arrays.

## 2026-11-20 - Optimize Np.Linalg.Norm in Bundle Adjustment
**Learning:** In bundle adjustment (and other multidimensional vector math operations) replacing `np.linalg.norm(arr, axis=1)` with `np.sqrt(np.einsum('ij,ij->i', arr, arr))` reduces NumPy overhead by avoiding temporary array allocations. However, `np.linalg.norm` creates temporary variables implicitly. For 2D matrices where calculating differences `diff = a - b` happens before norming, `np.sqrt(np.einsum('ij,ij->i', diff, diff))` is significantly faster (~35% speedup) because it avoids those allocations inside `norm(..., axis=1)`.
**Action:** Always replace `np.linalg.norm(diff, axis=1)` with pre-calculated differences and `np.sqrt(np.einsum('ij,ij->i', diff, diff))` for performance-critical path routines like physics and bundle adjustment in NumPy arrays with >1 dimension.

## 2026-09-12 - [Optimize Euclidean Distance in 2D Array]
**Learning:** Using `np.sqrt(np.einsum('ij,ij->i', diff, diff))` is significantly faster than `np.linalg.norm(..., axis=1)` for multi-dimensional distance calculations since it bypasses the overhead of np.linalg.norm which internally does checks and allocations. Using `np.einsum('ij,ij->i', diff, diff)` directly to compute squared distances allows for further optimizations when calculating RMS or max distance, saving square root evaluations.
**Action:** Replace `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum('ij,ij->i', diff, diff))` or `np.einsum('ij,ij->i', diff, diff)` in `src/shared/python/biomechanics/joint_conventions.py` and `src/shared/python/biomechanics/golf_trajectory.py`.

## 2024-05-20 - [Optimize RMS and Max Norm Calculation With Einsum]
**Learning:** Using `np.linalg.norm(..., axis=1)` to compute array magnitudes followed by another power operation like `distances**2` creates unnecessary intermediate arrays and performs redundant square root operations. By replacing it directly with `sq_distances = np.einsum('ij,ij->i', diff_arr, diff_arr)`, we avoid `np.linalg.norm` dispatch overhead and temporary allocations. We can then compute both the RMS and max directly from the squared distances (`np.sqrt(np.mean(sq_distances))` and `np.sqrt(np.max(sq_distances))`), which is ~10-15% faster.
**Action:** Replace `distances = np.linalg.norm(prediction - target, axis=1)` with `sq_distances = np.einsum('ij,ij->i', diff_arr, diff_arr)` when scalar reductions (like mean or max) are subsequently needed on the distances, to optimize computation.
## 2024-05-20 - [Optimize Euclidean Distance for 1D Arrays]
**Learning:** To optimize element-wise Euclidean distance calculations for NumPy arrays, replace `np.sqrt(a**2 + b**2)` with `np.hypot(a, b)`. When reducing the array, replacing `np.sum(np.sqrt(a**2 + b**2))` with `np.hypot(a, b).sum()` is surprisingly ~2.3x faster for 1D arrays, as it avoids temporary allocations and bypasses the overhead of the global `np.sum()`.
**Action:** Replace `np.sum(np.sqrt(a**2 + b**2))` with `np.hypot(a, b).sum()` for 1D arrays.

## 2024-05-20 - Fast Small Array Magnitude
**Learning:** `np.linalg.norm()` is known to be relatively slow for small arrays (like 3D vectors) due to internal overhead and instance checks. Built-in `math.sqrt(np.vdot(arr, arr))` provides a significant speedup by bypassing `np.linalg.norm` overhead while leveraging the fast C-level `np.vdot`. For simple length threshold checks like `np.linalg.norm(diff) <= 1e-9`, squaring the threshold (`np.vdot(diff, diff) <= 1e-18`) completely skips the square root.
**Action:** When computing vector norms for small 1D arrays, replace `np.linalg.norm(v)` with `math.sqrt(np.vdot(v, v))`. For simple threshold checks, replace `np.linalg.norm(v) < threshold` with `np.vdot(v, v) < threshold**2`.

## 2024-05-20 - [Optimize Norm Calculation in Fit2d]
**Learning:** Using `np.linalg.norm(..., axis=3)` to compute array magnitudes for 4D arrays (like those in `fit2d.py` for shape `(T, V, L, 2)`) incurs significant overhead due to temporary array allocations. By replacing it directly with `d = np.sqrt(np.einsum('ijkl,ijkl->ijk', diff, diff))`, we avoid `np.linalg.norm` dispatch overhead and temporary allocations.
**Action:** Replace `np.linalg.norm(..., axis=3)` with `np.sqrt(np.einsum('ijkl,ijkl->ijk', diff, diff))` when scalar reductions are needed, to optimize computation.

## 2026-09-14 - Math.Sqrt(Np.Dot) Optimization
**Learning:** For small 1D NumPy arrays (e.g., 3D vectors), `math.sqrt(array.dot(array))` is significantly faster (~2.5x) than `np.linalg.norm(array)` because it bypasses NumPy's internal dispatching and instance checks. This is safe to use where array inputs are known to be small 1D vectors.
**Action:** Replace `float(np.linalg.norm(array))` with `float(math.sqrt(array.dot(array)))` in tight loops or where small 1D vector magnitudes are calculated frequently.

## 2025-05-19 - Vector Magnitude Calculation
**Learning:** `np.sqrt(np.einsum("ij,ij->i", v, v))` is significantly faster (~2.5x) than `np.linalg.norm(v, axis=1)` for multidimensional arrays in tight loops.
**Action:** Use `np.sqrt(np.einsum("ij,ij->i", v, v))` instead of `np.linalg.norm(v, axis=1)` for performance optimizations when calculating vector magnitudes along an axis.

## 2026-09-17 - [Optimization: Replace Np.Linalg.Norm With Math.Sqrt(Dot)]
**Learning:** For small 1D NumPy arrays (e.g. 3D vectors) in tight physics calculation loops, `np.linalg.norm` adds substantial Python dispatch and internal instance checking overhead. Built-in `math.sqrt(np.vdot(arr, arr))` or `math.sqrt(arr.dot(arr))` avoids this overhead entirely, yielding significant performance gains (~2.5x speedup) while being safe, domain-correct, and functionally equivalent.
**Action:** Always replace `float(np.linalg.norm(array))` with `float(math.sqrt(array.dot(array)))` where array sizes are small and statically known.

## 2026-09-16 - Safe SPEC.md Modification Pattern Update
**Learning:** Even using a pattern like `line.startswith('| YYYY-MM-DD |')` may fail the `repo-structure-gates` tests because older rows, like `| Date | PR | Summary |`, might not be found. We must explicitly search for the header format in the specific `## 12. Change Log` section and insert below the actual table header separator (e.g. `| --- | --- | --- |`).
**Action:** When updating `SPEC.md` programmatically, use a script that correctly finds the header separator in the proper section and inserts the new row. Additionally, the unit tests inside `repo_hygiene` like `test_spec_changelog_integrity.py` are a great way to verify the file was updated without breaking the parser rules.

## 2026-09-16 - [Small Array Norm Calculation]
**Learning:** For small 1D array norm calculation, math.sqrt(np.vdot(array, array)) is significantly faster than np.linalg.norm. Avoid using math.hypot(*array) because it causes test regressions in some contexts despite being slightly faster in isolated testing.
**Action:** Replace np.linalg.norm with math.sqrt(np.vdot) for small 3D vectors when safe, ensuring no regressions.

## 2024-05-21 - [Optimize Norm Calculation in Motion Retargeting]
**Learning:** In the motion capture retargeting pipeline (e.g., `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_mocap_retargeting.py`), calling `np.linalg.norm(pos_error)` on small 1D arrays (like 3D position errors) incurs significant overhead due to NumPy's internal dispatching and instance checks. Replacing it with `math.sqrt(pos_error.dot(pos_error))` bypasses this overhead and is significantly faster (~2.5x).
**Action:** Replace `np.linalg.norm(pos_error)` with `math.sqrt(pos_error.dot(pos_error))` for small 1D array magnitude calculations where possible.

## 2026-09-20 - Inline Operations in Mathematical Substitutions
**Learning:** Using Python walrus operators `:=` inline as arguments to mathematical functions (like `np.einsum('...', diff := (a-b), diff)`) violates formatting and stylistic standards and creates syntax errors. Even if technically valid in modern Python, it fails strict linters, drops code readability (which is explicitly against the boundaries), and leads to unexpected CI failures.
**Action:** When refactoring calculations to avoid intermediate arrays by reusing terms, always assign the term on a separate line (e.g., `diff = a - b`) before invoking the substitution function like `np.einsum('...', diff, diff)`.

## 2026-09-20 - SPEC.md Row Constraints
**Learning:** `SPEC.md` requires that changelog insertions use either an actual GitHub PR number (e.g., `#1234`) or `n/a` for the second column. Using a placeholder like `#<pr>` will cause `scripts/ci/check_spec_changelog_duplicates.py` to raise a `row contract violated` error.
**Action:** Always use `n/a` in the second column of the `SPEC.md` changelog when modifying it during offline testing or before a pull request number is assigned.

## 2026-09-20 - Committing Temporary Files
**Learning:** Generating utility scripts (like `patch_all.py` or `run_tests_opensim.sh`) and then tracking them into the branch breaks the `repo-structure-gates` check.
**Action:** Always delete shell scripts and Python utility files used to modify the repo before checking `git status` and invoking `git commit`. Use `git ls-files --others --exclude-standard` to verify the working tree is clean.

## 2024-05-21 - [Optimize Terminal Norm Calculation in RL]
**Learning:** In reinforcement learning reward and metric calculations (e.g. `src/reinforcement_learning/trajectory_funnel_benchmark.py`), calculating the distance between the final state and the reference state using `np.linalg.norm(states[-1] - reference[-1])` incurs overhead. Replacing it with `math.sqrt(np.vdot(diff, diff))` avoids intermediate array allocations and NumPy dispatch overhead for small 1D state arrays.
**Action:** Replace `np.linalg.norm(states[-1] - reference[-1])` with pre-calculated differences and `math.sqrt(np.vdot(diff, diff))` for terminal state error calculations.

## 2026-09-22 - Optimize Shaft Model Apply Load Norm
**Learning:** Even in shaft modal deformation applying load, calculating modal force using `np.linalg.norm(force)` on small 3D vectors takes significant dispatch overhead.
**Action:** Replace `np.linalg.norm(force)` with `math.sqrt(np.vdot(force, force))` for single 3D vector norms in `_shaft_model.py`.
