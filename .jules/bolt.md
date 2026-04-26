## 2026-04-26 - Optimizing Sum of Squares in Trajectory Benchmark
**Learning:** `np.sum(error**2)` allocates a temporary array in memory to store the squared values before summing them, which is slow.
**Action:** Replace `np.sum(error**2)` with `np.vdot(error, error)` on flat/real-numbered arrays. It computes the dot product directly at the C level, yielding up to a ~3-4x performance speedup by avoiding intermediate array allocations.

## 2024-04-26 - Optimize Mean Squared Error calculations
**Learning:** When computing Mean Squared Error (MSE) across arrays, `np.vdot(diff, diff) / diff.size` is significantly faster (~2x) than `np.mean(diff**2)` because it leverages optimized C code and completely bypasses the memory allocation overhead of creating a temporary squared array.
**Action:** Use `np.vdot(diff, diff) / diff.size` instead of `np.mean(diff**2)` when calculating MSE in hot paths.