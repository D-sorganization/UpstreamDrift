## 2026-04-26 - Optimizing Sum of Squares in Trajectory Benchmark
**Learning:** `np.sum(error**2)` allocates a temporary array in memory to store the squared values before summing them, which is slow.
**Action:** Replace `np.sum(error**2)` with `np.vdot(error, error)` on flat/real-numbered arrays. It computes the dot product directly at the C level, yielding up to a ~3-4x performance speedup by avoiding intermediate array allocations.
