
## 2026-04-17 - Optimize squared distance calculation with np.einsum
**Learning:** Explicit element-wise sum of squares using `np.einsum("ij,ij->i", diff, diff)` is ~2x faster than `np.sum(diff ** 2, axis=1)` because it avoids intermediate array allocations from computing the square matrix before summation.
**Action:** Default to using `np.einsum` when computing reductions like sum of squared differences over small inner dimensions.
