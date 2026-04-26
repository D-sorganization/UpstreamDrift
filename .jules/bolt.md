
## 2024-04-26 - Optimize Mean Squared Error calculations
**Learning:** When computing Mean Squared Error (MSE) across arrays, `np.vdot(diff, diff) / diff.size` is significantly faster (~2x) than `np.mean(diff**2)` because it leverages optimized C code and completely bypasses the memory allocation overhead of creating a temporary squared array.
**Action:** Use `np.vdot(diff, diff) / diff.size` instead of `np.mean(diff**2)` when calculating MSE in hot paths.
