## 2026-01-08 - NumPy Reduction Overhead on Small Axis
**Learning:** When calculating min/max for an array with shape `(N, 2)` where N is large (100k+), doing `np.min(data, axis=0)` is significantly slower (~17x) than accessing columns separately `np.min(data[:, 0])`. This counter-intuitive result is likely due to the overhead of the general axis reduction mechanism in NumPy versus the optimized contiguous memory scan for a single slice.

**Action:** For arrays with a very small second dimension (e.g., 2D or 3D points), prefer separate column operations over `axis=0` reduction if performance is critical.
