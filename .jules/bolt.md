## 2024-05-17 - Fast Euclidean Distance in NumPy
**Learning:** For calculating sum of squared differences of 1D arrays (e.g. Euclidean distance computations in nearest neighbor structures like `_tree_index.py`), using `np.sum(diff ** 2)` is an anti-pattern as it creates a temporary intermediate array for the squared differences, leading to memory allocation overhead and slowing down hot loops.
**Action:** Replace `np.sum(diff ** 2)` with `np.vdot(diff, diff)` which performs the dot product directly and avoids the intermediate array allocation, resulting in a ~3.5x speedup for 1D arrays.
