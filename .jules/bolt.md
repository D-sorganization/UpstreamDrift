## 2024-06-10 - [Optimize 1D Array Norm Calculation]
**Learning:** `np.linalg.norm` has significant dispatch and object-creation overhead for very small 1D arrays (like quaternions). `math.hypot` (which supports N arguments in Python 3.8+) operates directly on the floats and is measurably faster (~4.5x) for tiny vectors.
**Action:** Use `math.hypot(v[0], v[1], ...)` instead of `np.linalg.norm(v)` when calculating the magnitude of small fixed-size arrays where performance matters.
