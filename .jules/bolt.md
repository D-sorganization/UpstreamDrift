## 2024-06-10 - [Optimize 1D Array Norm Calculation]
**Learning:** `np.linalg.norm` has significant dispatch and object-creation overhead for very small 1D arrays (like quaternions). `math.hypot` (which supports N arguments in Python 3.8+) operates directly on the floats and is measurably faster (~4.5x) for tiny vectors.
**Action:** Use `math.hypot(v[0], v[1], ...)` instead of `np.linalg.norm(v)` when calculating the magnitude of small fixed-size arrays where performance matters.
## 2024-06-13 - [Optimize frequent DataFrame element access in loops]
**Learning:** Using `df.iloc[i]` frequently inside loops (like plotting or processing loops) creates new Series objects and incurs significant pandas dispatch overhead. Converting whole columns to NumPy arrays via `.values` before the loop and indexing them is drastically faster (over 50x in benchmarks).
**Action:** When accessing single elements of multiple columns in a loop, extract the columns as `.values` (NumPy arrays) outside the loop rather than repeatedly using `.iloc`.
