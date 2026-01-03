## 2025-02-23 - NumPy Slice Assignment Overhead
**Learning:** In very small arrays (e.g., 6x6), using NumPy slice assignment (e.g., `out[3:6, 3:6] = out[0:3, 0:3]`) can be significantly slower (~1.7x) than manually assigning individual scalar elements (e.g., 5-6 `__setitem__` calls) in Python.
**Action:** For high-frequency, small-matrix operations (like 6x6 spatial transforms), prefer simple scalar assignments or fully vectorised operations over complex slicing if the slice setup overhead dominates. Always benchmark micro-optimizations.
## 2024-05-22 - [Buffer Swap Optimization]
**Learning:** In rigid body dynamics algorithms (CRBA), symmetric matrix accumulation involves repeatedly copying data from a scratch buffer to a persistent state array. Python's tuple unpacking allows swapping buffer references (pointers) instantly, eliminating the O(N) copy operation entirely.
**Action:** Replace 'buffer[:] = scratch' with 'buffer, scratch = scratch, buffer' in iterative algorithms where the buffer state is carried forward.
