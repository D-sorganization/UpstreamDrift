## 2025-02-23 - NumPy Slice Assignment Overhead
**Learning:** In very small arrays (e.g., 6x6), using NumPy slice assignment (e.g., `out[3:6, 3:6] = out[0:3, 0:3]`) can be significantly slower (~1.7x) than manually assigning individual scalar elements (e.g., 5-6 `__setitem__` calls) in Python.
**Action:** For high-frequency, small-matrix operations (like 6x6 spatial transforms), prefer simple scalar assignments or fully vectorised operations over complex slicing if the slice setup overhead dominates. Always benchmark micro-optimizations.
