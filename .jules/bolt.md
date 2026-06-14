## 2026-06-14 - math.hypot for 3D vector magnitudes
**Learning:** For small fixed-size numpy arrays (like 3D velocity vectors), extracting individual components and using `math.hypot(x, y, z)` avoids NumPy's type checking, dispatching, and temporary array allocation overhead.
**Action:** Use explicit component extraction and `math.hypot` instead of `np.linalg.norm` in tight inner loops processing very small arrays, but ensure the array size is rigidly fixed to avoid IndexError.
