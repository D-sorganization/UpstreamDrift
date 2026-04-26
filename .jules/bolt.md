## 2026-04-26 - [Optimize small vector norm calculation]
**Learning:** Replacing np.linalg.norm(v) with np.sqrt(np.vdot(v, v)) avoids NumPy reduction overhead for small geometry vectors, yielding roughly a ~1.5x speedup.
**Action:** Prefer np.sqrt(np.vdot(v, v)) instead of np.linalg.norm when operating over arrays in performance-critical hot paths.
