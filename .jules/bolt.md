## 2024-05-23 - NumPy Flatten vs Ravel
**Learning:** `np.flatten()` always forces a copy, while `np.ravel()` returns a view if possible. In high-frequency physics loops (like spatial algebra), this copy overhead adds up.
**Action:** Use `np.ravel()` for shape validation/sanitization when a copy is not strictly required.
