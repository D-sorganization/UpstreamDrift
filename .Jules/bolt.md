## 2024-05-24 - Python Overhead in Small Matrix Operations
**Learning:** In high-frequency loops with small matrices (6x6), replacing `A @ B` (which allocates) with `np.matmul(A, B, out=C)` (which avoids allocation) can actually be *slower* due to Python interpreter overhead for the function call and argument parsing. The allocation of small numpy arrays is highly optimized and often negligible compared to the overhead of an extra Python function call.
**Action:** Profile carefully before refactoring mathematical expressions to use `out=` parameters for small arrays. For larger matrices (>100x100), `out=` is still preferred.

## 2024-05-24 - Lazy Mocking in Tests
**Learning:** When mocking heavy dependencies (like `mujoco`) in `conftest.py`, always use `try-except ImportError` blocks. Unconditional mocking breaks integration tests in environments where the dependency IS installed, causing tests to run against mocks instead of the real engine.
**Action:** Wrap global mocks in `conftest.py` with import checks to allow graceful degradation only when necessary.
