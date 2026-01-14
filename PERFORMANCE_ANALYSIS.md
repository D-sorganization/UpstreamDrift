# Performance Anti-Patterns Analysis Report

**Generated**: 2026-01-14
**Codebase**: Golf Modeling Suite

## Executive Summary

This analysis identifies **15 critical performance issues** across the codebase, including N+1 query-like patterns with physics engine calls, O(n^2) algorithms, memory inefficiencies, and I/O blocking operations.

---

## Critical Issues (Severity: HIGH/CRITICAL)

### 1. Repeated Engine Calls Per Simulation Frame (N+1 Pattern)

**Location**: `shared/python/dashboard/recorder.py:205-289`

**Issue**: The `record_step()` method makes 3-7+ separate engine calls per simulation frame:

```python
# Lines 205-206: Two separate state queries
q, v = self.engine.get_state()        # Call 1
t = self.engine.get_time()            # Call 2

# Line 223: Expensive mass matrix computation
M = self.engine.compute_mass_matrix() # Call 3 - O(n^3) operation!

# Lines 236-289: Additional expensive computations per frame
self.engine.compute_ztcf(q, v)              # If enabled
self.engine.compute_zvcf(q)                  # If enabled
self.engine.compute_drift_acceleration()     # If enabled
self.engine.compute_control_acceleration(tau) # If enabled
```

**Impact**: For a 1000-frame simulation with all features enabled:
- Minimum: 3,000 engine calls
- With analysis: 7,000+ calls
- Mass matrix alone: 1,000 x O(n^3) operations

**Recommendation**:
- Batch state queries into a single `get_full_state()` call
- Cache mass matrix (only changes when q changes significantly)
- Compute analysis metrics post-hoc or at reduced frequency

---

### 2. Induced Acceleration Loop (Multiplied N+1)

**Location**: `shared/python/dashboard/recorder.py:274-289`

**Issue**: For each tracked source, creates a zero vector and makes a separate engine call:

```python
for src_idx in sources:
    tau_single = np.zeros_like(tau)
    tau_single[src_idx] = tau[src_idx]
    self.data["induced_accelerations"][src_idx][idx] = (
        self.engine.compute_control_acceleration(tau_single)
    )
```

**Impact**: 10 sources x 1000 frames = 10,000 expensive compute_control_acceleration calls

**Recommendation**:
- Compute induced accelerations post-hoc using vectorized operations
- Use matrix decomposition to compute all sources in one pass

---

### 3. Time Lag Matrix - O(n^2) Cross-Correlation

**Location**: `shared/python/statistical_analysis.py:2685-2691`

**Issue**: Nested loop computing cross-correlation for all joint pairs:

```python
for i in range(n_joints):
    for j in range(i + 1, n_joints):
        lag = signal_processing.compute_time_shift(
            data[:, i], data[:, j], fs, max_lag=max_lag
        )
```

**Impact**: For 30 joints = 435 cross-correlation computations, each involving FFT operations

**Recommendation**:
- Parallelize using `concurrent.futures` or `joblib`
- Pre-compute FFTs and reuse them across pairs

---

### 4. Dynamic Time Warping - Pure Python O(n*m) Loop

**Location**: `shared/python/signal_processing.py:410-454`

**Issue**: DTW implemented with nested Python loops instead of optimized C/Cython:

```python
for i in range(1, n + 1):
    j_start = max(1, i - w)
    j_end = min(m + 1, i + w + 1)
    for j in range(j_start, j_end):
        cost = (series1[i - 1] - series2[j - 1]) ** 2
        last_min = min(
            dtw_matrix[i - 1, j],
            dtw_matrix[i, j - 1],
            dtw_matrix[i - 1, j - 1],
        )
        dtw_matrix[i, j] = cost + last_min
```

**Impact**: For 1000-sample time series = 1M+ Python loop iterations

**Recommendation**:
- Use `fastdtw` library or `dtw-python` with C backend
- Apply Numba JIT compilation: `@numba.jit(nopython=True)`

---

## Medium Severity Issues

### 5. CWT Per-Scale FFT Without Wavelet Caching

**Location**: `shared/python/signal_processing.py:218-259`

**Issue**: Creates new wavelet for each frequency scale without caching:

```python
for i, s in enumerate(scales):
    M = int(2 * 5 * s + 1)
    wavelet = morlet_func(M, s, w=w0)  # Recreated each iteration
    wavelet_fft = fft.fft(wavelet, n=n_fft)
    # ...
```

**Impact**: 50-100 redundant wavelet computations per CWT call

**Recommendation**: Cache wavelets for common scale values using `functools.lru_cache`

---

### 6. Buffer Over-Allocation

**Location**: `shared/python/dashboard/recorder.py:21-30, 95-170`

**Issue**: Pre-allocates 100,000 samples regardless of actual recording length:

```python
def __init__(self, engine: PhysicsEngine, max_samples: int = 100000) -> None:
    self.max_samples = max_samples
    # ...
    self.data["times"] = np.zeros(self.max_samples)
    self.data["joint_positions"] = np.zeros((self.max_samples, nq))
```

**Impact**: 100,000 samples x 20+ joints x 10+ variables = 100+ MB pre-allocation

**Recommendation**:
- Use dynamic resizing with `numpy.resize()` or list appending with final conversion
- Implement a `trim_buffers()` method called after recording

---

### 7. Recurrence Matrix - O(n^2) Space and Time

**Location**: `shared/python/statistical_analysis.py:1795-1796`

**Issue**: Full distance matrix computation for recurrence plots:

```python
dists = pdist(normalized_state, metric=metric)
dist_matrix = squareform(dists)
```

**Impact**: For 1000 samples = 500K pairwise distances + 1M element matrix

**Recommendation**:
- Use sparse representation for recurrence matrices
- Implement FAN (Fixed Amount of Nearest Neighbors) thresholding
- Consider approximate methods for large datasets

---

### 8. Synchronous File I/O

**Location**: `shared/python/output_manager.py:168-233`

**Issue**: All file operations block the main thread:

```python
if format_type == OutputFormat.CSV:
    df.to_csv(file_path, index=False)
elif format_type == OutputFormat.JSON:
    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=2, default=json_serializer)
elif format_type == OutputFormat.HDF5:
    results.to_hdf(file_path, key="data", mode="w")
```

**Impact**: Large simulations (1GB+) can block for seconds

**Recommendation**:
- Use `asyncio` or `concurrent.futures.ThreadPoolExecutor` for I/O
- Implement background save with callback notification

---

### 9. Post-Hoc Analysis Re-simulation

**Location**: `shared/python/dashboard/recorder.py:404-457`

**Issue**: `compute_analysis_post_hoc()` replays entire trajectory frame-by-frame:

```python
for i in range(n_frames):
    q = qs[i]
    v = vs[i]
    tau = taus[i]

    self.engine.set_state(q, v)
    self.engine.set_control(tau)
    self.engine.forward()

    ztcf = self.engine.compute_ztcf(q, v)
    zvcf = self.engine.compute_zvcf(q)
    drift = self.engine.compute_drift_acceleration()
    ctrl_acc = self.engine.compute_control_acceleration(tau)
```

**Impact**: 4 expensive computations x n_frames = 4000+ engine calls for 1000 frames

**Recommendation**:
- Batch similar computations
- Use vectorized implementations where possible
- Consider computing only for key frames (impact, transition, etc.)

---

## Low Severity Issues

### 10. Swing Phase Detection Linear Search

**Location**: `shared/python/statistical_analysis.py:553-564`

**Issue**: Uses Python for-loop for threshold detection:

```python
for i in range(1, transition_idx):
    if smoothed_speed[i] > speed_threshold:
        takeaway_idx = i
        break
```

**Recommendation**: Use `np.argmax(smoothed_speed > speed_threshold)` for vectorized search

---

### 11. Unnecessary Array-to-List Conversions

**Location**: `shared/python/ellipsoid_visualization.py:228-230` and others

**Issue**: Converting numpy arrays to lists for JSON serialization:

```python
"center": ellipsoid.center.tolist(),
"radii": ellipsoid.radii.tolist(),
"axes": ellipsoid.axes.tolist(),
```

**Recommendation**: Use custom JSON encoder that handles numpy arrays directly

---

### 12. Range-to-List Conversion

**Location**: `shared/python/plotting.py:274, 868, 1011, 1054`

**Issue**: Creating list from range unnecessarily:

```python
joint_indices = list(range(positions.shape[1]))
```

**Recommendation**: Use `range()` directly as an iterator

---

### 13. Unused Cache Declarations

**Location**: `shared/python/statistical_analysis.py:256-257`

**Issue**: Cache dictionaries declared but never populated:

```python
self._work_metrics_cache: dict[int, dict[str, float]] = {}
self._power_metrics_cache: dict[int, JointPowerMetrics] = {}
```

**Recommendation**: Either implement caching logic or remove dead code

---

### 14. Manual Cache Invalidation Required

**Location**: `shared/python/plotting.py:135, 209-216`

**Issue**: Data cache requires manual invalidation:

```python
self._data_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

def clear_cache(self) -> None:
    self._data_cache.clear()
```

**Recommendation**: Implement automatic invalidation on data source changes or use TTL-based cache

---

### 15. Cross Wavelet Transform Double CWT Computation

**Location**: `shared/python/signal_processing.py:290-291`

**Issue**: Computes CWT twice for XWT:

```python
f1, t1, w1 = compute_cwt(data1, fs, freq_range, num_freqs, w0)
f2, t2, w2 = compute_cwt(data2, fs, freq_range, num_freqs, w0)
```

**Recommendation**: If comparing multiple signals to a reference, cache the reference CWT

---

## Performance Optimization Priority Matrix

| Priority | Issue | Estimated Speedup | Effort |
|----------|-------|-------------------|--------|
| 1 | Batch engine calls (#1, #2) | 10-50x | Medium |
| 2 | Cache mass matrix | 5-20x | Low |
| 3 | Use optimized DTW library (#4) | 100x | Low |
| 4 | Parallelize lag matrix (#3) | 4-8x (cores) | Low |
| 5 | Async file I/O (#8) | Non-blocking | Medium |
| 6 | JIT compile tight loops | 10-100x | Medium |
| 7 | Sparse recurrence matrices (#7) | 10x memory | Medium |
| 8 | Dynamic buffer sizing (#6) | Memory only | Low |

---

## Recommended Immediate Actions

1. **Create `EngineState` batch query** - Single call returns (q, v, t, M) tuple
2. **Add `@functools.lru_cache` to wavelet generation**
3. **Replace DTW with `fastdtw` or Numba-compiled version**
4. **Add `@numba.jit(nopython=True)` to inner loops in signal processing**
5. **Implement `compute_induced_accelerations_batch()` for vectorized computation**

---

## Additional Critical Issues (API/Database Layer)

### 16. Memory Leak - Unbounded Task Dictionary

**Location**: `api/server.py:158`

**Issue**: Background task results stored in global dictionary with no cleanup:

```python
active_tasks: dict[str, Any] = {}  # No TTL, no cleanup, no size limit
```

**Impact**: Long-running API servers will exhaust memory as completed tasks accumulate indefinitely.

**Recommendation**:

- Implement TTL-based cleanup (e.g., remove tasks older than 1 hour)
- Add maximum size limit with LRU eviction
- Consider using Redis or similar for production deployments

---

### 17. N+1 API Key Verification (Full Table Scan)

**Location**: `api/auth/dependencies.py:75-82`

**Issue**: API key verification loads ALL active keys and iterates with bcrypt:

```python
# Loads entire table into memory
active_keys = db.query(APIKey).filter(APIKey.is_active).all()

for key_candidate in active_keys:
    # Bcrypt verify is intentionally slow (~100ms each)
    if security_manager.verify_api_key(api_key, str(key_candidate.key_hash)):
        api_key_record = key_candidate
        break
```

**Impact**: O(n) bcrypt operations where n = total active API keys. With 1000 users having 3 keys each, worst case is 3000 bcrypt verifications.

**Recommendation**:

- Store a fast-hashable prefix (first 8 chars SHA256) for filtering
- Query: `WHERE prefix_hash = ? AND is_active = true` then verify only matches
- This reduces bcrypt calls from O(n) to O(1) average case

---

### 18. N+1 Query in Migration Script

**Location**: `scripts/migrate_api_keys.py:173-175`

**Issue**: User lookup inside loop:

```python
for idx, record in enumerate(api_records, 1):
    user = db.session.query(User).filter(User.id == record.user_id).first()
```

**Impact**: 1000 API keys = 1001 database queries.

**Recommendation**: Use `joinedload` or batch fetch all users first:

```python
user_ids = [r.user_id for r in api_records]
users = db.session.query(User).filter(User.id.in_(user_ids)).all()
user_map = {u.id: u for u in users}
```

---

### 19. Missing Database Connection Pooling

**Location**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py`

**Issue**: Multiple independent `sqlite3.connect()` calls without pooling:

- Line 73, 102, 262, 285, 372, 407, 467

Each method opens/closes connection separately, incurring connection overhead.

**Recommendation**:

- Use a connection pool or singleton connection manager
- For SQLite, consider using `check_same_thread=False` with thread-local connections

---

### 20. Repeated Statistics Queries

**Location**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py:471-497`

**Issue**: `get_statistics()` executes 5+ separate queries that could be combined:

```python
cursor.execute("SELECT COUNT(*) FROM recordings")           # Query 1
cursor.execute("SELECT AVG(duration) FROM recordings")      # Query 2
cursor.execute("SELECT club_type, COUNT(*) FROM ...")       # Query 3
# etc.
```

**Recommendation**: Combine into single query with multiple aggregations or use subqueries.

---

## Updated Performance Optimization Priority Matrix

| Priority | Issue | Estimated Speedup | Effort |
|----------|-------|-------------------|--------|
| 1 | Fix memory leak in active_tasks (#16) | Prevents OOM | Low |
| 2 | Batch engine calls (#1, #2) | 10-50x | Medium |
| 3 | API key prefix indexing (#17) | 100-1000x | Medium |
| 4 | Use optimized DTW library (#4) | 100x | Low |
| 5 | Cache mass matrix | 5-20x | Low |
| 6 | Parallelize lag matrix (#3) | 4-8x (cores) | Low |
| 7 | DB connection pooling (#19) | 2-5x | Low |
| 8 | Async file I/O (#8) | Non-blocking | Medium |
| 9 | JIT compile tight loops | 10-100x | Medium |
| 10 | Sparse recurrence matrices (#7) | 10x memory | Medium |

---

## Appendix: Files Analyzed

- `shared/python/dashboard/recorder.py`
- `shared/python/statistical_analysis.py`
- `shared/python/signal_processing.py`
- `shared/python/output_manager.py`
- `shared/python/plotting.py`
- `shared/python/ball_flight_physics.py`
- `shared/python/ellipsoid_visualization.py`
- `api/services/simulation_service.py`
- `api/server.py`
- `api/auth/dependencies.py`
- `api/routes/auth.py`
- `scripts/migrate_api_keys.py`
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py`
- `tools/urdf_generator/urdf_builder.py`
- `launchers/golf_launcher.py`
