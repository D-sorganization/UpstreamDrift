# Performance Issues Report

**Golf Modeling Suite**
**Date**: January 14, 2026
**Analyst**: Claude (Automated Analysis)

---

## Executive Summary

This report documents **20 performance anti-patterns** identified in the Golf Modeling Suite codebase. Issues are categorized by severity and type, with specific file locations, code examples, and recommended fixes.

### Impact Overview

| Severity | Count | Potential Impact |
|----------|-------|------------------|
| Critical | 4 | Memory exhaustion, 100-1000x slowdown |
| High | 5 | 10-50x slowdown |
| Medium | 7 | 2-10x slowdown |
| Low | 4 | Minor inefficiency |

---

## Critical Issues

### 1. Memory Leak - Unbounded Task Dictionary

**Location**: `api/server.py:158`

**Problem**: Background task results are stored in a global dictionary with no cleanup mechanism:

```python
active_tasks: dict[str, Any] = {}  # No TTL, no cleanup, no size limit
```

**Impact**: Long-running API servers will exhaust memory as completed tasks accumulate indefinitely. In production with continuous usage, this could cause OOM crashes within days.

**Recommendation**:
- Implement TTL-based cleanup (remove tasks older than 1 hour)
- Add maximum size limit with LRU eviction
- Consider Redis or similar for production deployments

---

### 2. N+1 API Key Verification (Full Table Scan)

**Location**: `api/auth/dependencies.py:75-82`

**Problem**: API key verification loads ALL active keys and iterates with bcrypt:

```python
# Loads entire table into memory
active_keys = db.query(APIKey).filter(APIKey.is_active).all()

for key_candidate in active_keys:
    # Bcrypt verify is intentionally slow (~100ms each)
    if security_manager.verify_api_key(api_key, str(key_candidate.key_hash)):
        api_key_record = key_candidate
        break
```

**Impact**: O(n) bcrypt operations where n = total active API keys. With 1000 users having 3 keys each, worst case is 3000 bcrypt verifications (~5 minutes per auth request).

**Recommendation**:
- Store a fast-hashable prefix (first 8 chars SHA256) for filtering
- Query: `WHERE prefix_hash = ? AND is_active = true`
- Reduces bcrypt calls from O(n) to O(1) average case

---

### 3. N+1 Engine Calls in Recorder

**Location**: `shared/python/dashboard/recorder.py:300-440`

**Problem**: The `record_step()` method makes 3-7+ separate physics engine calls per simulation frame:

```python
# Multiple separate engine calls per frame
q, v = self.engine.get_state()        # Call 1
t = self.engine.get_time()            # Call 2
M = self.engine.compute_mass_matrix() # Call 3 - O(n³) operation!
self.engine.compute_ztcf(q, v)        # Call 4 (if enabled)
self.engine.compute_zvcf(q)           # Call 5 (if enabled)
```

**Impact**: A 1000-frame simulation results in 7,000+ expensive engine calls. Each mass matrix computation is O(n³) in joint count.

**Recommendation**:
- Create `get_full_state()` batch method returning `(q, v, t, M)` tuple
- Cache mass matrix when configuration hasn't changed
- Implement `compute_analysis_batch()` for ZTCF/ZVCF

---

### 4. Dynamic Time Warping - Pure Python O(n×m) Loop

**Location**: `shared/python/signal_processing.py:410-454`

**Problem**: DTW algorithm uses nested Python loops instead of vectorized/compiled code:

```python
for i in range(1, n + 1):
    for j in range(j_start, j_end):
        cost = (series1[i - 1] - series2[j - 1]) ** 2
        last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], ...)
        dtw_matrix[i, j] = cost + last_min
```

**Impact**: For 1000-sample time series, this executes 1M+ Python loop iterations. Python loops are ~100x slower than compiled code.

**Recommendation**:
- Replace with `fastdtw` library (pip install fastdtw)
- Or add `@numba.jit(nopython=True)` decorator
- Expected speedup: 100x

---

## High Severity Issues

### 5. Time Lag Matrix - O(n²) Cross-Correlation

**Location**: `shared/python/statistical_analysis.py:2685-2691`

**Problem**: Time lag computation between joints uses nested loops without parallelization:

```python
for i in range(n_joints):
    for j in range(i + 1, n_joints):
        lag = signal_processing.compute_time_shift(data[:, i], data[:, j], ...)
        lag_matrix[i, j] = lag
```

**Impact**: For 30 joints, this computes 435 FFT-based cross-correlations sequentially.

**Recommendation**:
- Use `concurrent.futures.ThreadPoolExecutor` for parallel computation
- Expected speedup: 4-8x (depending on CPU cores)

---

### 6. N+1 Query in Migration Script

**Location**: `scripts/migrate_api_keys.py:173-175`

**Problem**: User lookup inside loop:

```python
for idx, record in enumerate(api_records, 1):
    user = db.session.query(User).filter(User.id == record.user_id).first()
```

**Impact**: 1000 API keys = 1001 database queries (classic N+1).

**Recommendation**:
```python
user_ids = [r.user_id for r in api_records]
users = db.session.query(User).filter(User.id.in_(user_ids)).all()
user_map = {u.id: u for u in users}
```

---

### 7. Missing Database Connection Pooling

**Location**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py`

**Problem**: Multiple independent `sqlite3.connect()` calls without pooling:
- Line 73, 102, 262, 285, 372, 407, 467

Each method opens/closes connection separately.

**Impact**: Connection overhead on every operation (typically 5-20ms each).

**Recommendation**:
- Implement singleton connection manager
- For SQLite: use `check_same_thread=False` with thread-local connections
- For PostgreSQL: use SQLAlchemy connection pool

---

### 8. Induced Acceleration Analysis Loop

**Location**: `shared/python/dashboard/recorder.py:399-422`

**Problem**: For each tracked torque source, creates separate engine calls:

```python
for src_idx in sources:
    tau_single = np.zeros_like(tau)
    tau_single[src_idx] = tau[src_idx]
    acc = self.engine.compute_forward_dynamics(q, v, tau_single)
```

**Impact**: 10 torque sources × 1000 frames = 10,000 expensive dynamics computations.

**Recommendation**:
- Implement vectorized `compute_induced_accelerations_batch()`
- Process all sources in single matrix operation

---

### 9. Repeated Statistics Queries

**Location**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py:471-497`

**Problem**: `get_statistics()` executes 5+ separate queries:

```python
cursor.execute("SELECT COUNT(*) FROM recordings")           # Query 1
cursor.execute("SELECT AVG(duration) FROM recordings")      # Query 2
cursor.execute("SELECT club_type, COUNT(*) FROM ...")       # Query 3
cursor.execute("SELECT swing_type, COUNT(*) FROM ...")      # Query 4
cursor.execute("SELECT golfer_name, COUNT(*) FROM ...")     # Query 5
```

**Impact**: 5 round trips to database for data that could be fetched in 1-2 queries.

**Recommendation**: Combine into single query with multiple aggregations.

---

## Medium Severity Issues

### 10. CWT Wavelet Not Cached

**Location**: `shared/python/signal_processing.py:218-259`

**Problem**: Creates new wavelet for each frequency scale:

```python
for i, s in enumerate(scales):
    wavelet = morlet_func(M, s, w=w0)  # Recreated each iteration
```

**Recommendation**: Use `@functools.lru_cache` for wavelet generation.

---

### 11. Buffer Over-Allocation

**Location**: `shared/python/dashboard/recorder.py:21-30`

**Problem**: Pre-allocates 100,000 samples regardless of actual recording length:

```python
def __init__(self, engine: PhysicsEngine, max_samples: int = 100000):
```

**Impact**: ~100+ MB pre-allocation even for short recordings.

**Recommendation**: Implement dynamic buffer sizing with growth factor.

---

### 12. Blocking time.sleep() Operations

**Locations**:
- `scripts/populate_refactor_issues.py:94` - `time.sleep(1.0)`
- `launchers/golf_launcher.py:2555` - `time.sleep(3)`
- `engines/physics_engines/pinocchio/python/pinocchio_golf/coppelia_bridge.py:98,138`

**Problem**: Blocking operations that could use async/await.

**Recommendation**: Convert to `asyncio.sleep()` in async contexts.

---

### 13. String Concatenation in Loops

**Location**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/grip_modelling_tab.py:332-358`

**Problem**:
```python
mocap_xml += """..."""
equality_xml += "..."
```

**Recommendation**: Use list append with final `''.join()`.

---

### 14. String SQL Query Building

**Location**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py:414-437`

**Problem**:
```python
query += " AND golfer_name LIKE ?"
query += " AND club_type = ?"
query += " AND swing_type = ?"
```

**Recommendation**: Use query builder pattern or list with join.

---

### 15. Recurrence Matrix - O(n²) Space

**Location**: `shared/python/statistical_analysis.py:1795-1796`

**Problem**:
```python
dists = pdist(normalized_state, metric=metric)
dist_matrix = squareform(dists)  # Full matrix materialization
```

**Impact**: 1000 samples = 1M element matrix (~8MB).

**Recommendation**: Use sparse matrix for threshold-based recurrence.

---

### 16. Synchronous File I/O

**Location**: `shared/python/output_manager.py:168-233`

**Problem**: All file operations block main thread for CSV, JSON, HDF5 exports.

**Recommendation**: Use `asyncio` or `concurrent.futures` for non-blocking I/O.

---

## Low Severity Issues

### 17. Unnecessary List Conversions

**Locations**:
- `tools/urdf_generator/model_library.py:440-441`
- `launchers/golf_launcher.py:1128`
- `shared/python/optimization/swing_optimizer.py:397`

**Problem**:
```python
for model_id in list(self.model_cards.keys()):  # Unnecessary list()
```

**Recommendation**: Iterate directly over dict views.

---

### 18. Unused Cache Declarations

**Location**: `shared/python/statistical_analysis.py:256-257`

**Problem**:
```python
self._work_metrics_cache: dict[int, dict[str, float]] = {}  # Never used
self._power_metrics_cache: dict[int, JointPowerMetrics] = {}  # Never used
```

**Recommendation**: Remove or implement caching logic.

---

### 19. Range-to-List Conversion

**Location**: `shared/python/plotting.py:274, 868, 1011, 1054`

**Problem**:
```python
joint_indices = list(range(positions.shape[1]))  # Unnecessary
```

**Recommendation**: Use range directly or numpy indexing.

---

### 20. Array-to-List for JSON

**Location**: `shared/python/ellipsoid_visualization.py:228-230`

**Problem**:
```python
"center": ellipsoid.center.tolist()
```

**Recommendation**: Use custom JSON encoder for numpy arrays.

---

## Priority Action Matrix

| Priority | Issue | Est. Speedup | Effort | Risk |
|----------|-------|--------------|--------|------|
| 1 | Fix memory leak (#1) | Prevents OOM | Low | Low |
| 2 | API key prefix indexing (#2) | 100-1000x | Medium | Medium |
| 3 | Batch engine calls (#3) | 10-50x | Medium | Low |
| 4 | Use optimized DTW (#4) | 100x | Low | Low |
| 5 | Parallelize lag matrix (#5) | 4-8x | Low | Low |
| 6 | Fix N+1 in migration (#6) | Linear | Low | Low |
| 7 | DB connection pooling (#7) | 2-5x | Low | Low |
| 8 | Batch induced accel (#8) | 10x | Medium | Low |
| 9 | Combine stats queries (#9) | 5x | Low | Low |
| 10 | Cache wavelets (#10) | 2-5x | Low | Low |

---

## Files Analyzed

| File | Issues Found |
|------|--------------|
| `api/server.py` | 1 |
| `api/auth/dependencies.py` | 1 |
| `shared/python/dashboard/recorder.py` | 3 |
| `shared/python/signal_processing.py` | 2 |
| `shared/python/statistical_analysis.py` | 3 |
| `shared/python/output_manager.py` | 1 |
| `shared/python/plotting.py` | 1 |
| `shared/python/ellipsoid_visualization.py` | 1 |
| `scripts/migrate_api_keys.py` | 1 |
| `engines/.../recording_library.py` | 3 |
| `engines/.../grip_modelling_tab.py` | 1 |
| `tools/urdf_generator/model_library.py` | 1 |
| `launchers/golf_launcher.py` | 1 |

---

## Conclusion

The Golf Modeling Suite has several performance bottlenecks that should be addressed before scaling to production workloads. The most critical issues are:

1. **Memory leak** in the API server that will cause crashes
2. **O(n) bcrypt verification** making API auth unusably slow at scale
3. **N+1 engine calls** causing 10-50x slowdown in simulations
4. **Pure Python DTW** that's 100x slower than compiled alternatives

Addressing the top 5 issues would provide the highest ROI, with estimated combined speedup of 100x+ for common workflows.
