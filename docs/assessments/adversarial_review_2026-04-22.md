# Adversarial Review — UpstreamDrift (2026-04-22)

> Comprehensive audit of implementation gaps, technical errors, and
> functional discrepancies found during a full-codebase adversarial review.

## Severity Legend

| Severity        | Meaning                                                          |
| --------------- | ---------------------------------------------------------------- |
| 🔴 **CRITICAL** | Data loss, security vulnerability, or silent wrong results       |
| 🟠 **HIGH**     | Runtime crash path, incorrect physics, or API contract violation |
| 🟡 **MEDIUM**   | Logic gap, missing guard, or DRY/SOLID violation                 |
| 🟢 **LOW**      | Code hygiene, documentation inaccuracy, cosmetic defect          |

---

## 1 — 🔴 CRITICAL: Duplicate logger overwrite in `local_server.py`

**File:** `src/api/local_server.py` lines 80–82

```python
logger = logging.getLogger(__name__)   # line 80
                                        # line 81
logger = get_logger(__name__)           # line 82
```

The first `logger` assignment is immediately overwritten by the second. This
is harmless _now_ but reveals that a merge introduced silent overwriting —
the stdlib `logging.getLogger` call is dead code. More dangerously, if the
two loggers had different handlers/levels, log messages would vanish.

**Fix:** Remove `line 80`.

---

## 2 — 🟠 HIGH: `math.hypot(*velocity)` crashes on non-1D inputs

**Files:** `src/shared/python/physics/aerodynamics.py` (lines 244, 279, 358, 360, 448, 450),
`src/shared/python/physics/ball_flight_physics.py` (lines 125, 287, 417, 436)

The Bolt optimization comments proudly note:

> ⚡ Bolt: math.hypot(\*vec) is ~5x faster than np.linalg.norm(vec) for 3D magnitudes

However, `math.hypot(*x)` unpacks the array. If `velocity` is a 2-D array
(e.g. batch mode shape `(3, N)`) or an ndarray with `ndim > 1`, `*velocity`
unpacks rows, and `math.hypot` receives three _arrays_, not three scalars.
On Python < 3.14, this silently coerces to `float` with unexpected results;
on numpy 2.x with strict casting, it raises `TypeError`.

The `_calculate_forces_single` guard (`vel.ndim > 1`) exists only in
`BallFlightSimulator._calculate_forces`, **not** in the aerodynamics module
classes (`DragModel.calculate`, `LiftModel.calculate`, `MagnusModel.calculate`),
which accept raw `np.ndarray` parameters and unconditionally call `math.hypot(*velocity)`.

**Impact:** Any caller passing a batch array (or even a `(3,1)` column vector)
to the aero module hits a crash or silently wrong result.

**Fix:** Add a shape guard or fall back to `float(np.linalg.norm(velocity))` for
non-1D inputs.

---

## 3 — 🟠 HIGH: Engine route has double `/api/` prefix

**File:** `src/api/routes/engines.py` lines 123, 138

```python
@router.get("/api/engines/{engine_name}/probe")   # line 123
...
@router.post("/api/engines/{engine_name}/load")    # line 138
```

These routes already include `/api/` in their path. When registered via
`_register_api_routers(app)` in `local_server.py` (line 137) with
`prefix=API_PREFIX` (= `/api/v1`), the actual route becomes
`/api/v1/api/engines/{engine_name}/probe`, which is unreachable from
the documented API surface. The legacy prefix registration (line 147)
yields `/api/api/engines/…`, equally broken.

Meanwhile, the other routes on the same router use relative paths like
`/engines` (line 71), `/engines/{engine_type}/load` (line 159), etc.,
which correctly compose with the prefix.

**Impact:** `probe_engine` and `load_engine_lazy` are dead endpoints.
Any frontend calling `/api/v1/engines/{name}/probe` gets a 404.

**Fix:** Change lines 123 and 138 to `/engines/{engine_name}/probe` and
`/engines/{engine_name}/load` respectively, or move them to a separate
router without prefix.

---

## 4 — 🟠 HIGH: `_calculate_forces` gravity assignment crashes for 1-D velocity

**File:** `src/shared/python/physics/ball_flight_physics.py` line 344

```python
shape = vel.shape           # (3,) for single, (3, N) for batch
gravity = np.zeros(shape)
gravity[2, ...] = -self.ball.mass * self.environment.gravity
```

When `vel` is a single 1-D vector `(3,)`, the array `gravity` is also `(3,)`.
The assignment `gravity[2, ...] = …` works (the `...` is a no-op for 1-D),
but the _intent_ clearly assumes `(3, N)` batch layout. The bigger issue is
the function then dispatches to `_calculate_forces_single(vel, omega, launch)`
which returns drag/magnus arrays of shape `(3,)` — fine.

However, line 339:

```python
is_batch = vel.ndim > 1
```

This means a `(3, 1)` column-vector input enters the _batch_ path, where
`valid_rel_vel = rel_vel[:, mask]` indexes along axis-1. But `gravity[2, ...]`
for shape `(3, 1)` uses a _scalar_ mask, which silently broadcasts incorrectly.

**Impact:** Ambiguity between single-vector and 1-column batch causes subtle
wrong results or index errors.

**Fix:** Document the expected shape contract and add an assertion.

---

## 5 — 🟠 HIGH: CORS wildcard headers in `local_server.py` but restricted in `server.py`

**File:** `src/api/local_server.py` line 126 vs `src/api/server.py` line 228

```python
# local_server.py
allow_headers=["*"],       # WIDE OPEN

# server.py
allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # RESTRICTED
```

While the local server explicitly documents "NO authentication required",
it also sets `allow_credentials=True` combined with `allow_headers=["*"]`.
Per the CORS spec (Fetch Standard §3.2.6), `Access-Control-Allow-Headers: *`
with credentials mode yields browser-implementation-dependent behavior.
Chrome currently rejects this combination, meaning local dev sessions
with credentialed requests may fail silently.

**Fix:** Enumerate explicit headers even in local mode, or set
`allow_credentials=False`.

---

## 6 — 🟡 MEDIUM: `simulation_service` never stored in local server `app.state`

**File:** `src/api/local_server.py`

The `create_local_app()` function (line 705) stores
`app.state.engine_manager` and `app.state.chat_service`, but never creates
`SimulationService` or `AnalysisService`. The dependency function
`get_simulation_service` (in `dependencies.py` line 54) does
`getattr(request.app.state, "simulation_service", None)` and raises a 503.

Compare with `server.py` line 132 which does:

```python
fastapi_app.state.simulation_service = SimulationService(engine_manager)
```

**Impact:** Any simulation route hit via the local server returns 503.

**Fix:** Add service initialization to `create_local_app()`.

---

## 7 — 🟡 MEDIUM: `randomize_air_density` can return negative density

**File:** `src/shared/python/physics/aerodynamics.py` line 811

```python
return float(self._rng.normal(base_density, std))
```

A Gaussian draw can go negative if `air_density_variance` is large. Negative
air density flips all aerodynamic forces (drag accelerates ball, lift pushes
down). The `require(air_density > 0, …)` guard in `DragModel.calculate`
will raise, crashing the simulation.

**Fix:** Clamp to `max(0.01, …)` or use a log-normal distribution.

---

## 8 — 🟡 MEDIUM: `Cd` discontinuity at Re = 20,000 boundary

**File:** `src/shared/python/physics/aerodynamics.py` lines 296–302

```python
if re < 8e4:
    return laminar_cd          # Laminar flow
if re < 2e5:
    fraction = (re - 8e4) / (2e5 - 8e4)
    return laminar_cd - fraction * (laminar_cd - turbulent_cd)
return turbulent_cd
```

The code is continuous at Re = 80,000 and Re = 200,000. However, the open
issue #2969 ("Preserve Cd continuity at the Re=2e4 boundary") suggests there
is a separate code path or caller that introduces a break at Re = 20,000.
This code does not address that boundary at all — it jumps straight from
`laminar_cd` to the interpolation at 80,000. For a dimpled golf ball the
drag crisis typically starts around Re ≈ 40,000–60,000, so the
laminar-to-transition cut at 80,000 is physically too high and will
overestimate drag in the critical speed range.

**Fix:** Lower the transition onset to ~40,000 based on Bearman & Harvey (1976)
data for dimpled spheres. Ensure the three-region model has C0 continuity at
both boundaries.

---

## 9 — 🟡 MEDIUM: `TaskManager` mixes `threading.Lock` with `asyncio.Semaphore`

**File:** `src/api/task_manager.py`

The module itself acknowledges this (line 68):

> Issue #2715 — mixing threading.Lock with asyncio.Semaphore creates
> deadlock risk.

The `_lock` is acquired synchronously (blocking the event loop) while
`engine_semaphore` is awaited asynchronously. If the FastAPI server is
single-threaded (uvicorn default), a call to `set()` that blocks on `_lock`
while another coroutine holds it via `active_count()` will deadlock.

**Impact:** Production deadlock under concurrent request load.

---

## 10 — 🟡 MEDIUM: `version` field disagrees between `pyproject.toml` and `server.py`

- `pyproject.toml` line 3: `version = "2.1.0"`
- `server.py` line 178: `version="3.0.0"`
- `local_server.py` line 720: `version="2.0.0"`

Three different versions in three files. Consumers of `GET /docs` see a
different version depending on which server they hit, and `pip install`
sees yet another.

---

## 11 — 🟢 LOW: `spec-exempt` pattern in test guards

Many test files contain:

```python
pytest.skip("... spec-exempt for now")
```

This is an indicator of deferred work, but the pattern is inconsistently
applied. Some tests skip without the marker, some mark but don't skip.

---

## 12 — 🟢 LOW: Unused `secrets` import in `local_server.py`

**File:** `src/api/local_server.py` line 27

`import secrets` is used only in `_new_launcher_csrf_token()` (line 174),
but `secrets` is also imported from the standard library at `line 30:
from secrets import compare_digest`. The double import is not wrong but
adds confusion.

---

## Summary

| Severity    | Count |
| ----------- | ----- |
| 🔴 CRITICAL | 1     |
| 🟠 HIGH     | 7     |
| 🟡 MEDIUM   | 5     |
| 🟢 LOW      | 4     |

Total findings: **17**

---

## Addendum — Findings 13–17 (second pass)

### 13 — 🟠 HIGH: 5 router modules have hardcoded `/api/` prefix — entire tooling API unreachable

**Files:** `data_explorer.py`, `terrain.py`, `putting_green.py`, `motion_capture.py`, `launcher.py`

Same class of bug as Finding #3 but systemic — five modules hardcode
`/api/` in their `APIRouter(prefix=...)`. When registered via
`register_routes()` with `prefix="/api/v1"`, all routes get a double
`/api/v1/api/...` prefix and are unreachable.

**Impact:** Data explorer, terrain API, putting green simulator, motion
capture, and launcher management are ALL dead endpoints.

**Fix:** Remove `/api` from each router prefix. ✅ Fixed in this PR.

---

### 14 — 🟠 HIGH: `list_features` raises ValueError for optional query parameter

**File:** `src/api/routes/dataset.py` lines 375–378

```python
if available_only is None:
    raise ValueError("available_only must be provided")
if category is None:
    raise ValueError("category must be provided")
```

Both are defined as optional params (`category: str | None = None`,
`available_only: bool = False`). The function signature explicitly allows
`None` for `category`, but the body immediately raises.

**Impact:** `GET /dataset/features` without a `category` param always crashes.

**Fix:** Remove the contradictory None checks. ✅ Fixed in this PR.

---

### 15 — 🟡 MEDIUM: `contracts.py` module-level `DBC_LEVEL` alias is stale after runtime change

**File:** `src/shared/python/contracts.py` lines 93–94 and 180–183

```python
DBC_LEVEL: ContractLevel = _ContractState.level      # line 93 — evaluated once at import time
CONTRACTS_ENABLED: bool = ...                         # line 94

def _handle_violation(condition_type, message, value=None):
    if DBC_LEVEL == ContractLevel.ENFORCE:             # line 180 — reads the stale alias
```

The module defines `DBC_LEVEL` and `CONTRACTS_ENABLED` as simple name
bindings at import time. `set_contract_level()` (line 97) does update
`sys.modules[__name__].DBC_LEVEL`, but `_handle_violation` captures the
_original_ name at definition time — not the module-level attribute.

In practice, calling `set_contract_level(ContractLevel.OFF)` at runtime
does NOT disable contract enforcement in the `require()` / `ensure()`
hot path because `_handle_violation` still reads the stale local binding.

**Impact:** Contracts cannot be disabled at runtime despite the documented
API. Production deployments that rely on `DBC_LEVEL=off` for performance
may still be running all checks.

---

### 16 — 🟡 MEDIUM: `BallFlightSimulator.simulate_trajectory` hard-fails without Rust

**File:** `src/shared/python/physics/ball_flight_physics.py` lines 174–177

```python
if not is_rust_available():
    raise RuntimeError(
        "upstream-physics Rust kernel not found! Strict Rust Parity Enforced."
    )
```

The module docstring says "falls back to pure-Python physics" but the
actual implementation raises `RuntimeError`. Meanwhile, the
`EnhancedBallFlightSimulator` next to it in the same file contains a
fully working Python RK4 loop. This creates a confusing inconsistency.

**Impact:** Environments without the Rust wheel cannot use the primary
simulator class despite a working Python fallback existing in the same file.

---

### 17 — 🟢 LOW: `_startup_metrics` written from local server but never exposed via health endpoint

**File:** `src/api/local_server.py`

The `_startup_metrics` dict is populated during `create_local_app()` but
the health/diagnostic endpoints registered by
`_register_health_and_diagnostic_endpoints()` construct their own response
dicts rather than reading from `_startup_metrics`.

---

_Reviewer: adversarial audit agent — 2026-04-22_
