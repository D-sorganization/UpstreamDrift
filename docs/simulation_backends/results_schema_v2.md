# Unified Results Schema v2

**Implemented by:** CC-4 (#6776)  
**Canonical module:** `src/shared/python/simulation_backends/trace_io.py`  
**Schema version constant:** `SCHEMA_VERSION = "2.0.0"` (in `protocol.py`)

This document is the authoritative reference for the unified HDF5 result
format that covers every simulation backend (ODE, MuJoCo, MuJoCo Warp) and
the BunkerShot3D sand-interaction results. Every file written by `write_trace`
is self-describing and versioned.

---

## Scope

Two formats existed before CC-4:

| Format | Owner | Groups |
|---|---|---|
| `Trace` v1.x | `simulation_backends` | `/t`, `/q`, `/v`, `/u` |
| BunkerShot3D | `bunkershot3d.io.schema` | `/clubhead/<t>/`, `/wrench/<t>/`, `/grains/<t>/` |

**v2 unifies them** by extending `Trace` with optional trajectory and
wrench groups. BunkerShot3D files can be imported via
`read_bunkershot3d_result` and are never used as an internal intermediate.

---

## HDF5 Layout

### Root attributes

| Attribute | Type | Description |
|---|---|---|
| `schema_version` | str | `"MAJOR.MINOR.PATCH"` stamped at write time |
| `backend` | str | Backend name, e.g. `"ode"`, `"mujoco"`, `"bunkershot3d"` |
| `dt` | float | Integration step [s] |
| `kind` | str | `"single"` (Trace) or `"batch"` (BatchTrace) |
| `meta_<key>` | scalar | One attribute per scalar provenance entry |

### Required datasets

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `t` | `(T,)` | float64 | Sample times [s] |
| `q` | `(T, nq)` or `(N, T, nq)` | float64 | Generalised positions [rad] |
| `v` | `(T, nv)` or `(N, T, nv)` | float64 | Generalised velocities [rad/s] |

### Optional datasets (v2+, single Trace only)

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `u` | `(T, nu)` | float64 | Applied controls [N·m]; omitted if passive |
| `torques` | `(T, nu)` | float64 | Joint torques / generalised forces [N·m] |
| `wrench` | `(T, 6)` | float64 | Contact wrench `[fx, fy, fz, tx, ty, tz]` [N, N·m] |
| `markers` | `(T, n_markers, 3)` | float64 | Predicted marker positions [m] |
| `contacts` | `(T, n_contacts, 3)` | float64 | Contact point positions [m] |

Datasets that are `None` are **omitted** from the file; the reader returns
`None` for absent datasets.

---

## BunkerShot3D Profile

BunkerShot3D files use a legacy time-keyed sub-group layout under
`/clubhead/`, `/wrench/`, and `/grains/`. They are **import-only**:
`read_bunkershot3d_result` maps them to the unified schema as follows:

| BunkerShot3D | Trace v2 field |
|---|---|
| `/clubhead/t_<t>/position` (3,) | `markers[:, 0, :]` shape `(T, 1, 3)` |
| `/wrench/t_<t>/force` + `torque` (3,) each | `wrench` columns `[:3]` + `[3:]` |
| `/grains/…` | dropped (not mapped) |
| — | `q`, `v` → empty `(T, 0)` arrays |

BunkerShot3D results are **never** written back in the BunkerShot3D format
by the analysis layer; all downstream analysis reads `Trace` objects.

---

## Versioning Policy

| File major | Accepted | Notes |
|---|---|---|
| `1` | ✓ (auto-migrated) | v1.x files lack optional datasets; all default to `None` |
| `2` | ✓ (current) | Full v2 support |
| other | ✗ | `ValueError` raised |

Minor and patch differences within an accepted major are always accepted
(additive changes only).

---

## Migration API

```python
from src.shared.python.simulation_backends.trace_io import (
    read_trace,             # reads v1 or v2; auto-migrates v1
    migrate_from_v1,        # explicitly reads a v1 file
    read_bunkershot3d_result,  # imports a BunkerShot3D HDF5 file
    write_trace,            # always writes v2
)
```

### Auto-migration example

```python
# Works transparently for both v1 and v2 files:
trace = read_trace("old_v1_file.h5")
assert trace.torques is None   # new fields default to None
```

### Explicit v1 migration

```python
trace = migrate_from_v1("old_v1_file.h5")
# trace.schema_version == "1.0.0" (preserved from file)
```

### BunkerShot3D import

```python
trace = read_bunkershot3d_result("bunker_sim.h5")
# trace.markers: clubhead positions, shape (T, 1, 3)
# trace.wrench:  contact wrench,    shape (T, 6)
# trace.q / v:   empty (T, 0) arrays — no joint states
```

---

## Provenance (pending CC-6)

The `meta_` attribute namespace carries scalar provenance today. When
CC-6 (#6778, `ProvenanceStamp`) merges, a dedicated `/provenance` group
will be written by every `write_trace` call. The flat `meta_` attributes
will remain for backward compatibility.

---

## See also

- `src/shared/python/simulation_backends/protocol.py` — `Trace` dataclass
- `src/shared/python/simulation_backends/trace_io.py` — read/write/migrate
- `src/bunkershot3d/io/schema.py` — legacy BunkerShot3D reader (for direct
  BunkerShot3D access; prefer `read_bunkershot3d_result` for cross-engine
  analysis)
- ADR-0023 — `simulation_backends` architecture
- CC-6 (#6778) — ProvenanceStamp (pending)
- CC-25 (#6798) — WrenchTrace into unified schema (pending)
