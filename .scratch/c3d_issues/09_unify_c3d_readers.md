# refactor(c3d): consolidate three duplicate C3D readers into one canonical module

## Why

There are currently **three** copies of `c3d_reader.py` in the working tree, each subtly different:

1. **Canonical** (per AGENTS.md): `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` (+ helpers in same dir: `_c3d_io.py`, `_c3d_models.py`, `_c3d_markers.py`, `_c3d_analog.py`).
2. **Legacy engine copy**: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py` (+ underscore helpers in same dir). Sideloaded at runtime via `importlib` from `src/shared/python/motion_matching/loaders/c3d.py:52` — see the `_import_c3d_reader()` function. Ugly.
3. **Vendor copy**: `vendor/ud-tools/src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` (vendored from `ud-tools` per the cross-repo dependency contract).

Each has its own `validate_export_path` etc., and they have drifted (different defensive duplicates, different log messages). This issue eliminates the duplication.

The legacy copy also contains a bot artefact: every method opens with `if not (file_path is not None): raise ValueError("file_path must be provided")` — duplicated twice (lines 58–61 of the canonical file and the equivalent in the legacy file). This is dead code that needs to go.

## What to do

1. **Adopt the canonical reader as the single source of truth**: `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py`.
2. **Replace the legacy engine copy** with a thin re-export shim:

```python
# src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py
"""Backwards-compatible shim. Use the canonical reader instead."""
from src.shared.python.upstream_drift_tools.lab.bio.c3d_reader import *  # noqa: F401,F403
```

3. **Update the motion-matching loader** (`src/shared/python/motion_matching/loaders/c3d.py`) to import the canonical reader directly — drop `_import_c3d_reader()` entirely.
4. **Delete the duplicated `if not (file_path is not None)` precondition pairs** in the canonical reader (every public method opens with two identical checks; one suffices, written via a guard helper or a precondition decorator).
5. **Vendor copy**: leave alone for one release; document in `vendor/ud-tools/README.md` that the vendored copy will be removed once the cross-repo dependency contract migrates the canonical path. Add a smoke test that `vendor/ud-tools/src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` and the in-tree canonical file have the same SHA256, to detect drift.

## Generic naming

The canonical reader is already source-neutral — no rename here. Just dedup.

## Acceptance criteria

- [ ] Only one definition of `C3DDataReader`; the engine path re-exports.
- [ ] `_import_c3d_reader()` and the `_C3D_READER_RELATIVE` path search are gone.
- [ ] Defensive-duplicate `if not (...)` pairs removed (audit with `git grep -E 'if not \(.* is not None\)' src/shared/python/upstream_drift_tools/lab/bio/`).
- [ ] All in-tree consumers import from the canonical path. Audit with:
  - `git grep -E "from .*c3d_reader import|import .*c3d_reader"` returns only canonical path or shim.
- [ ] Vendor-copy SHA256 sentinel test guarding drift.
- [ ] Existing C3D unit + integration tests still pass.
- [ ] No print, no TODO without an issue.

## Files touched

- Edit: `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` (clean defensive dups)
- Replace with shim: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py` and its `_c3d_*.py` siblings
- Edit: `src/shared/python/motion_matching/loaders/c3d.py` (use canonical import)
- Edit: `vendor/ud-tools/README.md`
- New: `tests/integration/test_vendor_c3d_drift.py` (sha256 sentinel)

## Sequencing

Lands before / in parallel with the body C3D loader issue. The body loader is written against the canonical reader, so this cleanup unblocks the import path on day one.
