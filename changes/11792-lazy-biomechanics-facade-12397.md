---
issue: 11792
summary: "Isolate shared muscle kernel imports with a compatible lazy biomechanics facade (#12397)."
dl_state: "in_progress"
next_step: "Build a separately pinned replay worker with complete execution provenance; validate force, activation, trajectory, and native replay behavior before any qualification claim."
branch: "feat/12397-lazy-biomechanics-facade"
---

The import-organization slice for child issue #12397 replaces eager package
initialization in `src/shared/python/biomechanics/__init__.py` with a
backward-compatible lazy facade patterned after `src/shared/python/motion_matching`.
Importing the package or core Hill/activation/equilibrium kernels in a fresh
interpreter process isolates cleanly without eagerly importing unrelated modules,
GUI/viewer frameworks (PyQt6, Matplotlib), analysis pipelines, or optional backends
(OpenSim, bunkershot3d).

Unconditional exports are resolved on demand via `__getattr__` and cached on the module,
preserving canonical source submodule object identity. Optional groups resolve paired
symbols on demand, preserve legacy `ImportError -> None` semantics, and propagate
non-ImportError exceptions without swallowing. The dynamic `__all__` property on
`_BiomechanicsModule` preserves the legacy ordered export sequence and import-star parity
(`_BASE_ALL` followed by available optional symbols). `dir()` exposes all documented
symbols without loading optional groups.

Validation: 12 focused facade tests in `tests/unit/biomechanics/test_biomechanics_facade.py`
and all 384 tests in `tests/unit/biomechanics/` pass. Ruff check, Ruff format, and diff-scoped
mypy pass. See `docs/development/feedback_controls/F08_MUSCLE_RECRUITMENT_TURNOVER.md`.
