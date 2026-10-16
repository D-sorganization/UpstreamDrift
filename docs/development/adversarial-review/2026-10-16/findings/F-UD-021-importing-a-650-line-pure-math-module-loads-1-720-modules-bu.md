# F-UD-021

**Importing a 650-line pure-math module loads 1,720 modules (bunkershot3d, pendulum engine, simpleeval, structlog)**

| Severity  | Category     | Confidence | Effort | Status   |
| --------- | ------------ | ---------- | ------ | -------- |
| 🟨 Medium | architecture | high       | M      | proposed |

**Locations:** `src/shared/python/simulation_backends/__init__.py:23`, `src/shared/python/simulation_backends/wrench_extractor.py:18`, `src/shared/python/simulation_backends/ztcf_zvcf.py:51`, `src/bunkershot3d/calibration/f1_continuum.py:55`

## Finding

- Severity: Medium - Category: architecture
- Confidence: high
- Location: src/shared/python/simulation_backends/**init**.py:23-83; src/shared/python/simulation_backends/wrench_extractor.py:18;
  src/shared/python/simulation_backends/ztcf_zvcf.py:51-56; src/bunkershot3d/calibration/f1_continuum.py:55
- Evidence: `from src.shared.python.simulation_backends.ztcf_zvcf import zvcf_acceleration` failed four times in a clean
  environment (`shared.python` seam → `bunkershot3d` → `structlog` → `simpleeval`) before succeeding; `len(sys.modules)` after
  import: 1,720 (1.9 s). `simulation_backends/__init__.py` eagerly re-exports `wrench_extractor`, which imports
  `bunkershot3d.postproc.wrench_trace`.
- Failure scenario: the ZTCF/ZVCF primitives — the thing the owner most wants other people to run — cannot be imported
  without the whole platform's dependency closure; a `structlog` upgrade breaks counterfactual analysis; every CLI start pays
  2 s; the "engine-agnostic" claim is true of the math and false of the import.
- Why a newer engineer misses this: `__init__.py` re-exports are convenient for `from pkg import X`; the transitive cost is
  invisible until the first missing optional dependency.
- Proposed change: make `simulation_backends/__init__.py` lazy (`__getattr__`), drop the `wrench_extractor` re-export, and
  add an import-isolation test (`python -I -c "import ...ztcf_zvcf"` in a venv with numpy only).
- Verification: `python -X importtime -c "from src.shared.python.simulation_backends import ztcf_zvcf" 2>&1 | wc -l` < 300.
- Effort: M

## Draft Pull Request

**Title:** `refactor(simulation_backends): lazy exports so ztcf_zvcf imports alone`

### Why This Matters (The Pitch)

Importing a 650-line pure-math module loads 1,720 modules in 1.9 s. In a clean environment, from simulation_backends.ztcf_zvcf import zvcf_acceleration failed four times, walking the dependency chain through bunkershot3d, structlog and simpleeval, before it succeeded. The cause is simulation_backends/**init**.py eagerly re-exporting wrench_extractor, which imports bunkershot3d.postproc.wrench_trace. So the ZTCF/ZVCF primitives, the thing the owner most wants other people to run, cannot be imported without the whole platform's dependency closure; a structlog upgrade breaks counterfactual analysis and every CLI start pays about 2 s. The engine-agnostic claim is true of the math and false of the import. Making **init**.py lazy via module **getattr**, dropping the wrench_extractor re-export and adding an import-isolation test fixes it for collaborators and CI alike, and the import-isolation test keeps it fixed. Size M.

### Proposed PR Body

## Problem

`src/shared/python/simulation_backends/__init__.py:23-83` eagerly re-exports `wrench_extractor` (`:18`), which imports `bunkershot3d.postproc.wrench_trace`. Importing `ztcf_zvcf` leaves `len(sys.modules)` at 1,720 (1.9 s) and failed four times in a clean environment on missing `structlog`/`simpleeval` before succeeding.

## Change

- [ ] Make `simulation_backends/__init__.py` lazy (`__getattr__`)
- [ ] Drop the `wrench_extractor` re-export
- [ ] Add an import-isolation test (`python -I -c "import ...ztcf_zvcf"` in a venv with numpy only)

## Verification

`python -X importtime -c "from src.shared.python.simulation_backends import ztcf_zvcf" 2>&1 | wc -l` < 300.

## Risk and Rollback

Low to medium: callers using `from simulation_backends import <name>` keep working through `__getattr__`; callers of the dropped re-export need updating. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-021

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
