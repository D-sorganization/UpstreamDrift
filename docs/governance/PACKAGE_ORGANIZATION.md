# Package Organization

> **Status:** Authoritative. Updated as of issue #5908.
> Questions? Open an issue tagged `architecture`.

## The cohesion problem

`src/shared/python/` currently holds **93 entries** (packages, modules, and
support files). Many were added incrementally without a placement decision
framework, which produced:

- **Naming collision pairs** where two packages cover overlapping ground (see
  below).
- **UpstreamDrift-specific code** sitting alongside genuinely cross-repo
  utilities, making it unclear what external consumers can rely on.
- **Growth pressure**: every new feature defaults to `src/shared/python/`
  because "shared" sounds right, even when the code is only used by one engine
  or one tool.

**Until the cohesion question is resolved (tracked by #5908), do not add new
top-level packages to `src/shared/python/`.** If you need a new package,
follow the placement rules below and get an explicit decision in the PR.

---

## Placement decision: `src/shared/python/X` vs `src/X`

| Question                                                  | If YES                                                            | If NO                   |
| --------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------- |
| Used by more than one top-level `src/` subdirectory?      | `src/shared/python/X`                                             | `src/<domain>/`         |
| Consumed by an external repo (Tools, Gasification_Model)? | `vendor/ud-tools/` or `src/shared/python/X`                       | `src/<domain>/`         |
| Physics-engine abstraction or cross-engine primitive?     | `src/shared/python/engine_core/` or `src/shared/python/<domain>/` | `src/engines/<engine>/` |

When in doubt, place code in the most specific `src/<domain>/` directory and
promote it to `src/shared/python/` only when a _second_ real consumer appears.

---

## Known naming collision pairs (resolve before adding related code)

These pairs need consolidation. Do **not** add code to either side without
first checking which package should own it long-term.

| Pair                                                     | Notes                                                                                                                                                                    |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `biomech` / `biomechanics`                               | `biomechanics` holds muscle/dynamics code; `biomech` holds exercise schemas. Needs merge decision.                                                                       |
| `plot_engine` / `plot_style` / `plot_theme` / `plotting` | Four packages covering rendering backends, color channels, theme management, and animation. `plot_theme` likely belongs in `sidekick.theme`.                             |
| `data_processing` / `data_processor`                     | `data_processing` is the UI-agnostic extraction (issue #407); `data_processor` is the Rust bulk-I/O engine (issue #2989). Different concerns, confusingly similar names. |
| `upstream_drift` / `upstream_drift_tools`                | `upstream_drift_tools` is a **deprecated alias** (see below). `upstream_drift` is a separate package — do not conflate them.                                             |

---

## Cross-repo shared packages

The following packages are consumed by **Tools** (`vendor/ud-tools/`) and/or
**Gasification_Model**. Breaking changes here require a coordinated PR.

- `sidekick` — canonical shared utility library (calculators, data processing,
  theme, UI widgets, utils). **This is the primary cross-repo surface.**
- `sidekick.theme` — fleet-wide color theme system (13+ themes, PyQt6 integration).
- `sidekick.ui` — PyQt6 widgets shared across the fleet.

All other `src/shared/python/` packages are considered UpstreamDrift-internal
until explicitly documented otherwise.

---

## `upstream_drift_tools` — deprecated, do not use

`src/shared/python/upstream_drift_tools/` is a **compatibility shim** created
during the `upstream_drift_tools` → `sidekick` rename (issues #5619, #5623).

- It re-exports every public symbol from `sidekick`.
- It emits a `DeprecationWarning` on first import.
- It will be **removed** in a future major release.
- The hygiene test `tests/unit/repo_hygiene/test_no_deprecated_imports.py`
  enforces that no `src/` or `tests/` file imports `upstream_drift_tools`.

**Migration:**

```python
# Before (deprecated)
from upstream_drift_tools.theme import CatppuccinTheme

# After (correct)
from sidekick.theme import CatppuccinTheme
```

---

## Adding new code — checklist

1. **Search first.** Run `grep -r "concept_name" src/shared/python/` before
   writing anything. See `AGENTS.md §A` for the full discovery workflow.
2. **Apply the placement table above.** Default to the specific `src/<domain>/`
   directory, not `src/shared/python/`.
3. **No new top-level packages in `src/shared/python/`** without a decision
   recorded in a PR comment or ADR (blocked by #5908).
4. **Import from `sidekick`**, not `upstream_drift_tools`, for all shared
   utility code.
5. If your code genuinely belongs in `src/shared/python/`, add it as a
   **submodule of an existing package** rather than a new top-level package
   when possible.
