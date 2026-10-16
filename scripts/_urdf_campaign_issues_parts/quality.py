"""URDF Hardening Campaign issue definitions: code-quality issues."""

from __future__ import annotations

ISSUES: list[tuple[str, list[str], str]] = []

# ============================================================
# CODE QUALITY
# ============================================================
ISSUES.append(
    (
        "[quality] Remove duplicated `if not (X is not None)` guards in api.py",
        ["code-quality", "DRY", "character-builder", "tech-debt", "priority:medium"],
        """## Problem

`src/shared/python/humanoid_character_builder/interfaces/api.py` contains
~15 instances of duplicated nonsense validation guards, each repeated
verbatim twice in a row:

```python
if not (output_dir is not None):
    raise ValueError("output_dir must be provided")
if not (output_dir is not None):
    raise ValueError("output_dir must be provided")
```

Examples at lines 159-162, 243-246, 280-283, 358-361, 473-476, 529-532,
561-564, 620-623, 676-679, 711-714.

These guards are:
1. **Tautological** — `output_dir` is a typed parameter; if the caller
   passes `None` the type checker would catch it.
2. **Duplicated** — every guard appears twice in succession.
3. **Wrong signal** — `if not (x is not None)` is a confusing way to
   write `if x is None`.

This is unreviewed AI-generated boilerplate.

## Fix

Sweep the file. For each duplicated block:
- If the parameter is non-Optional in the signature, **delete** the guard.
- If the parameter genuinely needs runtime validation (e.g. user-facing
  paths from config), keep ONE guard with a useful message.

Apply the same sweep to any other file with `if not (.* is not None):`
duplicates.

## Acceptance

- [ ] `grep -c "if not (.* is not None)"` in `interfaces/api.py` is 0
      (or matches a justified count documented in commit message)
- [ ] No new test failures
- [ ] Ruff passes
- [ ] Net line reduction ~30 lines

## Search command

```bash
grep -rn "if not (.* is not None)" src/shared/python/humanoid_character_builder/
```
""",
    )
)

ISSUES.append(
    (
        "[quality] Resolve ARCHITECTURE_DEBT in interfaces/api.py — split into focused modules",
        [
            "code-quality",
            "refactor",
            "character-builder",
            "tech-debt",
            "architecture",
            "priority:medium",
        ],
        """## Problem

`src/shared/python/humanoid_character_builder/interfaces/api.py` opens
with:

```python
# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates
# excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal
# classes appropriately.
```

The file is 723 lines and contains:
- `SegmentMeshInfo` (data class)
- `ExportOptions` (data class)
- `CharacterBuildResult` (orchestrator with simulate / preview / export)
- `CharacterBuilder` (main builder with build / inertia / preset methods)
- Module-level `quick_build` / `quick_urdf` helpers

## Proposed split

```
interfaces/
  __init__.py        # public re-exports
  api.py             # CharacterBuilder only (orchestration)
  results.py         # CharacterBuildResult, SegmentMeshInfo
  options.py         # ExportOptions
  preview.py         # MuJoCo simulate() / preview() helpers
  quick.py           # quick_build / quick_urdf
```

## Acceptance

- [ ] No file in `interfaces/` exceeds 400 lines
- [ ] `from humanoid_character_builder import CharacterBuilder` still works
- [ ] All existing tests pass without modification
- [ ] `ARCHITECTURE_DEBT` comment removed
- [ ] CLAUDE.md mentions the split if it affects contributor workflow

## Depends on

- "[bug] CharacterBuildResult missing solver_status" (do that first to
  avoid merge churn)
- "[quality] Remove duplicated guards in api.py" (cleanup before split)
""",
    )
)

ISSUES.append(
    (
        "[quality] Replace silent except (PermissionError, OSError) in CharacterBuilder.build()",
        ["code-quality", "error-handling", "character-builder", "priority:medium"],
        """## Problem

`CharacterBuilder.build()` catches only `(PermissionError, OSError)`:

```python
except (PermissionError, OSError) as e:
    logger.error(f"Character build failed: {e}")
    return CharacterBuildResult(success=False, params=params, error_message=str(e))
```

This:
1. **Misses** the most common real-world failures: `ValueError` from
   parameter validation, `KeyError` from missing segments, `RuntimeError`
   from mesh backends, `ImportError` from optional deps.
2. **Conflates** environment errors with build errors (a missing temp dir
   permission ≠ a malformed body parameter).
3. Returns a **misleading** result — the caller can't tell why it failed.

## Fix

After the `solver_status` contract lands, classify failures:

```python
try:
    ...
except ValueError as e:           # bad params
    return _failure(BuildStatus.FAILURE, "validation", e)
except (PermissionError, OSError) as e:   # filesystem
    return _failure(BuildStatus.FAILURE, "io", e)
except ImportError as e:          # missing optional backend
    return _failure(BuildStatus.PARTIAL, "missing_backend", e)
# everything else → propagate (real bug)
```

## Acceptance

- [ ] Build failures distinguishable by error category
- [ ] Tests for each failure class
- [ ] Unexpected exceptions propagate (no bare except)

## Depends on

- "[arch] Define unified BuildResult contract with solver_status field"
""",
    )
)
