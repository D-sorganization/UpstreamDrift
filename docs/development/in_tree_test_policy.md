# In-tree test policy (`src/**/tests`, root-level `tests/test_*.py`)

This note documents how the repository treats test files that live **outside**
the canonical `tests/` topic-subdirectory layout, closing the gap raised in
issue #7126 (default `pytest` silently skips in-tree tests).

## The default signal

`pyproject.toml` pins:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = ["src", "archive", "legacy", "node_modules", "vendor", ".git", "*.egg-info"]
```

So a plain `pytest` (and every CI lane that relies on the default collection)
only exercises the `tests/` tree. Test files colocated under `src/**/tests/`
are **not** collected by default. This is deliberate, not accidental — see
below — but it must be explicit and enforced so changes cannot silently regress
those modules under a green default run.

## Why in-tree `src/**/tests` are excluded by default

1. **Import-path collisions.** Collected from the repo root, a file like
   `src/shared/python/calc_backend/tests/test_standard_response.py` is imported
   as `src.shared.python.calc_backend.tests.test_standard_response`, but it was
   written to be run from its own subtree and uses sibling-relative imports
   (`from .standard_response import ...`) that do not resolve under the root
   rootdir. Forcing collection turns these into hard `collection errors`.
2. **Conftest / fixture shadowing.** Engine subtrees ship their own
   `conftest.py` files; recursing into `src/` re-introduces
   `ImportPathMismatchError` and duplicate-fixture clashes that
   `norecursedirs = ["src", ...]` exists to prevent.
3. **Legacy migration in progress.** These directories are being migrated into
   the canonical `tests/` layout. They are grandfathered, not endorsed.

## The intentional exclusion list (with rationale)

`scripts/check_test_layout.py` is the single source of truth. It carries two
grandfathered allowlists:

- `LEGACY_SRC_TEST_DIRS` — in-tree `src/**/tests` directories that predate the
  canonical layout. Rationale: legacy subtree tests pending migration (per the
  import-path / conftest issues above).
- `LEGACY_ROOT_TEST_FILES` — flat `tests/test_*.py` files at the root of
  `tests/` that should eventually move into topic subdirectories.

**Anything not on those lists is rejected** by the `Test Layout Guard`
(`python3 scripts/check_test_layout.py`, run in CI's `quality-gate`). That is
the policy gate that prevents _new_ uncollected in-tree tests from being added
by accident — the list can only shrink, never grow, without an explicit edit
that reviewers will see.

## Named command to inspect in-tree collection

In-tree directories are not part of the default run, but they can be inspected
explicitly:

```bash
make test-in-tree          # collect-only diagnostic over LEGACY_SRC_TEST_DIRS
```

This is a **diagnostic** lane: it reports which legacy in-tree tests still
collect and which have outstanding import/migration debt. It is intentionally
not part of the blocking default lane, because several legacy subtrees have
known collection errors (the very debt this policy tracks). New tests must be
added under `tests/` where the default lane collects them.

## Removing a directory from the allowlist

When an in-tree test directory is migrated into `tests/` (or deleted), remove
its entry from `LEGACY_SRC_TEST_DIRS` in `scripts/check_test_layout.py`. The
guard and `tests/scripts/test_check_test_layout.py` keep the list honest.
