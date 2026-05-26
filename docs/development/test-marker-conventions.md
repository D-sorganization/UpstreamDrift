# Test Marker Conventions

This document defines the conventions for `pytest.mark.skip`, `pytest.mark.skipif`,
`pytest.mark.xfail`, and `requires_*` markers used across the UpstreamDrift test suite.

## Quick Reference

| Marker                                  | When to use                         | Who auto-applies it |
| --------------------------------------- | ----------------------------------- | ------------------- |
| `@pytest.mark.requires_X`               | Test needs optional dep X           | — (you add it)      |
| `@pytest.mark.skipif(cond, ...)`        | Conditionally skip (platform, env)  | — (you add it)      |
| `@pytest.mark.skip(reason=...)`         | Unconditionally skip                | Avoid — see below   |
| `@pytest.mark.xfail(strict=False, ...)` | Known failure tied to an open issue | — (you add it)      |
| `@pytest.mark.xfail(strict=True, ...)`  | Must fail; XPASS is a CI error      | — (you add it)      |

## `requires_*` markers — preferred over `skipif`

All optional-dependency gates must use a `requires_*` marker instead of
an inline `skipif`. The `pytest_collection_modifyitems` hook in
`tests/conftest.py` reads every `requires_*` marker and automatically skips
the test when the dependency is not importable.

### Registered `requires_*` markers

| Marker                    | Dependency checked                  |
| ------------------------- | ----------------------------------- |
| `requires_drake`          | `pydrake`                           |
| `requires_opensim`        | `opensim`                           |
| `requires_mujoco`         | `mujoco`                            |
| `requires_pinocchio`      | `pinocchio`                         |
| `requires_matlab`         | `matlab`                            |
| `requires_matlab_engine`  | `matlab`                            |
| `requires_torch`          | `torch`                             |
| `requires_ort`            | `onnxruntime`                       |
| `requires_gpu`            | always runs (manual skip if needed) |
| `requires_network`        | always runs (manual skip if needed) |
| `requires_mocap_fixtures` | always runs (manual skip if needed) |
| `requires_gl`             | always runs (manual skip if needed) |

### Correct pattern

```python
# DO
@pytest.mark.requires_drake
def test_drake_something() -> None:
    import pydrake
    ...
```

### Incorrect pattern (creates redundant markers)

```python
# DON'T
import importlib.util
_DRAKE_AVAILABLE = importlib.util.find_spec("pydrake") is not None

@pytest.mark.requires_drake
@pytest.mark.skipif(not _DRAKE_AVAILABLE, reason="pydrake not installed")
def test_drake_something() -> None:
    ...
```

### Adding a new `requires_*` marker

1. Add the marker definition to the `markers` list in `pyproject.toml`.
2. Add the dep → importable-module mapping to `_REQUIRES_DEP_MAP` in
   `tests/conftest.py`.
3. Use the marker on the test. No `skipif` needed.

## `pytest.mark.skip` — avoid unless truly unconditional

`pytest.mark.skip(reason=...)` skips the test on every run with no
condition check. It should only be used for tests that are genuinely
broken and not expected to pass in any environment.

Rules:

- The `reason` must reference a GitHub issue number: `reason="Broken by #1234"`.
- Do not use `pytest.mark.skip` as a substitute for `requires_*` or `skipif`.
- Review open `skip` markers at least once per release cycle.

## `pytest.mark.xfail` — for known failures tied to tracked work

`xfail_strict = true` is set globally in `pyproject.toml`, which means any
test marked `@pytest.mark.xfail` **without** an explicit `strict` parameter
will inherit `strict=True`. An unexpected pass (XPASS) under `strict=True`
is a CI failure.

### When to use `xfail`

Use `xfail` only when:

- The failure is known and tracked in a GitHub issue.
- You expect the test to start passing once a specific PR lands.
- The test exercises real production logic (not a placeholder).

### Required parameters

```python
@pytest.mark.xfail(
    reason="Description of what is broken (issue #NNNN)",
    strict=False,   # or strict=True — must be explicit
)
def test_something_known_broken() -> None:
    ...
```

Both `reason` and `strict` are required. The CI ratchet will flag new
`xfail` markers that omit `strict`.

### `strict=True` vs `strict=False`

| `strict` | Test fails       | Test passes                        |
| -------- | ---------------- | ---------------------------------- |
| `True`   | XFAIL (expected) | XPASS → **CI failure**             |
| `False`  | XFAIL (expected) | XPASS (reported but not a failure) |

Use `strict=True` when you are certain the test must fail today (e.g.,
a feature is not yet implemented). Switch it to `strict=False` if the
failure is intermittent.

### Removing stale xfail markers

When a bug is fixed, remove the `xfail` marker and let the test pass
normally. If the test now unexpectedly fails without the `xfail`, that
is a regression bug and should be fixed immediately.

## Class-level vs method-level markers

When multiple tests in the same class share the same skip condition,
apply the marker at the class level:

```python
# DO — single point of truth
@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo not installed")
class TestMuJoCoPhysics:
    def test_one(self) -> None: ...
    def test_two(self) -> None: ...
    def test_three(self) -> None: ...
```

```python
# DON'T — duplicated condition
class TestMuJoCoPhysics:
    @pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_one(self) -> None: ...
    @pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_two(self) -> None: ...
    @pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_three(self) -> None: ...
```

Similarly, use module-level `pytestmark`:

```python
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test"),
]
```

## Platform and environment guards

For Windows-only or POSIX-only tests, use `skipif(sys.platform == ...)`.
Add it to the module-level `pytestmark` list so it applies to all tests
in that file.

## Counting markers

To audit the current skip/xfail counts:

```bash
# Count pytest.mark.skip (hard skip)
grep -r "pytest.mark.skip\b" tests/ --include="*.py" | wc -l

# Count pytest.mark.skipif
grep -r "pytest.mark.skipif" tests/ --include="*.py" | wc -l

# Count pytest.mark.xfail
grep -r "pytest.mark.xfail" tests/ --include="*.py" | wc -l
```

The ≥30% reduction target from issue #6095 is tracked against the baseline
of ~117 `pytest.mark.skip` + `skipif` markers in the `.py` files.
