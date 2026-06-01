---
title: "simulation_backends: `UnknownBackendError.__str__` produces repr-quoted messages because of KeyError subclassing"
labels: [bug, ux, simulation-backends, fleet-followup]
priority: low
discovered_in: PR claude/test-coverage-improvements (branch)
discovered_at: 2026-06-01
reporter: claude
status: open
related_to: tests/unit/simulation_backends/test_capabilities_contract_extra.py
---

## Summary

`src/shared/python/simulation_backends/exceptions.py::UnknownBackendError` is
defined as:

```python
class UnknownBackendError(BackendError, KeyError):
    """Raised when a backend name is not registered in the factory.

    Subclasses :class:`KeyError` so existing ``except KeyError`` sites keep
    working, while new code can catch the precise type.
    """
```

Because Python's `KeyError.__str__` wraps the first argument in
`repr()` (so `KeyError("foo")` becomes `"'foo'"` when stringified), the
inheritance bleeds into our exception's user-facing string:

```python
>>> str(UnknownBackendError("backend 'foo' not found"))
"'backend 'foo' not found'"
```

The new test
`tests/unit/simulation_backends/test_capabilities_contract_extra.py::TestExceptionHierarchy::test_exception_messages_preserved`
pins this behaviour and documents the workaround (`exc.args[0]`) for
callers who need a clean string.

## Reproduction

```python
from src.shared.python.simulation_backends import UnknownBackendError
err = UnknownBackendError("no backend named 'foo'")
print(repr(str(err)))   # -> "'no backend named \\'foo\\''"
print(err.args[0])      # -> "no backend named 'foo'"
```

## Impact

- `logging.exception()` shows the repr-quoted string in user logs.
- CLI / REST API error responses that include `str(exc)` show the
  repr-quoted string.
- Downstream code that does `f"backend error: {exc}"` produces ugly
  output.

## Recommended fix

Override `__str__` in `UnknownBackendError` to return `self.args[0]` if
present, falling back to the default `Exception.__str__`. This is a
small one-line patch; the new contract test should be updated to
expect a clean string.

```python
class UnknownBackendError(BackendError, KeyError):
    """..."""
    def __str__(self) -> str:  # type: ignore[override]
        if self.args:
            return str(self.args[0])
        return super().__str__()
```

## Acceptance criteria

- `str(UnknownBackendError("foo")) == "foo"`.
- The contract test
  `tests/unit/simulation_backends/test_capabilities_contract_extra.py::TestExceptionHierarchy::test_exception_messages_preserved`
  passes without the `exc.args[0]` workaround.
- `except KeyError` and `except BackendError` continue to catch
  `UnknownBackendError` (both inheritance relationships must remain
  intact).

## Related

- `src/shared/python/simulation_backends/exceptions.py` — production code.
- `tests/unit/simulation_backends/test_capabilities_contract_extra.py` — new tests.
