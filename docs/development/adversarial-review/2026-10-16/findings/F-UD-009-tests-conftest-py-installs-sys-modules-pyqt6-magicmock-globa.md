# F-UD-009

**`tests/conftest.py` installs `sys.modules["PyQt6"] = MagicMock()` globally, which CLAUDE.md forbids**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | tests    | high       | S      | proposed |

**Locations:** `tests/conftest.py:170`, `tests/conftest.py:280`, `CLAUDE.md:146`, `AGENTS.md:216`

## Finding

- Severity: Medium - Category: tests
- Confidence: high
- Location: tests/conftest.py:170-172; tests/conftest.py:280-284; CLAUDE.md:146; AGENTS.md:216-232
- Evidence:
  ```python
  if not _has_pyqt6:
      ...
      pyqt_mock = MagicMock()
      sys.modules["PyQt6"] = pyqt_mock
      sys.modules["PyQt6.QtCore"] = mock_core
  ```
- Failure scenario: 312 test files import PyQt6; 2 check the `__ud_fake__` flag. On any machine where PyQt6 fails to import
  (the Windows DLL case AGENTS.md describes), `pytest.importorskip("PyQt6")` succeeds against the mock and GUI tests "pass" by
  calling MagicMocks — the exact outcome the "skip cleanly" rule was written to prevent. CI installs PyQt6 from the lock, so this
  bites local runs and lean lanes, i.e. exactly where the owner runs tests.
- Why a newer engineer misses this: the mock was added to let _collection_ survive; nobody re-ran the suite with PyQt6 absent to
  see how many tests went green instead of yellow.
- Proposed change: delete the mock; register a collection-time `pytest.skip` for modules that import PyQt6 when it is missing
  (the `OptionalCollectionRule` machinery at tests/conftest.py:324 already exists for this).
- Verification: `pip uninstall PyQt6; pytest tests/ui -q` shows skips, not passes.
- Effort: S

## Draft Pull Request

**Title:** `test(conftest): drop the global PyQt6 MagicMock; skip GUI tests instead`

### Why This Matters (The Pitch)

tests/conftest.py installs sys.modules['PyQt6'] = MagicMock() when PyQt6 is missing, which CLAUDE.md:146 explicitly forbids. With 312 test files importing PyQt6 and only 2 checking the **ud_fake** flag, pytest.importorskip('PyQt6') succeeds against the mock, so GUI tests go green by calling MagicMocks instead of skipping. That is the exact outcome the 'skip cleanly' rule in AGENTS.md was written to prevent, and it bites on local runs and lean lanes, which is where the owner actually runs tests (CI installs PyQt6 from the lock). A green result that means nothing is worse than a yellow one. The repo already has OptionalCollectionRule machinery at tests/conftest.py:324 to skip modules at collection time, so the change is to delete the mock and reuse it. Size S, and test results regain their meaning.

### Proposed PR Body

## Problem

`tests/conftest.py:170-172` and `:280-284` set `sys.modules["PyQt6"] = pyqt_mock` (and `PyQt6.QtCore`) when PyQt6 is absent, violating `CLAUDE.md:146` and `AGENTS.md:216-232`. 312 test files import PyQt6; 2 check the `__ud_fake__` flag, so `pytest.importorskip("PyQt6")` passes against the mock.

## Change

- [ ] Delete the module-level PyQt6 mock
- [ ] Register a collection-time `pytest.skip` for modules that import PyQt6 when it is missing, using the `OptionalCollectionRule` machinery at `tests/conftest.py:324`

## Verification

`pip uninstall PyQt6; pytest tests/ui -q` shows skips, not passes.

## Risk and Rollback

Low: lean environments will show more skips rather than false passes. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-009

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
