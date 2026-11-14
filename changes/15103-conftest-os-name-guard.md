---
issue: 15103
summary: "Guard root conftest pytest_runtest_makereport and teardown against leaked os.name patches (#15103)"
---

- Record `_HOST_OS_NAME` in `conftest.py` and wrap `pytest_runtest_makereport` (`tryfirst=True, hookwrapper=True`) to restore `_os.name` to `_HOST_OS_NAME` while formatting reports (and restore the call-phase value in `finally` so `monkeypatch.undo()` remains consistent), plus a `trylast=True` `pytest_runtest_teardown` reset (#15103).
- Add `tests/unit/ci/test_conftest_os_name_makereport_guard.py` (`@pytest.mark.unit`) reproducing the `NotImplementedError`/`UnsupportedOperation` crash in subprocess `pytest` when failing with mutated `os.name` and verifying hook behavior directly.
