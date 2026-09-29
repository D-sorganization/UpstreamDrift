from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

_TESTS_CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
_SPEC = importlib.util.spec_from_file_location(
    "tests_collection_policy_conftest", _TESTS_CONFTEST_PATH
)
assert _SPEC is not None
assert _SPEC.loader is not None
collection_policy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = collection_policy
_SPEC.loader.exec_module(collection_policy)


def test_optional_process_calculator_tests_are_ignored_when_anchor_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(collection_policy, "_module_available", lambda module: False)

    should_ignore = collection_policy._should_ignore_optional_collection_path(
        Path("tests/unit/process_calculators/test_flare_calculator.py")
    )

    assert should_ignore is True


def test_optional_process_calculator_tests_collect_when_anchor_exists(
    monkeypatch,
) -> None:
    monkeypatch.setattr(collection_policy, "_module_available", lambda module: True)
    monkeypatch.setattr(
        collection_policy, "_symbol_available", lambda module, symbol: True
    )

    should_ignore = collection_policy._should_ignore_optional_collection_path(
        Path("tests/unit/process_calculators/test_flare_calculator.py")
    )

    assert should_ignore is False


def test_stale_root_script_tests_are_ignored_when_module_missing(monkeypatch) -> None:
    monkeypatch.setattr(collection_policy, "_module_available", lambda module: False)

    should_ignore = collection_policy._should_ignore_optional_collection_path(
        Path("tests/unit/test_setup_golf_suite.py")
    )

    assert should_ignore is True


def test_unrelated_unit_test_is_not_ignored(monkeypatch) -> None:
    monkeypatch.setattr(collection_policy, "_module_available", lambda module: False)
    monkeypatch.setattr(
        collection_policy, "_symbol_available", lambda module, symbol: False
    )

    should_ignore = collection_policy._should_ignore_optional_collection_path(
        Path("tests/unit/core/test_version.py")
    )

    assert should_ignore is False


def test_pytest_ignore_collect_warns_once_for_optional_stack_skip(
    monkeypatch,
) -> None:
    monkeypatch.setattr(collection_policy, "_module_available", lambda module: False)
    monkeypatch.setattr(
        collection_policy, "_symbol_available", lambda module, symbol: False
    )
    monkeypatch.setattr(collection_policy, "_OPTIONAL_COLLECTION_WARNED_PATHS", set())
    target = Path("tests/unit/test_setup_golf_suite.py")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = collection_policy.pytest_ignore_collect(target, object())
        second = collection_policy.pytest_ignore_collect(target, object())

    assert first is True
    assert second is True
    assert len(caught) == 1
    assert "Skipping optional test collection" in str(caught[0].message)
    assert "tests/unit/test_setup_golf_suite.py" in str(caught[0].message)
    assert "setup_golf_suite" in str(caught[0].message)
