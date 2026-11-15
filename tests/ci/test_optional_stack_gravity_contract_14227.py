"""Runbook contract for CI Optional Stack mechanical slices (#14227)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_RUNBOOK = _REPO / "docs" / "ci" / "optional_stack_14227.md"
_MODEL_PROVIDER = _REPO / "src/shared/python/optimization/model_provider.py"
_REGRESSION_TEST = _REPO / "tests/unit/optimization/test_model_provider.py"
_LAUNCHER_SERVICE = _REPO / "src/api/services/launcher_service.py"
_LAUNCHER_REGRESSION = _REPO / "tests/unit/api/test_launcher_service_processes_14227.py"


def test_optional_stack_14227_runbook_exists() -> None:
    assert _RUNBOOK.is_file(), "docs/ci/optional_stack_14227.md must document slice 1"


@pytest.mark.parametrize(
    "needle",
    [
        "GRAVITY_M_S2",
        "build_pinocchio_model",
        "test_pinocchio_model_uses_the_shared_gravity_constant",
        "model_provider.py",
        "Slice 2",
        "Slice 3",
        "#15069",
        "#15089",
        "must not",
        "Scheduled Lane Status",
    ],
)
def test_runbook_names_optional_stack_slices_and_evidence(needle: str) -> None:
    text = _RUNBOOK.read_text(encoding="utf-8")
    assert needle in text


def test_pinocchio_builder_sets_shared_gravity_on_main() -> None:
    text = _MODEL_PROVIDER.read_text(encoding="utf-8")
    assert "GRAVITY_M_S2" in text
    assert "build_pinocchio_model" in text
    assert "model.gravity.linear = np.array([0.0, 0.0, -float(GRAVITY_M_S2)])" in text


def test_regression_test_symbol_is_present() -> None:
    text = _REGRESSION_TEST.read_text(encoding="utf-8")
    assert "def test_pinocchio_model_uses_the_shared_gravity_constant" in text
    assert "GRAVITY_M_S2" in text


def test_launcher_service_filters_unspec_list_processes_rows() -> None:
    text = _LAUNCHER_SERVICE.read_text(encoding="utf-8")
    assert "def get_running_processes" in text
    assert 'isinstance(row.get("slot_name"), str)' in text
    assert "running_processes" in text


def test_launcher_api_regression_module_exists() -> None:
    assert _LAUNCHER_REGRESSION.is_file(), (
        "tests/unit/api/test_launcher_service_processes_14227.py must guard slice 3"
    )


def test_runbook_records_green_scheduled_optional_stack_after_slice3() -> None:
    """#14227 acceptance: post-#15089 scheduled lane success on main."""
    text = _RUNBOOK.read_text(encoding="utf-8")
    assert "43688504401" in text, (
        "runbook must record the first green scheduled CI Optional Stack run after #15089"
    )
    assert "optional-stack-check" in text
    assert "Green" in text or "success" in text.lower()
