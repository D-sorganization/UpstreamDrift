from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_python_version_coherence import (
    PythonMinor,
    PythonVersionPolicy,
    check_repository,
    validate_policy,
)

pytestmark = pytest.mark.unit


def _policy(**overrides: object) -> PythonVersionPolicy:
    values: dict[str, object] = {
        "requires_floor": PythonMinor(3, 11),
        "classifiers": frozenset({PythonMinor(3, 11), PythonMinor(3, 12)}),
        "mypy_target": PythonMinor(3, 11),
        "installer_floor": PythonMinor(3, 11),
        "lock_version": PythonMinor(3, 12),
        "docker_versions": frozenset({PythonMinor(3, 12)}),
        "ci_standard_versions": frozenset({PythonMinor(3, 11), PythonMinor(3, 12)}),
        "sidekick_wheel_versions": frozenset({PythonMinor(3, 11), PythonMinor(3, 12)}),
    }
    values.update(overrides)
    return PythonVersionPolicy(**values)  # type: ignore[arg-type]


def test_validate_policy_accepts_coherent_supported_versions() -> None:
    assert validate_policy(_policy()) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"requires_floor": PythonMinor(3, 10)}, "pyproject requires-python floor"),
        ({"mypy_target": PythonMinor(3, 10)}, "mypy python_version"),
        (
            {"classifiers": frozenset({PythonMinor(3, 10), PythonMinor(3, 11)})},
            "classifiers advertise unsupported",
        ),
        ({"lock_version": PythonMinor(3, 13)}, "requirements.lock generation version"),
        ({"docker_versions": frozenset({PythonMinor(3, 13)})}, "Docker base version"),
        (
            {
                "ci_standard_versions": frozenset(
                    {PythonMinor(3, 10), PythonMinor(3, 11)}
                )
            },
            "ci-standard.yml tests unsupported",
        ),
        (
            {
                "sidekick_wheel_versions": frozenset(
                    {PythonMinor(3, 10), PythonMinor(3, 11)}
                )
            },
            "package-standalone-sidekick.yml tests unsupported",
        ),
        (
            {"sidekick_wheel_versions": frozenset({PythonMinor(3, 11)})},
            "does not smoke-test Python 3.12",
        ),
    ],
)
def test_validate_policy_rejects_incoherent_versions(
    override: dict[str, object],
    expected: str,
) -> None:
    findings = validate_policy(_policy(**override))

    assert any(expected in finding for finding in findings)


def test_repository_python_version_declarations_are_coherent() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert check_repository(repo_root) == []
