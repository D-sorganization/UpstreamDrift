"""Corrective RED authority-provenance and CI contracts for #9236."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any

import pytest

from scripts import check_root_clutter
from scripts.research.proximal_distal_energy import (
    register_articulated_manufactured_solution_claims as claim_registration,
)
from scripts.research.proximal_distal_energy import (
    run_articulated_manufactured_solution as runner,
)

yaml = pytest.importorskip("yaml")
pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github/workflows/ci-optional-stack.yml"
AUTHORITY_JOB = "articulated-manufactured-authority"
STUDY_SCRIPT_ROOT = ROOT / "scripts/research/proximal_distal_energy"
HYBRID_CONTRACT = "tests/research/test_articulated_manufactured_hybrid_authority_red.py"
AUTHORITY_IMPORT_REQUIREMENTS = {
    "defusedxml": "0.7.1",
    "pydantic": "2.12.5",
    "pyyaml": "6.0.3",
}


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _authority_job() -> dict[str, Any]:
    job = _workflow()["jobs"].get(AUTHORITY_JOB)
    assert isinstance(job, dict)
    return job


def _commands() -> str:
    return "\n".join(
        step.get("run", "")
        for step in _authority_job()["steps"]
        if isinstance(step.get("run", ""), str)
    )


def _lock_compile_python() -> str:
    text = runner.AUTHORITY_LOCK.read_text(encoding="utf-8")
    matched = re.search(r"--python-version\s+(\d+\.\d+\.\d+)", text)
    assert matched is not None, "lock provenance must name an exact Python patch"
    return matched.group(1)


def _authority_input() -> Path:
    return runner.AUTHORITY_LOCK.with_suffix(".in")


def _requirement_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", " ")):
            continue
        requirement = line.removesuffix(" \\")
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", maxsplit=1)
        versions[name.lower()] = version
    return versions


def _set_authority_platform(
    monkeypatch: pytest.MonkeyPatch, python_version: tuple[int, int, int]
) -> None:
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runner.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runner.sys, "version_info", python_version)
    monkeypatch.setattr(
        runner,
        "_distribution_version",
        lambda name: runner._AUTHORITY_DISTRIBUTIONS[name],
    )


def test_authority_workflow_pins_lock_python_patch_exactly() -> None:
    """The interpreter selector must match the lock's exact target patch."""

    setup = next(
        step
        for step in _authority_job()["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )

    assert str(setup["with"]["python-version"]) == _lock_compile_python()


def test_authority_runtime_rejects_adjacent_python_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching minor with the wrong patch is not the authority runtime."""

    expected = tuple(int(part) for part in _lock_compile_python().split("."))
    wrong_patch = (expected[0], expected[1], expected[2] - 1)
    _set_authority_platform(monkeypatch, wrong_patch)

    with pytest.raises(RuntimeError, match="Python|patch|authority"):
        runner.validate_authority_environment()


def test_authority_runtime_accepts_exact_declared_python_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-patch rejection test has a legitimate positive control."""

    parts = tuple(int(part) for part in _lock_compile_python().split("."))
    assert len(parts) == 3
    expected = (parts[0], parts[1], parts[2])
    _set_authority_platform(monkeypatch, expected)

    runner.validate_authority_environment()


def test_authority_workflow_uses_fresh_runner_temp_virtual_environment() -> None:
    """Self-hosted authority execution cannot inherit global site packages."""

    commands = _commands()
    creation = re.search(
        r"python3?\s+-m\s+venv\s+[\"']?(\$RUNNER_TEMP/[^\"'\s]+)", commands
    )
    assert creation is not None, "authority job must create a RUNNER_TEMP venv"
    venv_python = f"{creation.group(1)}/bin/python"
    assert commands.count(venv_python) >= 4
    assert "--system-site-packages" not in commands


def test_authority_dependency_files_are_scoped_to_the_study() -> None:
    """Specialized research locks do not expand root-level project clutter."""

    authority_input = _authority_input()

    assert runner.AUTHORITY_LOCK.parent != ROOT
    assert runner.AUTHORITY_LOCK.is_relative_to(STUDY_SCRIPT_ROOT)
    assert authority_input.parent == runner.AUTHORITY_LOCK.parent


def test_authority_input_closes_governed_test_import_dependencies() -> None:
    """The isolated no-deps lane must pin every observed import dependency."""

    direct = _requirement_versions(_authority_input())

    assert AUTHORITY_IMPORT_REQUIREMENTS.items() <= direct.items()


def test_repository_root_clutter_gate_accepts_authority_layout() -> None:
    """The committed authority layout must satisfy the existing root contract."""

    assert check_root_clutter.main() == 0


def test_source_register_covers_every_direct_local_import() -> None:
    """Direct local computation modules cannot drift outside source_sha256."""

    source_path = Path(runner.__file__).resolve()
    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    expected: set[str] = set()
    for node in ast.walk(syntax):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("scripts.research.proximal_distal_energy"):
            continue
        imported = ROOT / f"{node.module.replace('.', '/')}.py"
        if imported.is_file():
            expected.add(imported.relative_to(ROOT).as_posix())

    assert expected
    assert expected <= set(runner.SOURCE_PATHS)


def test_governed_claim_evidence_includes_authority_contract_surface() -> None:
    """Downstream evidence must carry every authority-defining input."""

    lock = runner.AUTHORITY_LOCK.relative_to(ROOT).as_posix()
    authority_input = _authority_input().relative_to(ROOT).as_posix()
    workflow = WORKFLOW_PATH.relative_to(ROOT).as_posix()
    expected = set(runner.SOURCE_PATHS) | {
        lock,
        authority_input,
        workflow,
        HYBRID_CONTRACT,
    }

    assert expected <= set(claim_registration.ARTIFACTS)


def test_lock_records_generator_index_and_exact_input_digest() -> None:
    """Cross-target generation must be attributable and input-fresh."""

    authority_input = _authority_input()
    text = runner.AUTHORITY_LOCK.read_text(encoding="utf-8")
    expected_input_sha = hashlib.sha256(authority_input.read_bytes()).hexdigest()

    assert re.search(r"(?m)^# generator: uv==\d+\.\d+\.\d+$", text)
    assert re.search(r"(?m)^# index-url: https://pypi\.org/simple$", text)
    assert f"# input-sha256: {expected_input_sha}" in text
    assert "--index-url https://pypi.org/simple" in text
    assert "--only-binary :all:" in text
    assert "--no-sources" in text


def test_every_direct_input_pin_matches_the_resolved_lock() -> None:
    """The lock-freshness RED tests have a legitimate direct-pin control."""

    direct = _requirement_versions(_authority_input())
    locked = _requirement_versions(runner.AUTHORITY_LOCK)

    assert direct
    assert direct.items() <= locked.items()
