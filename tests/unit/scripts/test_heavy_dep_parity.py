"""Heavy-runner contracts require the robotics Pinocchio distributions."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_heavy_dep_parity as parity
from scripts.check_heavy_dep_parity import (
    CANONICAL_HEAVY_DEPS,
    DOCKERFILE,
    FORBIDDEN_HEAVY_DISTRIBUTIONS,
    WORKFLOW,
    _extract_pip_packages,
    main,
)

pytestmark = pytest.mark.unit


def test_heavy_runtime_uses_robotics_pin_and_rejects_name_collisions() -> None:
    docker_packages = _extract_pip_packages(DOCKERFILE.read_text(encoding="utf-8"))
    workflow_packages = _extract_pip_packages(WORKFLOW.read_text(encoding="utf-8"))

    assert {"pin", "pin-pink"} <= CANONICAL_HEAVY_DEPS
    assert {"pin", "pin-pink"} <= docker_packages
    assert {"pin", "pin-pink"} <= workflow_packages
    assert FORBIDDEN_HEAVY_DISTRIBUTIONS.isdisjoint(docker_packages)
    assert FORBIDDEN_HEAVY_DISTRIBUTIONS.isdisjoint(workflow_packages)
    assert main() == 0


def test_parser_handles_quoted_constraints_and_exposes_wrong_distributions() -> None:
    packages = _extract_pip_packages(
        'RUN pip install "pin>=2.6,<5" "pin-pink>=1,<5" pinocchio pink'
    )

    assert {"pin", "pin-pink", "pinocchio", "pink"} == packages
    assert FORBIDDEN_HEAVY_DISTRIBUTIONS & packages == {"pinocchio", "pink"}


def test_checker_rejects_wrong_distribution_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dockerfile = tmp_path / "Dockerfile.heavy_test"
    workflow = tmp_path / "heavy-tests-opt-in.yml"
    canonical = " ".join(sorted(CANONICAL_HEAVY_DEPS - {"pin", "pin-pink"}))
    wrong = f"RUN pip install {canonical} pinocchio pink\n"
    dockerfile.write_text(wrong, encoding="utf-8")
    workflow.write_text(wrong, encoding="utf-8")
    monkeypatch.setattr(parity, "DOCKERFILE", dockerfile)
    monkeypatch.setattr(parity, "WORKFLOW", workflow)

    assert parity.main() == 1
    error = capsys.readouterr().err
    assert "missing: pin, pin-pink" in error
    assert "forbidden distributions: pink, pinocchio" in error
