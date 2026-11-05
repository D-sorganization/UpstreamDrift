from __future__ import annotations

from pathlib import Path

import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _job_steps(workflow: str, job: str) -> list[dict[str, object]]:
    data = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    return data["jobs"][job]["steps"]


@pytest.mark.unit
def test_ci_standard_installs_trivy_and_skips_action_setup() -> None:
    steps = _job_steps("ci-standard.yml", "security-scans")

    installer_index = next(
        index
        for index, step in enumerate(steps)
        if "install_trivy.sh" in str(step.get("run", ""))
    )
    trivy_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("aquasecurity/trivy-action@")
    ]
    assert trivy_steps
    assert all(index > installer_index for index, _ in trivy_steps)
    assert all(
        step.get("with", {}).get("skip-setup-trivy") == "true"
        for _, step in trivy_steps
    )


@pytest.mark.unit
def test_docker_security_scan_installs_trivy_and_skips_action_setup() -> None:
    steps = _job_steps("docker-security-scan.yml", "trivy-scan")

    installer_index = next(
        index
        for index, step in enumerate(steps)
        if "install_trivy.sh" in str(step.get("run", ""))
    )
    trivy_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("aquasecurity/trivy-action@")
    ]
    assert trivy_steps
    assert all(index > installer_index for index, _ in trivy_steps)
    assert all(
        step.get("with", {}).get("skip-setup-trivy") == "true"
        for _, step in trivy_steps
    )


@pytest.mark.unit
def test_docker_security_scan_runs_when_installer_changes() -> None:
    data = yaml.safe_load(
        (WORKFLOWS / "docker-security-scan.yml").read_text(encoding="utf-8")
    )
    triggers = data.get("on", data.get(True, {}))

    for trigger in ("push", "pull_request"):
        assert "scripts/ci/install_trivy.sh" in triggers[trigger]["paths"]
