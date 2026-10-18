from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci import docker_feature_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


def test_extract_status_from_registry_json() -> None:
    stdout = '{\n  "name": "api",\n  "status": "AVAILABLE"\n}\n'

    assert probe.extract_status(stdout) == "AVAILABLE"


def test_extract_status_tolerates_leading_stdout_noise() -> None:
    stdout = 'warning: optional import emitted noise\n{"status": "UNAVAILABLE"}\n'

    assert probe.extract_status(stdout) == "UNAVAILABLE"


def test_extract_status_rejects_output_without_status() -> None:
    with pytest.raises(probe.ProbeParseError):
        probe.extract_status("traceback without json\n")


def test_main_reports_probe_output_when_status_is_missing(monkeypatch, capsys) -> None:
    def fake_run_probe(_image: str, _feature: str) -> probe.ProbeResult:
        return probe.ProbeResult(
            returncode=1,
            stdout="traceback on stdout\n",
            stderr="module import failed\n",
        )

    monkeypatch.setattr(probe, "run_probe", fake_run_probe)

    assert probe.main(["upstream-drift-smoke:slim", "api"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "docker exit=1" in captured.err
    assert "traceback on stdout" in captured.err
    assert "module import failed" in captured.err


def test_run_probe_invokes_registry_check(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_subprocess_run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(argv, 1, '{"status":"UNAVAILABLE"}', "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = probe.run_probe("upstream-drift-smoke:slim", "pendulum")

    assert calls == [
        [
            "docker",
            "run",
            "--rm",
            "upstream-drift-smoke:slim",
            "python",
            "-m",
            "src.shared.python.feature_registry",
            "--check",
            "pendulum",
            "--json",
        ]
    ]
    assert result.returncode == 1
    assert probe.extract_status(result.stdout) == "UNAVAILABLE"


def test_docker_smoke_workflow_uses_diagnostic_probe_helper() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "docker-smoke.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("scripts/ci/docker_feature_probe.py") == 2
    assert "src.shared.python.feature_registry --check" not in workflow
    assert "2>/dev/null" not in workflow


_SHARED_BUILDER = (
    "${{ steps.buildkit.outputs.builder || steps.buildx-fallback.outputs.name }}"
)


def test_docker_workflows_build_on_the_host_shared_buildkit_builder() -> None:
    """One layer cache per host (#13208): every image build attaches the shared
    ``ud-buildkit`` builder and no job imports or exports a per-runner cache."""
    workflow_paths = [
        REPO_ROOT / ".github" / "workflows" / "docker-smoke.yml",
        REPO_ROOT / ".github" / "workflows" / "docker-size-gates.yml",
    ]

    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        builds = workflow.count("uses: docker/build-push-action@")

        assert builds > 0
        assert "type=gha" not in workflow
        assert "buildx-cache/upstream-drift" not in workflow
        assert "cache-from:" not in workflow
        assert "cache-to:" not in workflow
        assert "BuildKit cache" not in workflow
        assert "max-parallel: 1" in workflow
        # Each build job ensures the shared builder and builds on it ...
        assert workflow.count("python3 scripts/ci/shared_buildkit.py ensure") == builds
        assert workflow.count(f"builder: {_SHARED_BUILDER}") == builds
        # ... where an ensure crash degrades to a cold build, never a red job ...
        ensure_step = (
            "continue-on-error: true\n"
            "        shell: bash\n"
            "        run: python3 scripts/ci/shared_buildkit.py ensure"
        )
        assert workflow.count(ensure_step) == builds
        # ... and only falls back to a throwaway builder when ensure reports none.
        assert workflow.count("uses: docker/setup-buildx-action@") == builds
        assert workflow.count("if: steps.buildkit.outputs.builder == ''") == builds
