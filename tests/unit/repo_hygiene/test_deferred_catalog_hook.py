"""Exercise the deployed catalog gate through its configured hook command."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CATALOG = Path("docs/development/planning/catalog.json")


def _hook() -> dict[str, object]:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    matches = [
        hook
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "deferred-validation"
    ]
    assert len(matches) == 1, "The published catalog needs exactly one enforcing hook"
    hook = matches[0]
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False
    assert hook["language"] == "python"
    return dict(hook)


def _run_hook(root: Path) -> subprocess.CompletedProcess[str]:
    command = shlex.split(str(_hook()["entry"]))
    assert command[0] == "python"
    return subprocess.run(
        [sys.executable, *command[1:]],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def _copy_gate(tmp_path: Path) -> None:
    shutil.copytree(ROOT / CATALOG.parent, tmp_path / CATALOG.parent)
    destination = tmp_path / "shared_scripts"
    destination.mkdir()
    for name in (
        "deferred_validation.py",
        "deferred_planning.py",
        "handoff_validator.py",
    ):
        shutil.copyfile(ROOT / "shared_scripts" / name, destination / name)


def test_published_catalog_passes_configured_hook() -> None:
    result = _run_hook(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "catalog OK" in result.stdout


@pytest.mark.parametrize(
    "fault", ["premature_activation", "missing_checker", "dual_catalog"]
)
def test_configured_hook_rejects_invalid_publication(
    tmp_path: Path, fault: str
) -> None:
    _hook()
    _copy_gate(tmp_path)
    if fault == "premature_activation":
        path = tmp_path / CATALOG
        data = json.loads(path.read_text(encoding="utf-8"))
        data["entries"][0]["status"] = "activated"
        data["entries"][0]["activation_issue"] = (
            "https://github.com/D-sorganization/UpstreamDrift/issues/10783"
        )
        path.write_text(json.dumps(data), encoding="utf-8")
    elif fault == "missing_checker":
        (tmp_path / "shared_scripts/deferred_planning.py").unlink()
    else:
        second = tmp_path / "docs/planning/deferred-validation.json"
        second.parent.mkdir(parents=True)
        second.write_text("{}", encoding="utf-8")
    result = _run_hook(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "failed validation" in result.stdout


def test_installed_bundle_matches_published_receipt() -> None:
    """A formatter or local edit must not silently fork the shared validator."""
    receipt = json.loads(
        (ROOT / "docs/development/deferred-catalog-bundle.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(receipt["files"]) == 3
    for relative, expected in receipt["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
