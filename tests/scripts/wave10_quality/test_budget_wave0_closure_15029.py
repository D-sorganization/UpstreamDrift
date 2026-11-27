"""Wave-0 (#15028) budget expiry cohort closure checks for #15029."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATHS = (
    _REPO_ROOT / "scripts/config/architecture_budget.json",
    _REPO_ROOT / "scripts/config/file_size_budget.json",
    _REPO_ROOT / "scripts/config/module_size_budget_baseline.json",
    _REPO_ROOT / "scripts/config/doc_size_budget.json",
)
_WAVE0_EXPIRED_ON = "2026-11-30"
_WAVE0_RENEWED_ON = "2027-01-31"
_WAVE0_ISSUE = 15029


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_no_budget_exception_expires_on_2026_11_30() -> None:
    """After #15031 / #15074, the 2026-11-30 cohort must not remain in config."""
    for path in _CONFIG_PATHS:
        data = _load_json(path)
        for entry in data.get("exceptions", []):
            assert entry.get("expires_on") != _WAVE0_EXPIRED_ON, (
                f"{path.relative_to(_REPO_ROOT)} still lists {_WAVE0_EXPIRED_ON}: "
                f"{entry.get('path') or entry.get('symbol')}"
            )


@pytest.mark.unit
def test_file_and_doc_size_wave0_renewals_track_15029() -> None:
    """File, module, and doc exceptions renewed under #15029 cite the tracker."""
    file_cfg = _load_json(_REPO_ROOT / "scripts/config/file_size_budget.json")
    wave0_file = [
        e
        for e in file_cfg["exceptions"]
        if f"#{_WAVE0_ISSUE}" in str(e.get("reason", ""))
    ]
    assert len(wave0_file) == 3
    for entry in wave0_file:
        assert entry["expires_on"] == _WAVE0_RENEWED_ON

    doc_cfg = _load_json(_REPO_ROOT / "scripts/config/doc_size_budget.json")
    dev_log = next(
        e
        for e in doc_cfg["exceptions"]
        if e.get("path") == "docs/development/DEVELOPMENT_LOG.md"
    )
    assert f"#{_WAVE0_ISSUE}" in str(dev_log["reason"])
    assert dev_log["expires_on"] == _WAVE0_RENEWED_ON


@pytest.mark.integration
def test_wave0_budget_gates_pass_as_of_2026_12_01() -> None:
    """Issue #15029 acceptance commands exit 0 past the prior expiry boundary."""
    as_of = "2026-12-01"
    cmds = [
        [
            sys.executable,
            "scripts/ci/check_architecture_budget.py",
            "--all",
            "--as-of",
            as_of,
        ],
        [sys.executable, "scripts/ci/check_file_size_budget.py", "--as-of", as_of],
        [sys.executable, "scripts/check_module_size_budget.py"],
        [sys.executable, "scripts/check_doc_size_budget.py"],
    ]
    for cmd in cmds:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}"
        )


@pytest.mark.unit
def test_wave0_renewed_rows_active_on_eval_day() -> None:
    """Renewed architecture rows remain active on 2026-12-01."""
    config = _load_json(_REPO_ROOT / "scripts/config/architecture_budget.json")
    evaluation_day = date(2026, 12, 1)
    wave0 = [
        e
        for e in config["exceptions"]
        if e.get("issue") == _WAVE0_ISSUE
        and "Part of #15028" in str(e.get("reason", ""))
    ]
    assert wave0
    for entry in wave0:
        assert date.fromisoformat(str(entry["expires_on"])) >= evaluation_day
