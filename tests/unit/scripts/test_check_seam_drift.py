"""Tests for scripts/shared_tools/check_seam_drift.py (UD #9406)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from scripts.shared_tools import check_seam_drift as gate

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _rulings(tmp_path: Path, rulings: dict[str, dict[str, object]]) -> None:
    _write(
        tmp_path / gate.DEFAULT_RULINGS,
        json.dumps({"schema_version": 1, "rulings": rulings}),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    ud = tmp_path / gate.DEFAULT_UD_ROOT
    tools = tmp_path / gate.DEFAULT_TOOLS_ROOT
    _write(tools / "theme/__init__.py", "x = 1\n")
    _write(tools / "theme/colors.py", "y = 1\n")
    _write(tools / "safe_eval.py", "s = 1\n")
    _write(tools / "model_generation/core.py", "m = 1\n")
    _write(ud / "theme/__init__.py", "x = 1\n")
    _write(ud / "theme/colors.py", "y = 2\n")
    _write(ud / "theme/layout_metrics.py", "ud only\n")
    _write(ud / "safe_eval.py", "s = 1\n")
    _write(ud / "model_generation/core.py", "m = 2\n")
    return tmp_path


def _base_rulings() -> dict[str, dict[str, object]]:
    return {
        "theme": {"ruling": "split", "status": "pending-cleanup", "rationale": "r"},
        "safe_eval.py": {
            "ruling": "tools-canonical",
            "status": "pending-cleanup",
            "rationale": "r",
        },
        "model_generation": {
            "ruling": "ud-canonical",
            "status": "pending-cleanup",
            "rationale": "r",
            "tools_ledger_row": "Tools#4915:model_generation",
        },
    }


def test_pending_rulings_pass_with_notes(repo: Path) -> None:
    _rulings(repo, _base_rulings())
    violations, notes = gate.check(repo)
    assert violations == []
    assert any("safe_eval.py" in note for note in notes)


def test_strict_mode_fails_pending(repo: Path) -> None:
    _rulings(repo, _base_rulings())
    violations, _ = gate.check(repo, strict=True)
    assert {v.package for v in violations} == {"theme", "safe_eval.py"}


def test_tools_canonical_cleaned_but_file_remains_fails(repo: Path) -> None:
    rulings = _base_rulings()
    rulings["safe_eval.py"]["status"] = "cleaned"
    _rulings(repo, rulings)
    violations, _ = gate.check(repo)
    assert [v.package for v in violations] == ["safe_eval.py"]


def test_tools_canonical_cleaned_shim_is_allowed(repo: Path) -> None:
    rulings = _base_rulings()
    rulings["safe_eval.py"]["status"] = "cleaned"
    _rulings(repo, rulings)
    _write(
        repo / gate.DEFAULT_UD_ROOT / "safe_eval.py",
        f"# {gate.SEAM_SHIM_MARKER}\nfrom shared.python.safe_eval import *\n",
    )
    violations, _ = gate.check(repo)
    assert violations == []


def test_split_cleaned_with_overlap_fails_and_ud_only_is_fine(repo: Path) -> None:
    rulings = _base_rulings()
    rulings["theme"]["status"] = "cleaned"
    _rulings(repo, rulings)
    violations, _ = gate.check(repo)
    assert [v.package for v in violations] == ["theme"]
    (repo / gate.DEFAULT_UD_ROOT / "theme/colors.py").unlink()
    _write(
        repo / gate.DEFAULT_UD_ROOT / "theme/__init__.py",
        f"# {gate.SEAM_SHIM_MARKER}\n",
    )
    violations, _ = gate.check(repo)
    assert violations == []


def test_ud_canonical_needs_ledger_row(repo: Path) -> None:
    rulings = _base_rulings()
    rulings["model_generation"]["tools_ledger_row"] = ""
    _rulings(repo, rulings)
    violations, _ = gate.check(repo)
    assert [v.package for v in violations] == ["model_generation"]
    assert "Tools #4915" in violations[0].message


def test_vendor_entry_without_ruling_fails(repo: Path) -> None:
    _write(repo / gate.DEFAULT_TOOLS_ROOT / "brand_new_pkg/__init__.py", "n = 1\n")
    _rulings(repo, _base_rulings())
    violations, _ = gate.check(repo)
    assert [v.package for v in violations] == ["brand_new_pkg"]


def test_deferred_requires_reason(repo: Path) -> None:
    rulings = _base_rulings()
    rulings["theme"] = {
        "ruling": "deferred",
        "status": "pending-cleanup",
        "rationale": "r",
    }
    _rulings(repo, rulings)
    violations, _ = gate.check(repo)
    assert [v.package for v in violations] == ["theme"]


def test_invalid_ruling_value_is_rejected(repo: Path) -> None:
    rulings = _base_rulings()
    rulings["theme"]["ruling"] = "whatever"
    _rulings(repo, rulings)
    with pytest.raises(ValueError, match="invalid 'ruling'"):
        gate.check(repo)


def test_cli_exit_codes(repo: Path) -> None:
    _rulings(repo, _base_rulings())
    assert gate.main(["--repo-root", str(repo), "--quiet"]) == 0
    assert gate.main(["--repo-root", str(repo), "--strict"]) == 1
    (repo / gate.DEFAULT_TOOLS_ROOT).rename(repo / "gone")
    assert gate.main(["--repo-root", str(repo)]) == 2


def test_committed_rulings_pass_on_this_checkout() -> None:
    """Fail closed unless opted out via SEAM_TESTS_ALLOW_SKIP=1."""
    from tests.helpers.seam_guards import require_vendor_path

    repo_root = Path(__file__).resolve().parents[3]
    vendor_root = repo_root / gate.DEFAULT_TOOLS_ROOT
    require_vendor_path(vendor_root)

    violations, _ = gate.check(repo_root)
    assert violations == [], "\n".join(str(v) for v in violations)
