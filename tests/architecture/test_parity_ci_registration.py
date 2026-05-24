from __future__ import annotations

from pathlib import Path


def test_pyproject_registers_parity_marker() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "parity: cross-implementation parity" in pyproject


def test_ci_standard_defines_scheduled_parity_smoke_lane() -> None:
    workflow = Path(".github/workflows/ci-standard.yml").read_text(encoding="utf-8")
    assert "parity-smoke:" in workflow
    assert "github.event_name == 'schedule'" in workflow
    assert "tests/api/test_api_parity.py" in workflow
    assert "tests/config/launcher_manifest/test_parity.py" in workflow
    assert "tests/parity/test_fitresult_field_parity.py" in workflow
    assert "tests/unit/theme/test_sidekick_parity.py" in workflow
    assert "-m parity" in workflow
