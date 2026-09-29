"""Tests for scripts/shared_tools/check_tools_pins.py (UD #9406)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.shared_tools import check_tools_pins as pins

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SHA = "c0a395d59ec0a78aa70d4a989ccfc8f0a9605319"


def _cargo(rev: str) -> str:
    return (
        "[workspace.dependencies]\n"
        f'tools-core = {{ git = "{pins.TOOLS_REPO_URL}", rev = "{rev}" }}\n'
    )


def test_read_cargo_pin(tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo(_SHA), encoding="utf-8")
    assert pins.read_cargo_pin(cargo) == (pins.TOOLS_REPO_URL, _SHA)
    assert pins.read_cargo_pin(tmp_path / "missing.toml") == (None, None)


def test_read_pyproject_pins(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'tools = [\n    "ud-tools @ git+{pins.TOOLS_REPO_URL}@{_SHA}",\n]\n',
        encoding="utf-8",
    )
    assert pins.read_pyproject_pins(pyproject) == [_SHA]


def test_check_pins_consistent() -> None:
    result = pins.check_pins(
        [
            pins.Pin("submodule gitlink", "vendor/ud-tools", _SHA),
            pins.Pin("Cargo.toml tools-core", "Cargo.toml", _SHA),
            pins.Pin("pyproject ud-tools pin #1", "pyproject.toml", _SHA[:12]),
        ]
    )
    assert result == []


def test_check_pins_reports_mismatch_and_missing() -> None:
    result = pins.check_pins(
        [
            pins.Pin("submodule gitlink", "vendor/ud-tools", _SHA),
            pins.Pin("Cargo.toml tools-core", "Cargo.toml", "ea26903624"),
            pins.Pin("pyproject ud-tools pin #1", "pyproject.toml", None),
        ]
    )
    assert len(result) == 2
    assert "Cargo.toml" in result[0] and "gitlink" in result[0]
    assert "no Tools pin found" in result[1]


def test_check_pins_without_gitlink() -> None:
    assert pins.check_pins([pins.Pin("submodule gitlink", "vendor/ud-tools", None)])


def test_main_on_this_checkout_is_consistent() -> None:
    """The Cargo pin must equal the vendor/ud-tools gitlink on every commit."""
    repo_root = Path(__file__).resolve().parents[3]
    assert pins.main(["--repo-root", str(repo_root)]) == 0


def test_wheel_pin_is_reported_not_compared(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        'tools = [\n    "ud_tools @ https://github.com/D-sorganization/Tools/releases/'
        'download/v1.15.0/ud_tools-1.15.0-py3-none-any.whl",\n]\n',
        encoding="utf-8",
    )
    assert pins.read_pyproject_wheel_pins(pyproject) == [("1.15.0", "1.15.0")]
    result = pins.check_pins(
        [
            pins.Pin("submodule gitlink", "vendor/ud-tools", _SHA),
            pins.Pin("Cargo.toml tools-core", "Cargo.toml", _SHA),
            pins.Pin(
                "pyproject ud_tools wheel v1.15.0", "pyproject.toml", "WHEEL:1.15.0"
            ),
        ]
    )
    assert result == []
