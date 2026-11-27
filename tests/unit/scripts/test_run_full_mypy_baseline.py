from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import run_full_mypy_baseline as mod


def _error(
    path: str = "src/package/module.py",
    line: int = 12,
    message: str = 'Incompatible return value type (got "str", expected "int")',
    code: str = "return-value",
) -> mod.MypyError:
    return mod.MypyError(
        path=path,
        line=line,
        column=None,
        severity="error",
        message=message,
        code=code,
    )


def test_parse_mypy_line_normalizes_error_code_and_windows_path() -> None:
    parsed = mod.parse_mypy_line(
        "src\\package\\module.py:12: error: "
        'Incompatible return value type (got "str", expected "int") [return-value]'
    )
    constructor_note = mod.parse_mypy_line(
        "src\\gui\\calibration_dialog.py:257: note:     def __new__(cls, "
        "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc = ..., /) -> int"
    )

    assert parsed == _error()
    assert constructor_note == mod.MypyError(
        path="src/gui/calibration_dialog.py",
        line=257,
        column=None,
        severity="note",
        message=(
            "def int(str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc = "
            "..., /) -> int"
        ),
        code=None,
    )


def test_compare_to_baseline_reports_new_and_stale_errors() -> None:
    current = [_error(path="src/new.py")]
    baseline = [_error(path="src/old.py")]

    new_errors, stale_errors = mod.compare_to_baseline(current, baseline)

    assert new_errors == current
    assert stale_errors == baseline


def test_main_passes_when_current_errors_match_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "baseline.json"
    expected = _error()
    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": "python -m mypy src --config-file pyproject.toml",
                "errors": [expected.to_json()],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod, "run_mypy", lambda target, config_file: (1, _line(expected))
    )

    assert mod.main(["--baseline", str(baseline)]) == 0


def test_main_fails_when_mypy_reports_new_error(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": "python -m mypy src --config-file pyproject.toml",
                "errors": [_error(path="src/old.py").to_json()],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod, "run_mypy", lambda target, config_file: (1, _line(_error()))
    )

    assert mod.main(["--baseline", str(baseline)]) == 1


def test_update_baseline_writes_current_errors(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    expected = _error(path="src/current.py")
    monkeypatch.setattr(
        mod, "run_mypy", lambda target, config_file: (1, _line(expected))
    )

    assert mod.main(["--baseline", str(baseline), "--update-baseline"]) == 0

    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert data["errors"] == [expected.to_json()]


@pytest.mark.unit
def test_ndarray_stub_message_variants_are_distinct_baseline_keys() -> None:
    """SciPy-stubs 1.17+ rephrases ndarray diagnostics; refresh the baseline when they drift."""
    stale = _error(
        path="src/api/services/recording_service.py",
        line=173,
        message=(
            'Argument 2 to "savez_compressed" has incompatible type '
            '"**dict[str, ndarray[tuple[Any, ...], dtype[Any]]]"; expected "bool"'
        ),
        code="arg-type",
    )
    current = _error(
        path="src/api/services/recording_service.py",
        line=173,
        message=(
            'Argument 2 to "savez_compressed" has incompatible type '
            '"**dict[str, ndarray[Any, Any]]"; expected "bool"'
        ),
        code="arg-type",
    )

    new_errors, stale_errors = mod.compare_to_baseline([current], [stale])

    assert new_errors == [current]
    assert stale_errors == [stale]


def test_ci_standard_uses_full_src_baseline_runner_on_push() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    workflow = (repo_root / ".github/workflows/ci-standard.yml").read_text(
        encoding="utf-8"
    )

    assert "python3 scripts/ci/run_full_mypy_baseline.py" in workflow
    assert "mypy src --config-file pyproject.toml" not in workflow


def _line(error: mod.MypyError) -> str:
    code = f" [{error.code}]" if error.code else ""
    return f"{error.path}:{error.line}: {error.severity}: {error.message}{code}\n"
