"""Tests for scripts/ci/check_suppression_ratchet.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "check_suppression_ratchet.py"


def _load_ratchet_module() -> object:
    """Load the script as a module."""
    spec = importlib.util.spec_from_file_location("suppression_ratchet", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["suppression_ratchet"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_baseline(path: pathlib.Path, counts: dict[str, int]) -> None:
    path.write_text(
        json.dumps({"_comment": "test", "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_src_file(
    src_dir: pathlib.Path,
    *,
    bare_type_ignore: int = 0,
    coded_type_ignore: int = 0,
    bare_noqa: int = 0,
    coded_noqa: int = 0,
) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.extend(["x = 1  # type: ignore"] * bare_type_ignore)
    lines.extend(["y = 2  # type: ignore[attr-defined]"] * coded_type_ignore)
    lines.extend(["z = 3  # noqa"] * bare_noqa)
    lines.extend(["w = 4  # noqa: F841"] * coded_noqa)
    (src_dir / "sample.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def ratchet_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_ratchet_module()
    src_dir = tmp_path / "src"
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(mod, "SRC_DIR", src_dir)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return mod, src_dir, baseline_path


def test_passes_when_counts_equal_baseline(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir, bare_type_ignore=2, bare_noqa=1, coded_noqa=3)
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 2,
            "bare_noqa": 1,
        },
    )
    assert mod.main([]) == 0


def test_coded_suppressions_do_not_count(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir, coded_type_ignore=2, coded_noqa=2)
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 0,
            "bare_noqa": 0,
        },
    )
    assert mod.main([]) == 0


def test_fails_when_any_count_grows(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir, bare_type_ignore=3, bare_noqa=2)
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 2,
            "bare_noqa": 1,
        },
    )
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 1
    assert any("bare_type_ignore" in record.message for record in caplog.records)
    assert any("bare_noqa" in record.message for record in caplog.records)


def test_passes_and_reports_improvements(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir, bare_type_ignore=1)
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 4,
            "bare_noqa": 1,
        },
    )
    with caplog.at_level("INFO"):
        assert mod.main([]) == 0
    assert any("IMPROVED" in record.message for record in caplog.records)


def test_exit_2_when_baseline_missing(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir)
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 2
    assert any("baseline" in record.message.lower() for record in caplog.records)


def test_exit_2_when_baseline_missing_keys(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir)
    baseline_path.write_text(
        json.dumps({"counts": {"bare_type_ignore": 0}}), encoding="utf-8"
    )
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 2
    assert any("missing" in record.message.lower() for record in caplog.records)


def test_update_baseline_rewrites_counts(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir, bare_type_ignore=1, bare_noqa=2)
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 3,
            "bare_noqa": 4,
        },
    )
    assert mod.main(["--update-baseline"]) == 0
    new = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert new["counts"] == {"bare_type_ignore": 1, "bare_noqa": 2}


def test_update_baseline_blocked_on_regression(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_file(src_dir, bare_type_ignore=4)
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 1,
            "bare_noqa": 0,
        },
    )
    assert mod.main(["--update-baseline"]) == 1
    new = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert new["counts"] == {"bare_type_ignore": 1, "bare_noqa": 0}


def test_count_patterns_skips_non_utf8_file(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "binary.py").write_bytes(b"\xff\xfe\x00not utf8")
    _write_baseline(
        baseline_path,
        {
            "bare_type_ignore": 0,
            "bare_noqa": 0,
        },
    )
    assert mod.main([]) == 0
