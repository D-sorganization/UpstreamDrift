"""Tests for scripts/ci/check_error_handling_ratchet.py.

Verifies the ratchet:
  * passes when counts equal baseline
  * passes (and reports) when counts decrease
  * fails when any count grows
  * exits 2 on configuration errors
  * --update-baseline rewrites only the counts field
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "check_error_handling_ratchet.py"


def _load_ratchet_module() -> object:
    """Load the script as a module (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("ratchet", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ratchet"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_baseline(path: pathlib.Path, counts: dict[str, int]) -> None:
    path.write_text(
        json.dumps({"_comment": "test", "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_src_with_patterns(
    src_dir: pathlib.Path,
    *,
    ble: int = 0,
    f841: int = 0,
    f401: int = 0,
    popen: int = 0,
    gather: int = 0,
) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.extend(["x = 1  # noqa: BLE001"] * ble)
    lines.extend(["y = 2  # noqa: F841"] * f841)
    lines.extend(["import os  # noqa: F401"] * f401)
    lines.extend(["subprocess.Popen([])"] * popen)
    lines.extend(["await asyncio.gather(coro())"] * gather)
    (src_dir / "sample.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def ratchet_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_ratchet_module()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(mod, "SRC_DIR", src_dir)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SELF_EXEMPT", set())
    return mod, src_dir, baseline_path


def test_passes_when_counts_equal_baseline(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, ble=3, popen=1)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 3,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 1,
            "gather_no_return_exceptions": 0,
        },
    )
    assert mod.main([]) == 0


def test_passes_when_counts_decrease(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, ble=1)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 5,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    with caplog.at_level("INFO"):
        assert mod.main([]) == 0
    assert any("IMPROVED" in r.message for r in caplog.records)


def test_fails_when_any_count_grows(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, ble=10)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 5,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 1
    assert any("REGRESSION" in r.message for r in caplog.records)
    assert any("noqa_BLE001" in r.message for r in caplog.records)


def test_detects_raw_popen_growth(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, popen=2)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 0,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 1,
            "gather_no_return_exceptions": 0,
        },
    )
    assert mod.main([]) == 1


def test_gather_without_return_exceptions_detected(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, gather=2)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 0,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 1,
        },
    )
    assert mod.main([]) == 1


def test_gather_with_return_exceptions_not_counted(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    (src_dir / "ok.py").write_text(
        "await asyncio.gather(coro(), return_exceptions=True)\n",
        encoding="utf-8",
    )
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 0,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    assert mod.main([]) == 0


def test_multiline_gather_with_return_exceptions_not_counted(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    (src_dir / "ok_multiline.py").write_text(
        "await asyncio.gather(\n"
        "    coro(),\n"
        "    other_coro(arg()),\n"
        "    return_exceptions=True,\n"
        ")\n",
        encoding="utf-8",
    )
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 0,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    assert mod.main([]) == 0


def test_multiline_gather_without_return_exceptions_detected(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    (src_dir / "bad_multiline.py").write_text(
        "await asyncio.gather(\n    coro(),\n    other_coro(arg()),\n)\n",
        encoding="utf-8",
    )
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 0,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    assert mod.main([]) == 1


def test_exit_2_when_baseline_missing(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir)
    # baseline_path intentionally not written
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 2
    assert any("baseline" in r.message.lower() for r in caplog.records)


def test_exit_2_when_baseline_missing_keys(ratchet_env, caplog):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir)
    baseline_path.write_text(
        json.dumps({"counts": {"noqa_BLE001": 0}}), encoding="utf-8"
    )
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 2
    assert any("missing" in r.message.lower() for r in caplog.records)


def test_update_baseline_rewrites_counts(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, ble=2)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 5,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    assert mod.main(["--update-baseline"]) == 0
    new = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert new["counts"]["noqa_BLE001"] == 2


def test_update_baseline_blocked_on_regression(ratchet_env):
    mod, src_dir, baseline_path = ratchet_env
    _write_src_with_patterns(src_dir, ble=10)
    _write_baseline(
        baseline_path,
        {
            "noqa_BLE001": 5,
            "noqa_F841": 0,
            "noqa_F401": 0,
            "raw_popen": 0,
            "gather_no_return_exceptions": 0,
        },
    )
    # main should fail before reaching the update branch
    assert mod.main(["--update-baseline"]) == 1
    new = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert new["counts"]["noqa_BLE001"] == 5  # unchanged
