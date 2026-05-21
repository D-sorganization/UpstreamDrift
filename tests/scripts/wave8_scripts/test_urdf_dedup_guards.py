"""Tests for scripts/_urdf_dedup_guards.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import _urdf_dedup_guards as mod


def test_dedup_file_removes_duplicate_guard(tmp_path: Path) -> None:
    src = (
        "def f(x):\n"
        "    if not (x is not None):\n"
        '        raise ValueError("x must be provided")\n'
        "    if not (x is not None):\n"
        '        raise ValueError("x must be provided")\n'
        "    return x\n"
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    n = mod.dedup_file(p)
    assert n == 1
    out = p.read_text()
    assert out.count("if not (x is not None):") == 1
    assert "return x" in out


def test_dedup_file_no_change_for_single_guard(tmp_path: Path) -> None:
    src = (
        "def f(x):\n"
        "    if not (x is not None):\n"
        '        raise ValueError("x must be provided")\n'
        "    return x\n"
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    original_mtime = p.stat().st_mtime
    n = mod.dedup_file(p)
    assert n == 0
    # File should not be rewritten
    assert p.read_text() == src
    assert p.stat().st_mtime == original_mtime


def test_dedup_file_handles_tabs(tmp_path: Path) -> None:
    indent = "\t"
    src = (
        "def f(x):\n"
        f"{indent}if not (x is not None):\n"
        f'{indent}    raise ValueError("bad")\n'
        f"{indent}if not (x is not None):\n"
        f'{indent}    raise ValueError("bad")\n'
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    n = mod.dedup_file(p)
    assert n == 1


def test_dedup_file_does_not_collapse_different_names(tmp_path: Path) -> None:
    src = (
        "def f(x, y):\n"
        "    if not (x is not None):\n"
        '        raise ValueError("x")\n'
        "    if not (y is not None):\n"
        '        raise ValueError("y")\n'
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    n = mod.dedup_file(p)
    assert n == 0


def test_dedup_file_multiple_dups(tmp_path: Path) -> None:
    block = (
        "    if not (a is not None):\n"
        '        raise ValueError("a")\n'
        "    if not (a is not None):\n"
        '        raise ValueError("a")\n'
    )
    src = "def f(a):\n" + block + "    a = 1\n" + block
    p = tmp_path / "m.py"
    p.write_text(src)
    n = mod.dedup_file(p)
    assert n == 2
    assert p.read_text().count("if not (a is not None):") == 2


def test_main_runs_against_synthetic_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Set up a fake src/shared/python tree
    root = tmp_path / "src" / "shared" / "python"
    hcb = root / "humanoid_character_builder"
    mg = root / "model_generation"
    hcb.mkdir(parents=True)
    mg.mkdir(parents=True)

    dup = (
        "def f(x):\n"
        "    if not (x is not None):\n"
        '        raise ValueError("x must be provided")\n'
        "    if not (x is not None):\n"
        '        raise ValueError("x must be provided")\n'
    )
    (hcb / "a.py").write_text(dup)
    (mg / "b.py").write_text(dup)
    # cache dir should be skipped
    cache = hcb / "__pycache__"
    cache.mkdir()
    (cache / "c.py").write_text(dup)
    # unrelated file with no dup
    (mg / "clean.py").write_text("x = 1\n")

    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "TARGETS", [hcb, mg])

    mod.main()
    out = capsys.readouterr().out
    assert "Deduplicated 2 guard blocks" in out
    assert "2 files" in out
    # cached file untouched
    assert (cache / "c.py").read_text() == dup


def test_main_no_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    (target / "x.py").write_text("y = 1\n")
    monkeypatch.setattr(mod, "TARGETS", [target])
    mod.main()
    out = capsys.readouterr().out
    assert "Deduplicated 0 guard blocks across 0 files" in out
