"""Tests for scripts/_extract_dims.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import _extract_dims as mod


def _write_csv(path: Path, header: list[str], row: list[str]) -> None:
    text = ",".join(header) + "\n" + ",".join(row) + "\n"
    path.write_text(text, encoding="utf-8")


def test_main_keeps_model_prefixed_dimensions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(
        csv_path,
        ["model_PelvisLength", "model_OtherThing", "model_TorsoMass", "junk"],
        ["0.25", "999", "12.0", "x"],
    )

    mod.main(csv_path)
    out = json.loads(capsys.readouterr().out)
    assert out == {"model_PelvisLength": "0.25", "model_TorsoMass": "12.0"}


def test_main_keeps_explicit_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(
        csv_path,
        ["ClubLogs_ClubMass", "SegmentInertiaLogs_GolferMass", "noise"],
        ["0.2", "75.0", "x"],
    )
    mod.main(csv_path)
    out = json.loads(capsys.readouterr().out)
    assert out == {
        "ClubLogs_ClubMass": "0.2",
        "SegmentInertiaLogs_GolferMass": "75.0",
    }


def test_main_keeps_segment_inertia_logs_inertia(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(
        csv_path,
        ["SegmentInertiaLogs_TorsoInertia", "SegmentInertiaLogs_FootCOM", "other"],
        ["0.5", "0.1", "n"],
    )
    mod.main(csv_path)
    out = json.loads(capsys.readouterr().out)
    assert "SegmentInertiaLogs_TorsoInertia" in out
    assert "SegmentInertiaLogs_FootCOM" in out
    assert "other" not in out


def test_main_drops_unmatched_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, ["random_col", "model_NoKeepSubstring"], ["1", "2"])
    mod.main(csv_path)
    out = json.loads(capsys.readouterr().out)
    assert out == {}
