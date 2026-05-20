"""Tests for simulation store: listing and cleanup."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.shared.python.data_io._simulation_store import (
    cleanup_old_files,
    get_simulation_list,
)


def _set_old_mtime(p: Path, days_old: int) -> None:
    age_seconds = days_old * 86400
    target = time.time() - age_seconds
    os.utime(p, (target, target))


def test_get_simulation_list_empty_directory(tmp_path: Path):
    assert get_simulation_list(tmp_path) == []


def test_get_simulation_list_missing_directory(tmp_path: Path):
    assert get_simulation_list(tmp_path / "nope") == []


def test_get_simulation_list_engine_filter(tmp_path: Path):
    mj = tmp_path / "mujoco"
    mj.mkdir()
    (mj / "a.csv").write_text("a")
    (mj / "b.csv").write_text("b")
    drake = tmp_path / "drake"
    drake.mkdir()
    (drake / "z.csv").write_text("z")

    assert get_simulation_list(tmp_path, engine="mujoco") == ["a.csv", "b.csv"]
    assert get_simulation_list(tmp_path, engine="drake") == ["z.csv"]


def test_get_simulation_list_engine_missing(tmp_path: Path):
    assert get_simulation_list(tmp_path, engine="nonexistent") == []


def test_get_simulation_list_all_engines(tmp_path: Path):
    (tmp_path / "loose.csv").write_text("x")
    mj = tmp_path / "mujoco"
    mj.mkdir()
    (mj / "a.csv").write_text("a")
    drake = tmp_path / "drake"
    drake.mkdir()
    (drake / "z.csv").write_text("z")

    result = get_simulation_list(tmp_path)
    assert "loose.csv" in result
    assert "a.csv" in result
    assert "z.csv" in result


def test_cleanup_old_files_negative_age_raises(tmp_path: Path):
    with pytest.raises(Exception):  # noqa: B017 — DbC precondition
        cleanup_old_files(
            cache_dir=tmp_path / "cache",
            simulations_dir=tmp_path / "sims",
            analysis_dir=tmp_path / "analysis",
            base_path=tmp_path,
            max_age_days=0,
        )


def test_cleanup_old_files_removes_old_temp(tmp_path: Path):
    cache = tmp_path / "cache"
    temp = cache / "temp"
    temp.mkdir(parents=True)
    old = temp / "old.txt"
    old.write_text("x")
    _set_old_mtime(old, days_old=5)

    fresh = temp / "fresh.txt"
    fresh.write_text("x")

    sims = tmp_path / "sims"
    sims.mkdir()
    analysis = tmp_path / "analysis"
    analysis.mkdir()

    count = cleanup_old_files(
        cache_dir=cache,
        simulations_dir=sims,
        analysis_dir=analysis,
        base_path=tmp_path,
        max_age_days=30,
    )
    assert count >= 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_old_files_archives_old_simulations(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sims = tmp_path / "sims"
    sims.mkdir()
    analysis = tmp_path / "analysis"
    analysis.mkdir()

    old = sims / "old_sim.csv"
    old.write_text("data")
    _set_old_mtime(old, days_old=60)

    count = cleanup_old_files(
        cache_dir=cache,
        simulations_dir=sims,
        analysis_dir=analysis,
        base_path=tmp_path,
        max_age_days=30,
    )
    assert count >= 1
    assert not old.exists()
    # archived path should exist
    archive = tmp_path / "archive"
    assert archive.exists()


def test_cleanup_old_files_keeps_fresh(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sims = tmp_path / "sims"
    sims.mkdir()
    analysis = tmp_path / "analysis"
    analysis.mkdir()

    fresh = sims / "fresh.csv"
    fresh.write_text("data")

    count = cleanup_old_files(
        cache_dir=cache,
        simulations_dir=sims,
        analysis_dir=analysis,
        base_path=tmp_path,
        max_age_days=30,
    )
    assert count == 0
    assert fresh.exists()
