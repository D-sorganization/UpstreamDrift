"""Tests for src/api/dependencies.py and src/api/routes/_route_utils.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import dependencies as deps
from src.api.routes import _route_utils

pytestmark = pytest.mark.unit


def _make_request(**state: object) -> SimpleNamespace:
    app = SimpleNamespace(state=SimpleNamespace(**state))
    return SimpleNamespace(app=app)


def test_get_engine_manager_missing() -> None:
    req = _make_request()
    with pytest.raises(HTTPException) as exc:
        deps.get_engine_manager(req)  # type: ignore[arg-type]
    assert exc.value.status_code == 503


def test_get_engine_manager_present() -> None:
    sentinel = object()
    req = _make_request(engine_manager=sentinel)
    assert deps.get_engine_manager(req) is sentinel  # type: ignore[arg-type]


def test_get_simulation_service_missing() -> None:
    with pytest.raises(HTTPException) as exc:
        deps.get_simulation_service(_make_request())  # type: ignore[arg-type]
    assert exc.value.status_code == 503


def test_get_simulation_service_present() -> None:
    sentinel = object()
    assert (
        deps.get_simulation_service(_make_request(simulation_service=sentinel))  # type: ignore[arg-type]
        is sentinel
    )


def test_get_analysis_service_missing_and_present() -> None:
    with pytest.raises(HTTPException):
        deps.get_analysis_service(_make_request())  # type: ignore[arg-type]
    s = object()
    assert (
        deps.get_analysis_service(_make_request(analysis_service=s))  # type: ignore[arg-type]
        is s
    )


def test_get_video_pipeline_missing_message() -> None:
    with pytest.raises(HTTPException) as exc:
        deps.get_video_pipeline(_make_request())  # type: ignore[arg-type]
    assert "Video pipeline" in exc.value.detail


def test_get_video_pipeline_present() -> None:
    s = object()
    assert deps.get_video_pipeline(_make_request(video_pipeline=s)) is s  # type: ignore[arg-type]


def test_get_task_manager_missing_and_present() -> None:
    with pytest.raises(HTTPException):
        deps.get_task_manager(_make_request())  # type: ignore[arg-type]
    s = object()
    assert deps.get_task_manager(_make_request(task_manager=s)) is s  # type: ignore[arg-type]


def test_get_logger_none_when_missing() -> None:
    assert deps.get_logger(_make_request()) is None  # type: ignore[arg-type]


def test_get_logger_returns_state() -> None:
    s = object()
    assert deps.get_logger(_make_request(logger=s)) is s  # type: ignore[arg-type]


def test_find_project_root_returns_directory() -> None:
    root = _route_utils.find_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    # pyproject.toml or src/shared/urdf marker should be in result tree
    assert (
        (root / "pyproject.toml").exists()
        or (root / "src" / "shared" / "urdf").exists()
        or root == Path.cwd()
    )
