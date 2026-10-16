"""``model_as_urdf``: the explorer's single model-file-to-URDF entry (#12878)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.tools.model_explorer.model_source import model_as_urdf
from src.tools.model_explorer.results_session import ResultsSession

pytestmark = [pytest.mark.unit]

URDF = '<robot name="r"><link name="base"/></robot>'
MJCF = (
    '<mujoco><worldbody><body name="b"><geom type="sphere" size="0.1"/>'
    "</body></worldbody></mujoco>"
)


def test_urdf_text_is_returned_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "r.urdf"
    path.write_text(URDF, encoding="utf-8")
    assert model_as_urdf(path) == URDF


def test_a_file_without_bodies_is_rejected_by_name(tmp_path: Path) -> None:
    path = tmp_path / "empty.urdf"
    path.write_text('<robot name="r"/>', encoding="utf-8")
    with pytest.raises(ValueError, match="empty.urdf"):
        model_as_urdf(path)


def test_plain_xml_that_is_not_mjcf_is_read_as_urdf(tmp_path: Path) -> None:
    path = tmp_path / "robot.xml"
    path.write_text(URDF, encoding="utf-8")
    assert model_as_urdf(path) == URDF


def test_mjcf_is_converted_to_urdf_links(tmp_path: Path) -> None:
    path = tmp_path / "scene.xml"
    path.write_text(MJCF, encoding="utf-8")
    urdf = model_as_urdf(path)
    assert "<link" in urdf


def test_missing_file_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        model_as_urdf(tmp_path / "absent.urdf")


def test_results_session_suggests_an_mp4_named_after_the_archive() -> None:
    source = cast(Any, SimpleNamespace(frame_count=2))
    session = ResultsSession(Path("runs/swing_07.npz"), source, "mujoco")
    assert session.default_video_name == "swing_07.mp4"
