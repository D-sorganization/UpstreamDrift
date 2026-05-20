"""Shared fixtures for wave7 model-library tests."""

from __future__ import annotations

from pathlib import Path

import pytest

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link"/>
  <link name="arm_link"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="1" lower="-1.57" upper="1.57"/>
  </joint>
</robot>
"""


@pytest.fixture
def simple_urdf(tmp_path: Path) -> Path:
    p = tmp_path / "simple.urdf"
    p.write_text(SIMPLE_URDF)
    return p


@pytest.fixture
def lib_config(tmp_path: Path):
    from model_generation.library._model_types import LibraryConfig

    return LibraryConfig(
        cache_dir=tmp_path / "lib_cache",
        index_file=tmp_path / "lib_index.json",
    )
