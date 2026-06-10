"""Tests for the legacy golf visualizer data processor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

_VISUALIZER_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "src"
    / "apps"
    / "golf_gui"
    / "Simscape Multibody Data Plotters"
    / "Python Version"
    / "golf_gui_r0"
)


def _load_data_processor() -> type[Any]:
    sys.path.insert(0, str(_VISUALIZER_DIR))
    spec = importlib.util.spec_from_file_location(
        "golf_visualizer_data", _VISUALIZER_DIR / "golf_visualizer_data.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load golf_visualizer_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DataProcessor


class _CountingIloc:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe
        self.calls = 0

    def __getitem__(self, index: int) -> pd.Series:
        self.calls += 1
        return self._dataframe.iloc[index]


class _CountingFrame:
    def __init__(self, data: dict[str, list[Any]]) -> None:
        self._dataframe = pd.DataFrame(data)
        self.iloc = _CountingIloc(self._dataframe)


def _dataset(force: list[float], torque: list[float]) -> _CountingFrame:
    return _CountingFrame(
        {
            "Butt": [[1.0, 2.0, 3.0]],
            "Clubhead": [[4.0, 5.0, 6.0]],
            "MidPoint": [[7.0, 8.0, 9.0]],
            "LeftWrist": [[10.0, 11.0, 12.0]],
            "LeftElbow": [[13.0, 14.0, 15.0]],
            "LeftShoulder": [[16.0, 17.0, 18.0]],
            "RightWrist": [[19.0, 20.0, 21.0]],
            "RightElbow": [[22.0, 23.0, 24.0]],
            "RightShoulder": [[25.0, 26.0, 27.0]],
            "Hub": [[28.0, 29.0, 30.0]],
            "TotalHandForceGlobal": [force],
            "EquivalentMidpointCoupleGlobal": [torque],
        }
    )


def test_extract_frame_data_fetches_each_dataset_row_once() -> None:
    processor = _load_data_processor()()
    datasets = {
        "BASEQ": _dataset([1.0, 0.0, 0.0], [0.1, 0.0, 0.0]),
        "ZTCFQ": _dataset([0.0, 2.0, 0.0], [0.0, 0.2, 0.0]),
        "DELTAQ": _dataset([0.0, 0.0, 3.0], [0.0, 0.0, 0.3]),
    }

    frame_data = processor.extract_frame_data(0, datasets)

    for dataset in datasets.values():
        assert dataset.iloc.calls == 1
    np.testing.assert_allclose(frame_data.clubhead, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(frame_data.forces["BASEQ"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(frame_data.torques["DELTAQ"], [0.0, 0.0, 0.3])
