"""Tests for the pure-data spine visualization adapter (SPN-15a, #12771)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.spine import viz_data
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import (
    LoadObservable,
    SpinalAxialLoadProvider,
    SpinalLoadSeries,
)
from src.shared.python.spine.segmental import LevelTransform, SegmentalPose
from src.shared.python.spine.viz_data import (
    HeatmapData,
    StackData,
    level_angle_stack,
    load_heatmap,
    to_tension_positive,
)

pytestmark = pytest.mark.unit

LEVELS = (SpinalLevel.L5_S1, SpinalLevel.L4_L5, SpinalLevel.L3_L4)
SOURCE = "invented-uniform-test-fixture"


def _pose(*axial_deg: float) -> SegmentalPose:
    return SegmentalPose(
        tuple(
            LevelTransform.from_isb_angles(lv, np.deg2rad([0.0, 0.0, -a]))
            for lv, a in zip(LEVELS, axial_deg, strict=False)
        )
    )


def _series(force: np.ndarray | None = None) -> SpinalLoadSeries:
    n = 4
    if force is None:
        force = np.zeros((n, 3, 3))
        force[:, :, 1] = np.array([100.0, 200.0, 300.0]) * np.arange(1, n + 1)[:, None]
        force[:, :, 0] = -5.0
        force[:, :, 2] = 7.0
    return SpinalLoadSeries(
        time_s=np.linspace(0.0, 0.3, force.shape[0]),
        levels=LEVELS,
        observable=LoadObservable.JOINT_REACTION,
        force_n=force,
        moment_nm=np.zeros_like(force),
        provenance={
            "model_id": "invented",
            "model_sha256": "0" * 64,
            "solver": "JointReaction",
            "cost_exponent": "n/a",
            "distribution_source": SOURCE,
            "external_loads": "none",
        },
    )


def _stack(**kw) -> StackData:
    poses = [_pose(10, 15, 20), _pose(20, 30, 40)]
    args = {
        "handedness": "right",
        "distribution_source": SOURCE,
        "scientific_status": "unqualified",
        **kw,
    }
    return level_angle_stack(np.array([0.0, 0.1]), poses, "x_factor_jcs_axial", **args)


def test_stack_shapes_order_units_and_metadata() -> None:
    data = _stack()
    assert data.values_deg.shape == (2, 3)
    assert data.levels == LEVELS  # caudal -> cranial
    assert data.units == "deg"
    assert data.term_id == "x_factor_jcs_axial"
    assert data.distribution_source == SOURCE
    assert data.scientific_status == "unqualified"
    assert data.values_deg[0] == pytest.approx([10.0, 15.0, 20.0])
    assert data.values_deg[1] == pytest.approx([20.0, 30.0, 40.0])
    assert data.remainder_deg.shape == (2,)
    assert np.allclose(data.remainder_deg, 0.0, atol=1e-9)
    assert not data.values_deg.flags.writeable
    assert not data.time_s.flags.writeable


def test_stack_totals_close_to_global_term() -> None:
    data = _stack()
    assert data.total_deg() == pytest.approx([45.0, 90.0])


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"distribution_source": ""}, ValueError),
        ({"distribution_source": 3}, TypeError),
        ({"scientific_status": "validated"}, ValueError),
        ({"handedness": "ambi"}, ValueError),
    ],
)
def test_stack_rejects_bad_metadata(kwargs, exc) -> None:
    with pytest.raises(exc):
        _stack(**kwargs)


def test_stack_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        level_angle_stack(
            np.array([0.0]),
            [_pose(1, 2, 3), _pose(1, 2, 3)],
            "x_factor_jcs_axial",
            handedness="right",
            distribution_source=SOURCE,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        level_angle_stack(
            np.array([0.1, 0.1]),
            [_pose(1, 2, 3), _pose(1, 2, 3)],
            "x_factor_jcs_axial",
            handedness="right",
            distribution_source=SOURCE,
        )
    with pytest.raises(ValueError, match="same levels"):
        level_angle_stack(
            np.array([0.0, 0.1]),
            [_pose(1, 2, 3), _pose(1, 2)],
            "x_factor_jcs_axial",
            handedness="right",
            distribution_source=SOURCE,
        )
    with pytest.raises(ValueError, match="non-empty"):
        level_angle_stack(
            np.array([]),
            [],
            "x_factor_jcs_axial",
            handedness="right",
            distribution_source=SOURCE,
        )
    with pytest.raises(TypeError, match="SegmentalPose"):
        level_angle_stack(
            np.array([0.0]),
            ["nope"],
            "x_factor_jcs_axial",
            handedness="right",
            distribution_source=SOURCE,
        )
    with pytest.raises(ValueError, match="no segmental decomposition"):
        level_angle_stack(
            np.array([0.0]),
            [_pose(1, 2, 3)],
            "side_bend_tilt",
            handedness="right",
            distribution_source=SOURCE,
        )


def test_to_tension_positive_flips_sign_only() -> None:
    comp = np.array([[100.0, -20.0]])
    out = to_tension_positive(comp)
    assert out.tolist() == [[-100.0, 20.0]]
    assert comp.tolist() == [[100.0, -20.0]]  # input untouched
    with pytest.raises(ValueError, match="finite"):
        to_tension_positive(np.array([np.nan]))


def test_heatmap_compression_shape_sign_units() -> None:
    series = _series()
    data = load_heatmap(series, "compression")
    assert isinstance(data, HeatmapData)
    assert data.values.shape == (4, 3)
    assert data.levels == LEVELS
    assert data.units == "N"
    assert data.sign_convention == "compression-positive"
    assert data.observable == "joint_reaction"
    assert data.distribution_source == SOURCE
    assert data.scientific_status == "unqualified"
    assert data.values[:, 0] == pytest.approx([100.0, 200.0, 300.0, 400.0])
    assert not data.values.flags.writeable


def test_heatmap_tension_matches_axial_provider_boundary() -> None:
    series = _series()
    data = load_heatmap(series, "tension")
    assert data.sign_convention == "tension-positive"
    assert data.values == pytest.approx(-series.compression_n())
    frame = SpinalAxialLoadProvider(
        series, float(series.time_s[2])
    ).get_segment_axial_loads()
    assert isinstance(frame, AxialLoadFrame)
    for k, level in enumerate(LEVELS):
        assert data.values[2, k] == pytest.approx(frame.values_n[level.name])


def test_heatmap_shears() -> None:
    series = _series()
    assert np.all(load_heatmap(series, "ap_shear").values == -5.0)
    assert np.all(load_heatmap(series, "ml_shear").values == 7.0)


def test_heatmap_body_weight_units() -> None:
    series = _series().in_body_weights(80.0)
    assert load_heatmap(series, "compression").units == "BW"


def test_heatmap_status_never_promoted() -> None:
    series = _series()
    assert load_heatmap(series, "compression").scientific_status == "unqualified"
    assert load_heatmap(series, "compression").status_label() == "UNQUALIFIED"
    qualified = load_heatmap(series, "compression", scientific_status="qualified")
    assert qualified.status_label() == "QUALIFIED"
    with pytest.raises(ValueError, match="scientific_status"):
        load_heatmap(series, "compression", scientific_status="ok")


def test_heatmap_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError, match="SpinalLoadSeries"):
        load_heatmap("series", "compression")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantity"):
        load_heatmap(_series(), "torque")


def test_module_is_pure_data() -> None:
    tree = ast.parse(Path(viz_data.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported & {"matplotlib", "PyQt6", "PySide6", "opensim"}
