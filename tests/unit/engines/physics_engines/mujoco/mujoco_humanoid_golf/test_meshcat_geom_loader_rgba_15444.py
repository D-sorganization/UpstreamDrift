"""Regression for #15444: MuJoCo geom_rgba ndarrays must load in MeshCat bindings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
    meshcat_geom_loader as loader,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.meshcat_adapter import (
    MuJoCoMeshcatAdapter,
)
from src.shared.python.plot_style.color_utils import rgba_to_hex

pytestmark = pytest.mark.unit


def test_normalize_geom_rgba_accepts_ndarray() -> None:
    rgba = np.array([0.2, 0.4, 0.6, 1.0], dtype=np.float64)
    out = loader.normalize_geom_rgba(rgba)
    assert out == (0.2, 0.4, 0.6, 1.0)


def test_normalize_geom_rgba_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        loader.normalize_geom_rgba((0.1, 0.2, float("nan"), 1.0))


def test_rgba_to_int_accepts_model_geom_rgba_slice() -> None:
    adapter = MuJoCoMeshcatAdapter.__new__(MuJoCoMeshcatAdapter)
    rgba = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    value = adapter._rgba_to_int(rgba)
    assert value == int(
        rgba_to_hex(tuple(float(x) for x in rgba), include_alpha=False)[1:], 16
    )


def test_build_meshcat_color_bindings_accepts_ndarray_rgba() -> None:
    rgba_row = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float64)
    model = SimpleNamespace(
        ngeom=1,
        geom_type=np.array([2], dtype=np.int32),
        geom_size=np.array([[0.05, 0.0, 0.0]], dtype=np.float64),
        geom_rgba=rgba_row.reshape(1, 4),
        geom_rbound=np.array([0.05], dtype=np.float64),
        geom_bodyid=np.array([0], dtype=np.int32),
    )

    adapter = MuJoCoMeshcatAdapter.__new__(MuJoCoMeshcatAdapter)
    calls: list[object] = []

    def rgba_to_int(rgba: object) -> int:
        calls.append(rgba)
        return adapter._rgba_to_int(rgba)

    fake_g = SimpleNamespace(
        MeshPhongMaterial=lambda **kw: SimpleNamespace(**kw),
        Sphere=lambda **kw: "sphere",
        Box=lambda lengths: "box",
        Cylinder=lambda **kw: "cylinder",
    )
    fake_tf = SimpleNamespace(rotation_matrix=lambda _a, _v: np.eye(4))

    vis = MagicMock()
    vis.__getitem__ = MagicMock(return_value=vis)

    original_g, original_tf = loader.g, loader.tf
    loader.g = fake_g  # type: ignore[assignment]
    loader.tf = fake_tf  # type: ignore[assignment]
    original_name = mujoco.mj_id2name
    mujoco.mj_id2name = lambda _m, _o, _i, default="": "torso"  # type: ignore[method-assign,assignment]
    try:
        bindings = loader.build_meshcat_color_bindings(
            vis, model, rgba_to_int=rgba_to_int
        )
    finally:
        loader.g = original_g
        loader.tf = original_tf
        mujoco.mj_id2name = original_name  # type: ignore[method-assign]

    assert bindings.get("torso")
    assert len(calls) == 1
