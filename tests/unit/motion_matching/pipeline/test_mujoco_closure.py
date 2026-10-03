"""Native grip closure is available without artificial marker attachments."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import plant

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.mark.parametrize("elbow", [0.0, -0.6])
def test_native_closure_without_observation_markers(elbow: float) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    native = plant.get_plant("mujoco", SPEC.read_bytes())
    q = np.zeros(len(native.coordinate_order))
    q[native.coordinate_order.index("REInput")] = elbow
    xml, _ = export_full_body_mjcf(SPEC.read_bytes())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name, value in zip(native.coordinate_order, q, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    expected = (
        data.site_xpos[model.site("native_closure_a").id]
        - data.site_xpos[model.site("native_closure_b").id]
    )
    actual = native.closure_residuals(q)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.shape == (3,)
    assert np.isfinite(actual).all()
    actual[:] = 999.0
    np.testing.assert_allclose(native.closure_residuals(q), expected, atol=1e-12)


@pytest.mark.parametrize("invalid", ["wrong_size", "nonfinite", "matrix"])
def test_native_closure_rejects_invalid_coordinate_vectors(invalid: str) -> None:
    native = plant.get_plant("mujoco", SPEC.read_bytes())
    size = len(native.coordinate_order)
    q = {
        "wrong_size": np.zeros(1),
        "nonfinite": np.full(size, np.nan),
        "matrix": np.zeros((size, 1)),
    }[invalid]
    with pytest.raises(ValueError, match="finite vector of model size"):
        native.closure_residuals(q)
