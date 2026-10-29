"""Velocity plumbing in the native-viewer exporter path (CAP-3, #14042).

Ensures SwingInput, overlay_source._forward, and HandMobility carry velocity v
and expose bias h, point Jdot_v, and drift acceleration at interface points,
failing if velocity is dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.core import (
    SwingInput,
    load_swing_input,
)

pytestmark = [pytest.mark.unit]

COORDS = ("A", "B")


def _test_bundle(steps: int = 10) -> InputBundle:
    spec = {"coordinate_order": list(COORDS)}
    q = np.ones((steps + 1, 2)) * 0.1
    v = np.ones((steps + 1, 2)) * 2.0
    return InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=COORDS,
        dt_s=0.001,
        q0=q[0],
        v0=v[0],
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=v,
        reference_engine="mujoco",
    )


def test_swing_input_stores_and_validates_velocity() -> None:
    b = _test_bundle()
    swing = SwingInput(b, b.reference_q, "driver", "Driver", "mujoco", v=b.reference_v)
    assert swing.v is not None
    np.testing.assert_array_equal(swing.v, b.reference_v)

    # Rejects mismatched velocity shape
    bad_shape_v = np.zeros((5, 2))
    with pytest.raises(ValueError, match="shape"):
        SwingInput(b, b.reference_q, "driver", "Driver", "mujoco", v=bad_shape_v)

    # Rejects non-finite velocity
    bad_val_v = b.reference_v.copy()
    bad_val_v[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SwingInput(b, b.reference_q, "driver", "Driver", "mujoco", v=bad_val_v)


def test_load_swing_input_defaults_to_bundle_reference_v(tmp_path: Path) -> None:
    b = _test_bundle()
    bundle_path = tmp_path / "bundle.npz"
    b.save(bundle_path)

    loaded = load_swing_input(bundle_path, swing="driver", club="Driver")
    assert loaded.v is not None
    np.testing.assert_array_equal(loaded.v, b.reference_v)


def test_overlay_source_bias_force_and_point_jdot_v() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )
    from src.shared.python.motion_matching.full_body_spec import (
        load_full_body_spec,
    )

    root = Path(__file__).resolve().parents[4]
    upper = (
        root
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    full = root / "docs/development/full_body_models/full_body_spec_v1.json"
    spec = load_full_body_spec(full, json.loads(upper.read_text(encoding="utf-8")))
    spec_bytes = json.dumps(spec).encode("utf-8")

    src = MujocoOverlaySource(spec_bytes)
    coords = dict.fromkeys(src.coordinate_order, 0.1)
    rates_zero = dict.fromkeys(src.coordinate_order, 0.0)
    rates_fast = dict.fromkeys(src.coordinate_order, 3.0)

    # Zero velocity -> pure gravity bias
    h_zero = src.bias_force(coords, rates_zero)
    # Non-zero velocity -> Coriolis / centrifugal + gravity
    h_fast = src.bias_force(coords, rates_fast)

    # Fails if velocity is dropped
    assert not np.allclose(h_zero, h_fast, atol=1e-3), (
        "Velocity was dropped: h(q, v) identical to h(q, 0)"
    )

    # Jdot_v at origin of lead hand / club body
    club_body_id = (
        src.model.body("club").id
        if "club" in [src.model.body(i).name for i in range(src.model.nbody)]
        else 1
    )
    pt = [0.0, 0.0, 0.0]
    jdot_v_zero = src.point_jdot_v(pt, club_body_id, coords, rates_zero)
    jdot_v_fast = src.point_jdot_v(pt, club_body_id, coords, rates_fast)

    np.testing.assert_allclose(jdot_v_zero, 0.0, atol=1e-12)
    assert np.linalg.norm(jdot_v_fast) > 1e-3, (
        "Velocity was dropped: Jdot_v is zero despite non-zero joint rates"
    )


def test_hand_mobility_source_evaluates_drift_and_jdot_v() -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.hand_mobility import (
        HandMobilitySource,
    )

    root = Path(__file__).resolve().parents[4]
    anthro = (
        root / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    )
    if not anthro.exists():
        pytest.skip("anthro driver spec missing")

    src = HandMobilitySource(anthro.read_bytes())
    q = np.zeros(len(src.coordinate_order))
    v = np.full(len(src.coordinate_order), 2.0)

    # drift acceleration and grip jdot_v
    jdot_v = src.grip_jdot_v("lead", q, v)
    assert jdot_v.shape == (3,)
    assert np.linalg.norm(jdot_v) > 1e-4

    drift = src.drift_acceleration(q, v, "lead", mode="constrained")
    assert drift.shape == (3,)
    assert np.all(np.isfinite(drift))

    # Hand mobility with v exposes drift_acceleration and jdot_v
    hm = src.hand_mobility(q, v=v)
    lead = hm["lead"]
    assert lead.drift_acceleration is not None
    assert lead.jdot_v is not None
    np.testing.assert_allclose(lead.jdot_v, jdot_v)
    np.testing.assert_allclose(lead.drift_acceleration, drift)
