"""Unit tests for the swing wrist and forearm readout (Issue #15275)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import plot_swing_wrist_readout as plot_mod
from src.shared.python.anatomical_angles import swing_wrist_readout as swr
from src.shared.python.motion_matching.arm_mirror import mirror_document_arm

pytestmark = pytest.mark.unit

STEP_DEG = 10.0
CROSS_TALK_LIMIT_DEG = 0.5
ROOT = Path(__file__).resolve().parents[3]
SPEC_REL = "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
_SERIES = ("flexion_deg", "ulnar_deg", "pronation_deg")


def _spec(*, mirrored_right: bool = True) -> dict[str, Any]:
    doc = json.loads((ROOT / SPEC_REL).read_text(encoding="utf-8"))
    if mirrored_right:
        mirror_document_arm(doc)
    return doc


def _q(spec: dict[str, Any], frames: list[dict[str, float]]) -> np.ndarray:
    order = spec["coordinate_order"]
    out = np.zeros((len(frames), len(order)), dtype=float)
    for row_idx, frame in enumerate(frames):
        for name, deg in frame.items():
            out[row_idx, order.index(name)] = np.deg2rad(deg)
    return out


@pytest.mark.parametrize(
    ("role", "side", "coordinate", "target_attr"),
    [
        ("lead", "L", "LWInputY", "flexion_deg"),
        ("lead", "L", "LWInputX", "ulnar_deg"),
        ("lead", "L", "LFInput", "pronation_deg"),
        ("trail", "R", "RWInputY", "flexion_deg"),
        ("trail", "R", "RWInputX", "ulnar_deg"),
        ("trail", "R", "RFInput", "pronation_deg"),
    ],
)
def test_single_coordinate_step_changes_target_by_10_deg_with_low_crosstalk(
    role: str, side: str, coordinate: str, target_attr: str
) -> None:
    spec = _spec(mirrored_right=True)
    readout = swr.swing_wrist_readout(spec, _q(spec, [{}, {coordinate: STEP_DEG}]))
    arm = getattr(readout, role)
    assert arm.side == side
    delta = getattr(arm, target_attr)[1] - getattr(arm, target_attr)[0]
    assert delta == pytest.approx(-STEP_DEG, abs=1e-6)
    for other in set(_SERIES) - {target_attr}:
        cross = abs(float(getattr(arm, other)[1] - getattr(arm, other)[0]))
        assert cross < CROSS_TALK_LIMIT_DEG


def test_mirrored_motion_lead_side_mapping_and_plot(tmp_path: Path) -> None:
    spec = _spec(mirrored_right=True)
    pose0 = {"LWInputY": 10.0, "RWInputY": 10.0}
    pose1 = {"LWInputY": -15.0, "LWInputX": -12.0, "LFInput": -20.0}
    pose1.update({"RWInputY": -15.0, "RWInputX": -12.0, "RFInput": -20.0})
    cfg = swr.SwingWristReadoutSettings(lead_side="L", dt_s=0.01)
    readout = swr.swing_wrist_readout(spec, _q(spec, [pose0, pose1]), settings=cfg)
    assert (readout.lead.side, readout.trail.side) == ("L", "R")
    assert readout.lead.flexion_deg[1] > 0.0 and readout.lead.ulnar_deg[1] > 0.0
    assert readout.lead.pronation_deg[1] == pytest.approx(20.0, abs=1e-6)
    for attr in _SERIES:
        assert np.allclose(getattr(readout.lead, attr), getattr(readout.trail, attr))
    expected_rel = readout.lead.flexion_deg[1] - readout.lead.flexion_deg[0]
    assert readout.lead.flexion_rel_address_deg[1] == pytest.approx(expected_rel)
    assert readout.time_s is not None and np.allclose(readout.time_s, [0.0, 0.01])

    unmirrored = _spec(mirrored_right=False)
    q_unmir = _q(unmirrored, [{"LWInputY": -10.0, "RWInputY": 15.0}])
    cfg_r = swr.SwingWristReadoutSettings(lead_side="R")
    swapped = swr.swing_wrist_readout(unmirrored, q_unmir, settings=cfg_r)
    assert (swapped.lead_side, swapped.trail_side) == ("R", "L")
    assert swapped.lead.flexion_deg[0] == pytest.approx(15.0, abs=1e-6)
    assert swapped.trail.flexion_deg[0] == pytest.approx(10.0, abs=1e-6)
    assert swapped.observability == swr.MODEL_JOINT_OBSERVABILITY
    decoded = json.loads(json.dumps(swapped.as_mapping(), allow_nan=False))
    assert (decoded["lead"]["side"], decoded["trail"]["side"]) == ("R", "L")

    q_plot = _q(spec, [{"LWInputY": 5.0}, {"LWInputY": -20.0}, {"LWInputY": 10.0}])
    bundle_path, out_png = tmp_path / "bundle.npz", tmp_path / "wrist.png"
    spec_bytes = np.frombuffer(json.dumps(spec).encode("utf-8"), dtype=np.uint8)
    np.savez(bundle_path, spec=spec_bytes, reference_q=q_plot)
    loaded = plot_mod.load_bundle_readout(bundle_path)
    assert plot_mod.plot_swing_wrist_readout(loaded, out_png).is_file()
    mid_val = plot_mod.summarize_readout(loaded)["lead_L"]["flexion_deg"]["mid_deg"]
    assert mid_val == pytest.approx(20.0)


def test_bad_inputs_are_refused() -> None:
    spec = _spec()
    n_coords = len(spec["coordinate_order"])
    valid_q = np.zeros((2, n_coords))
    with pytest.raises(TypeError, match="spec"):
        swr.swing_wrist_readout(None, valid_q)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coordinate_order"):
        swr.swing_wrist_readout({}, valid_q)
    bad_nan = valid_q.copy()
    bad_nan[0, 0] = np.nan
    for bad_q, match in (
        (np.zeros(n_coords), "2D"),
        (np.zeros((0, n_coords)), "at least one frame"),
        (np.zeros((2, 3)), "column"),
        (bad_nan, "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            swr.swing_wrist_readout(spec, bad_q)
    for bad_cfg, match in (
        (swr.SwingWristReadoutSettings(lead_side="X"), "side"),
        (swr.SwingWristReadoutSettings(dt_s=-0.01), "dt_s"),
        (swr.SwingWristReadoutSettings(address_frame=5), "address_frame"),
    ):
        with pytest.raises(ValueError, match=match):
            swr.swing_wrist_readout(spec, valid_q, settings=bad_cfg)
