"""MuJoCo scripted swing grip-margin receipt (#13885 PR 3).

``python3 -m scripts.grip.mujoco_grip_margin_swing_receipt``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.grip_margin_swing import (
    RECEIPT_DURATION_S,
    SwingReceiptConfig,
    grip_margin_from_contact_run,
    load_mu_bias_bound,
    receipt_payload,
    serialise_report,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "docs/development/full_body_models"
EVIDENCE = MODELS / "evidence/grip_kinetics"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
CALIBRATION = EVIDENCE / "mujoco_mu_crit_calibration.json"


def run_receipt(config: SwingReceiptConfig | None = None) -> dict:
    """Execute the MuJoCo pad-contact swing and return the receipt dict."""
    import mujoco  # noqa: F401, PLC0415

    from src.shared.python.grip_contact import (  # noqa: PLC0415
        GripInterface,
        build_topology_model,
        contact_topology,
        load_coordinate_swing,
    )
    from src.engines.physics_engines.mujoco.python.grip_contact_sim import (  # noqa: PLC0415
        simulate_grip_contact,
    )
    from src.shared.python.motion_matching.grip_contract import GripSpec  # noqa: PLC0415

    cfg = config or SwingReceiptConfig()
    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    style = GripStyle.OVERLAP
    interface = GripInterface.from_spec(spec)
    topo = contact_topology(GripSpec(style=style, trigger_finger=False))
    topo_model = build_topology_model(
        interface, topo, squeeze_per_hand_n=cfg.squeeze_per_hand_n
    )
    friction = float(interface.contact_material.static_friction)
    run = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=cfg.duration_s,
        hold=False,
        topology_model=topo_model,
    )
    bias = load_mu_bias_bound(CALIBRATION)
    report = grip_margin_from_contact_run(
        run, topo_model, spec, friction, config=cfg, mu_bias_bound=bias
    )
    payload = receipt_payload(
        report,
        config=cfg,
        style=style,
        friction=friction,
        calibration_path=CALIBRATION,
    )
    payload["report"] = serialise_report(report)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--duration-s", type=float, default=RECEIPT_DURATION_S)
    args = ap.parse_args(argv)
    cfg = SwingReceiptConfig(duration_s=args.duration_s)
    payload = run_receipt(cfg)
    out = args.out or EVIDENCE / "mujoco_grip_margin_swing_receipt.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
