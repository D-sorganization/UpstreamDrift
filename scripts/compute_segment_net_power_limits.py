"""Derive default symmetric net-power limits from swing_q_driver.npz (#15399)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (  # noqa: E402
    export_full_body_mjcf,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.interface_power import (  # noqa: E402
    MujocoInterfacePowerAnalyzer,
)
from src.shared.python.body_part_viz.segment_net_power_limits import (  # noqa: E402
    DEFAULT_EXCLUDE_FRAME_RANGES,
    DEFAULT_IMPACT_CORE_FRAMES,
    DEFAULT_IMPACT_MARGIN_FRAMES,
    derive_symmetric_limit_w,
    trajectory_mechanical_rates,
    write_limits_json,
)
from src.shared.python.grip_contact import load_coordinate_swing  # noqa: E402


def main() -> int:
    import mujoco

    spec_path = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        ROOT / "tests/fixtures/club_face/swing_q_driver.npz",
        ROOT / "tests/fixtures/club_face/address_poses.json",
        "driver",
        names,
    )
    xml, _ = export_full_body_mjcf(spec_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    analyzer = MujocoInterfacePowerAnalyzer(model)
    sample_hz = 1.0 / float(np.median(np.diff(swing.time_s)))

    def rates_at(q, qvel, qacc):
        frame = analyzer.frame_at(q, qvel, qacc)
        return [seg.mechanical_energy_rate_w for seg in frame.segments]

    samples, used_frames = trajectory_mechanical_rates(
        sample_hz,
        swing.q,
        rates_at,
        exclude_ranges=DEFAULT_EXCLUDE_FRAME_RANGES,
    )
    derivation = derive_symmetric_limit_w(
        samples,
        source="tests/fixtures/club_face/swing_q_driver.npz",
        sample_count=len(samples),
    )
    out = ROOT / "scripts/config/segment_net_power_limits.json"
    write_limits_json(
        out,
        derivation,
        fixture="tests/fixtures/club_face/swing_q_driver.npz",
        exclude_frame_ranges=list(DEFAULT_EXCLUDE_FRAME_RANGES),
        impact_core_frames=list(DEFAULT_IMPACT_CORE_FRAMES),
        impact_margin_frames=DEFAULT_IMPACT_MARGIN_FRAMES,
        impact_note=(
            "Frames 661–664 are the recorded ball-impact window "
            "(t≈1.322–1.328 s); ±margin frames are dropped before filtering."
        ),
        frames_used=used_frames,
        filter_hz=12.0,
        trim_frames=15,
    )
    print(f"Wrote {out}: ±{derivation.symmetric_limit_w:.2f} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())