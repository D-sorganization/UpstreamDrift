import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
    add_glyphs_to_scene,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
    MujocoForceTorqueSource,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs


def main() -> None:
    xml_path = Path("src/engines/physics_engines/mujoco/models/generated/golfer.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Step physics a bit so forces/contacts develop
    for _ in range(50):
        mujoco.mj_step(model, data)

    out_dir = Path("docs/development/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, 640, 480)

    # 1. Clean scene (before)
    renderer.update_scene(data)
    img_before = renderer.render()
    Image.fromarray(img_before).save(out_dir / "mujoco_golf_humanoid_before_fto10.png")
    print(f"Saved {out_dir / 'mujoco_golf_humanoid_before_fto10.png'}")

    # 2. Scene with 3D force, torque and contact glyphs (after)
    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)
    style = ForceGlyphStyle(
        force_scale_m_per_n=0.002,
        torque_scale_m_per_nm=0.01,
    )
    glyphs = build_glyphs(frame, style)
    renderer.update_scene(data)
    receipt = add_glyphs_to_scene(renderer.scene, glyphs)
    print(f"Added {receipt.added} geoms, dropped {receipt.dropped}")
    img_after = renderer.render()
    Image.fromarray(img_after).save(out_dir / "mujoco_golf_humanoid_after_fto10.png")
    print(f"Saved {out_dir / 'mujoco_golf_humanoid_after_fto10.png'}")


if __name__ == "__main__":
    main()
