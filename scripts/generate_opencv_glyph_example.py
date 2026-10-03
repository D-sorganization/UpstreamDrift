import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "vendor" / "ud-tools" / "src"))

import cv2
import numpy as np

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
)
from src.shared.python.force_overlay.palette import FORCE_KIND_PALETTE, hex_to_rgba
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    PinholeProjector,
    VideoGlyphStyle,
    draw_glyphs_on_frame,
)


def generate_example_png() -> Path:
    w, h = 1920, 1080
    # Gradient background from dark slate (30, 35, 45) to dark navy (15, 20, 30)
    y_coords = np.linspace(0.0, 1.0, h)[:, None, None]
    top_color = np.array([45, 35, 30], dtype=float)  # BGR
    bottom_color = np.array([25, 20, 15], dtype=float)
    background = (y_coords * bottom_color + (1.0 - y_coords) * top_color).astype(
        np.uint8
    )
    background = np.broadcast_to(background, (h, w, 3)).copy()

    k = intrinsics_from_fov(w, h, 55.0)
    pos = np.array([0.0, 1.0, 4.0])
    tgt = np.array([0.0, 0.2, 0.0])
    r = look_at(pos, tgt, up=np.array([0.0, 1.0, 0.0]))
    cam = PinholeCamera(
        camera_id="cam_example",
        matrix=k,
        rotation_world_from_camera=r,
        translation_world_from_camera_m=pos,
        image_size_px=(w, h),
    )
    projector = PinholeProjector(cam)

    arrows = [
        ArrowGlyph(
            label="joint_reaction:hip",
            kind="joint_reaction",
            tail_m=(-0.4, 0.2, 0.0),
            tip_m=(-0.4, 0.8, 0.0),
            head_base_m=(-0.4, 0.7, 0.0),
            shaft_radius_m=0.008,
            head_radius_m=0.018,
            rgba=hex_to_rgba(FORCE_KIND_PALETTE["joint_reaction"]),
            magnitude=240.0,
            units="N",
            clamped=False,
        ),
        ArrowGlyph(
            label="actuator:knee",
            kind="joint_actuator",
            tail_m=(0.0, 0.0, 0.0),
            tip_m=(0.5, 0.3, 0.0),
            head_base_m=(0.4, 0.24, 0.0),
            shaft_radius_m=0.008,
            head_radius_m=0.018,
            rgba=hex_to_rgba(FORCE_KIND_PALETTE["joint_actuator"]),
            magnitude=120.0,
            units="N",
            clamped=False,
        ),
        ArrowGlyph(
            label="contact:ground",
            kind="contact",
            tail_m=(0.4, -0.4, 0.0),
            tip_m=(0.4, 0.1, 0.0),
            head_base_m=(0.4, 0.0, 0.0),
            shaft_radius_m=0.008,
            head_radius_m=0.018,
            rgba=hex_to_rgba(FORCE_KIND_PALETTE["contact"]),
            magnitude=450.0,
            units="N",
            clamped=False,
        ),
    ]

    thetas = np.linspace(0, 1.5 * np.pi, 24)
    radius = 0.22
    center = np.array([0.0, 0.0, 0.0])
    poly = tuple(
        (
            float(center[0] + radius * np.cos(th)),
            float(center[1] + radius * np.sin(th)),
            float(center[2]),
        )
        for th in thetas
    )
    arc_tip = (
        float(center[0] + radius * np.cos(thetas[-1] + 0.12)),
        float(center[1] + radius * np.sin(thetas[-1] + 0.12)),
        float(center[2]),
    )
    arc = TorqueArcGlyph(
        label="torque:knee",
        kind="joint_reaction",
        center_m=tuple(center),
        axis_unit=(0.0, 0.0, 1.0),
        radius_m=radius,
        polyline_m=poly,
        head_tip_m=arc_tip,
        head_base_m=poly[-1],
        rgba=(0.0, 0.9, 0.9, 1.0),
        magnitude=45.0,
        units="N·m",
        clamped=False,
    )

    legend = LegendSpec(
        force_reference_n=200.0,
        force_reference_length_m=0.3,
        torque_reference_nm=50.0,
        torque_reference_radius_m=0.22,
        kinds_present=("joint_reaction", "joint_actuator", "contact"),
        unavailable_labels=("ligament_force",),
        engine="pinocchio",
        source_labels=("rnea", "contacts"),
    )
    glyphs = GlyphSet(
        time_s=1.25, arrows=tuple(arrows), torque_arcs=(arc,), legend=legend
    )

    out_frame, receipt = draw_glyphs_on_frame(
        background,
        glyphs,
        projector,
        qualification="synthetic calibrated verification",
    )

    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fto_8_opencv_glyph_example.png"
    cv2.imwrite(str(out_path), out_frame)
    return out_path


if __name__ == "__main__":
    path = generate_example_png()
    print(f"Generated example PNG at: {path}")
