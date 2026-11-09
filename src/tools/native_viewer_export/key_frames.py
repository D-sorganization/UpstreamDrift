"""Key-frame PNG writer and grip-clock inset helper for styled renders (SHOW-A10, #14735).

Extracts and exports key-frame still images (address, top of backswing, impact)
with the owner's face-on grip clock inset:
- 12 o'clock = 0 deg: back of wrist (dorsal axis) on top of the shaft
- Counterclockwise (+) toward the target side (left in face-on)
- Green tick at +15 deg (the owner default from SHOW-A7, #14731)
- Red pointer for lead wrist roll, blue pointer for trail wrist roll.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.shared.python.contracts import ensure, require

WHITE = (255, 255, 255)
GREY_OUTLINE = (200, 200, 200)
GREY_AXIS = (90, 90, 90)
GREEN_TARGET = (60, 220, 60)
COLORS = {"lead": (255, 60, 60), "trail": (80, 120, 255)}
DEFAULT_TARGET_DEG = 15.0


def draw_grip_clock(
    image: Image.Image | np.ndarray,
    rolls: Mapping[str, float],
    cx: int | None = None,
    cy: int | None = None,
    r: int = 70,
    target_deg: float = DEFAULT_TARGET_DEG,
) -> Image.Image:
    """Draw the owner's grip clock inset on an image.

    DbC:
        require: r >= 20; 'lead' and 'trail' in rolls with finite values.
        ensure: returned Image.Image matches input width and height.
    """
    require(r >= 20, "clock radius r must be at least 20 pixels", r)
    require(
        "lead" in rolls and "trail" in rolls,
        "rolls must contain 'lead' and 'trail' keys",
        list(rolls.keys()),
    )
    require(
        math.isfinite(rolls["lead"]) and math.isfinite(rolls["trail"]),
        "roll angles must be finite",
        (rolls["lead"], rolls["trail"]),
    )

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).copy()
    else:
        img = image.copy()

    w, h = img.size
    center_x = cx if cx is not None else w - r - 40
    center_y = cy if cy is not None else h - r - 50

    draw = ImageDraw.Draw(img)

    def at(deg: float, scale: float = 1.0) -> tuple[float, float]:
        rad = math.radians(deg)
        return (
            center_x - scale * r * math.sin(rad),
            center_y - scale * r * math.cos(rad),
        )

    # Dial circle and crosshairs
    draw.ellipse(
        (center_x - r, center_y - r, center_x + r, center_y + r),
        outline=GREY_OUTLINE,
        width=2,
    )
    draw.line((center_x - r, center_y, center_x + r, center_y), fill=GREY_AXIS)
    draw.line((center_x, center_y - r, center_x, center_y + r), fill=GREY_AXIS)

    # Default tick (+15 deg default target)
    draw.line((at(target_deg, 0.8), at(target_deg)), fill=GREEN_TARGET, width=3)

    # Dial labels
    draw.text((center_x - r - 35, center_y - 6), "target", fill=WHITE)
    draw.text((center_x - 16, center_y - r - 12), "12 = 0°", fill=WHITE)

    # Needles: lead (red, width 5), trail (blue, width 3)
    lead_deg = float(rolls["lead"])
    trail_deg = float(rolls["trail"])
    draw.line((center_x, center_y, *at(lead_deg, 0.9)), fill=COLORS["lead"], width=5)
    draw.line((center_x, center_y, *at(trail_deg, 0.85)), fill=COLORS["trail"], width=3)

    # Legend text beneath clock
    lead_str = f"{lead_deg:+.1f}°"
    trail_str = f"{trail_deg:+.1f}°"
    legend = f"grip: lead {lead_str}, trail {trail_str}"
    draw.text((center_x - r, center_y + r + 6), legend, fill=WHITE)
    draw.text(
        (center_x - r, center_y + r + 18),
        f"green tick = {target_deg:g}° default",
        fill=WHITE,
    )

    ensure(img.size == (w, h), "output image size must match input size")
    return img


def detect_key_frame_indices(
    times: Sequence[float] | np.ndarray,
    head_positions: np.ndarray | None = None,
    impact_time_s: float | None = None,
) -> dict[str, int]:
    """Detect address, top of backswing, and impact indices in a swing trajectory.

    DbC:
        require: len(times) >= 2.
        ensure: 0 <= address < top < impact < len(times).
    """
    t = np.asarray(times, dtype=np.float64)
    n = len(t)
    require(n >= 2, "times array must contain at least 2 samples", n)

    address_idx = 0

    if impact_time_s is not None and np.isfinite(impact_time_s):
        impact_idx = int(np.argmin(np.abs(t - impact_time_s)))
    elif head_positions is not None:
        from src.shared.python.model_appearance.club_face import ball_passage

        _, impact_idx, _ = ball_passage(t, head_positions)
        impact_idx = int(impact_idx)
    else:
        impact_idx = n - 1

    # Clamp impact_idx to at least index 2
    impact_idx = max(2, min(n - 1, impact_idx))

    if head_positions is not None and head_positions.shape[0] == n:
        from src.shared.python.model_appearance.club_face import top_of_backswing_index

        try:
            top_idx = int(top_of_backswing_index(t, head_positions))
        except (ValueError, RuntimeError):
            top_idx = max(1, int(round(0.6 * impact_idx)))
    else:
        top_idx = max(1, int(round(0.6 * impact_idx)))

    # Ensure strictly monotonic address < top < impact
    if top_idx >= impact_idx:
        top_idx = max(1, impact_idx - 1)
    if top_idx <= address_idx:
        top_idx = 1
        if impact_idx <= top_idx:
            impact_idx = min(n - 1, top_idx + 1)

    result = {
        "address": address_idx,
        "top": top_idx,
        "impact": impact_idx,
    }
    ensure(
        0 <= result["address"] < result["top"] < result["impact"] < n,
        "key frame indices must be strictly ordered",
        result,
    )
    return result


def compose_side_by_side(
    baseline: Image.Image | np.ndarray,
    candidate: Image.Image | np.ndarray,
    label_left: str = "Baseline (11-04)",
    label_right: str = "A10 Improved Best-Match",
) -> Image.Image:
    """Stitch baseline and candidate images side-by-side with header text."""
    b_img = Image.fromarray(baseline) if isinstance(baseline, np.ndarray) else baseline
    c_img = (
        Image.fromarray(candidate) if isinstance(candidate, np.ndarray) else candidate
    )

    w1, h1 = b_img.size
    w2, h2 = c_img.size
    height = max(h1, h2)
    width = w1 + w2

    canvas = Image.new("RGB", (width, height), (10, 10, 10))
    canvas.paste(b_img, (0, 0))
    canvas.paste(c_img, (w1, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((16, 16), label_left, fill=WHITE)
    draw.text((w1 + 16, 16), label_right, fill=WHITE)
    return canvas


def write_key_frame_stills(
    frames: Mapping[str, np.ndarray | Image.Image],
    rolls: Mapping[str, float],
    out_dir: Path,
    prefix: str = "",
    r: int = 70,
) -> dict[str, Path]:
    """Render key-frame still PNGs with grip clock inset.

    Returns mapping of ``{phase: path}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for phase, frame in frames.items():
        with_clock = draw_grip_clock(frame, rolls, r=r)
        dest = out_dir / f"{prefix}{phase}.png"
        with_clock.save(dest, format="PNG")
        paths[phase] = dest

    return paths


def render_a10_summary(
    model_id: str,
    source_commit: str,
    receipt_verdict: str,
    input_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    lead_roll_deg: float = 15.0,
    trail_roll_deg: float = -165.0,
    *,
    receipt_model_id: str | None = None,
    source_still_hashes: Mapping[str, str] | None = None,
) -> str:
    """Render SUMMARY.md for SHOW-A10 evidence deliverable."""
    id_line = f" (candidate receipt `{receipt_model_id}`)" if receipt_model_id else ""
    lines = [
        f"# Best-Match Styled Render: `{model_id}` (SHOW-A10)",
        "",
        f"- Model id: `{model_id}`{id_line}",
        f"- Source commit: `{source_commit}`",
        f"- Receipt verdict: {receipt_verdict}",
        f"- Clock overlay angles (illustrative target, not measured from rendered frames): lead {lead_roll_deg:+.1f}°, trail {trail_roll_deg:+.2f}° (SHOW-A7 15° CCW convention)",
        "- Rendered-frame hand roll measurement: unavailable (SHOW-A8 stills with fixed clock overlay)",
        "- Trunk: 6-DOF lumbar + thoracic segments (SHOW-A8, `full_body_spec_anthro_driver_v2.json`)",
        "- Shading: axial tension/compression segment shading enabled",
        "- Force vectors & torque arcs: enabled",
        "",
        "## Outputs (SHA-256)",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for name, digest in sorted(output_hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")

    lines.extend(
        [
            "",
            "## Inputs (SHA-256)",
            "",
            "| Input | SHA-256 |",
            "|---|---|",
        ]
    )
    for name, digest in sorted(input_hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")

    if source_still_hashes:
        lines.extend(
            [
                "",
                "## Source Stills Consumed (SHA-256)",
                "",
                "| Source Still | SHA-256 |",
                "|---|---|",
            ]
        )
        for name, digest in sorted(source_still_hashes.items()):
            lines.append(f"| `{name}` | `{digest}` |")

    lines.append("")
    return "\n".join(lines)
