#!/usr/bin/env python3
"""Export video force/torque overlay evidence for web UI (FTO-29, #11314).

Generates:
1. Standalone HTML visualizer: docs/development/evidence/video_force_overlay_evidence.html
2. Headless screenshot: docs/development/evidence/video_force_overlay_evidence.png
"""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
import sys

import numpy as np

_repo_root = Path(__file__).resolve().parents[2]
for _p in (_repo_root / "src" / "shared" / "python", _repo_root / "src", _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.motion_capture.reconstruct.cameras import (  # noqa: E402
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.shared.python.force_overlay import (  # noqa: E402
    ForceGlyphStyle,
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
    build_glyphs,
)
from src.shared.python.force_overlay.projection import (  # noqa: E402
    ProjectedGlyphSet,
    project_glyphs,
)
from src.shared.python.force_overlay.renderers.opencv_glyphs import (  # noqa: E402
    PinholeProjector,
)


def build_evidence_camera() -> PinholeCamera:
    """Build a 1920x1080 synthetic front camera calibrated for golf swing."""
    w, h = 1920, 1080
    matrix = intrinsics_from_fov(w, h, 60.0)
    pos = np.array([0.0, -3.5, 1.0])
    tgt = np.array([0.0, 0.0, 0.8])
    rot = look_at(pos, tgt, up=np.array([0.0, 0.0, 1.0]))
    return PinholeCamera(
        camera_id="cam_synthetic_front",
        matrix=matrix,
        rotation_world_from_camera=rot,
        translation_world_from_camera_m=pos,
        image_size_px=(w, h),
    )


def build_evidence_frame() -> ForceTorqueFrame:
    """Build synthetic golf club force frame at impact."""
    wrist_rx = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="reaction:wrist",
        body="shaft",
        point_m=(0.0, 0.0, 1.0),
        force_n=(30.0, 0.0, 160.0),
        source="engine",
    )
    wrist_tau = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:wrist",
        body="shaft",
        point_m=(0.0, 0.0, 1.0),
        torque_nm=(0.0, 16.0, 0.0),
        source="engine",
    )
    clubhead_contact = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:clubhead",
        body="clubhead",
        point_m=(0.3, 0.0, 0.2),
        force_n=(280.0, 0.0, 120.0),
        source="engine",
    )
    return ForceTorqueFrame(
        time_s=0.60,
        wrenches=(wrist_rx, wrist_tau, clubhead_contact),
        engine="pinocchio",
    )


def render_svg_elements(glyph_dict: dict) -> str:
    """Render projected glyphs dictionary into SVG elements matching React VideoForceOverlay."""
    elements: list[str] = []
    arrows = glyph_dict.get("arrows", [])
    arcs = glyph_dict.get("torque_arcs", [])

    # 1. Halo polylines first (dark halo underneath)
    for arrow in arrows:
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in arrow["polyline_px"])
        halo_w = arrow.get("halo_width_px", 4.0)
        elements.append(
            f'  <polyline points="{pts}" stroke="#000000" stroke-opacity="0.85" '
            f'stroke-width="{halo_w:.1f}" stroke-linecap="round" vector-effect="non-scaling-stroke" />'
        )
    for arc in arcs:
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in arc["polyline_px"])
        halo_w = arc.get("halo_width_px", 4.0)
        elements.append(
            f'  <polyline points="{pts}" stroke="#000000" stroke-opacity="0.85" '
            f'stroke-width="{halo_w:.1f}" stroke-linecap="round" fill="none" '
            f'vector-effect="non-scaling-stroke" />'
        )

    # 2. Main colored polylines
    for arrow in arrows:
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in arrow["polyline_px"])
        shaft_w = arrow.get("shaft_width_px", 2.0)
        color = arrow.get("color_hex", "#ffffff")
        elements.append(
            f'  <polyline points="{pts}" stroke="{color}" stroke-width="{shaft_w:.1f}" '
            f'stroke-linecap="round" vector-effect="non-scaling-stroke" />'
        )
    for arc in arcs:
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in arc["polyline_px"])
        shaft_w = arc.get("shaft_width_px", 2.0)
        color = arc.get("color_hex", "#ffffff")
        elements.append(
            f'  <polyline points="{pts}" stroke="{color}" stroke-width="{shaft_w:.1f}" '
            f'stroke-linecap="round" fill="none" vector-effect="non-scaling-stroke" />'
        )

    # 3. Halo arrowheads then colored arrowheads
    for item in arrows + arcs:
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in item["head_poly_px"])
        elements.append(
            f'  <polygon points="{pts}" fill="#000000" fill-opacity="0.85" '
            f'stroke="#000000" stroke-width="2" />'
        )
    for item in arrows + arcs:
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in item["head_poly_px"])
        color = item.get("color_hex", "#ffffff")
        elements.append(f'  <polygon points="{pts}" fill="{color}" />')

    return "\n".join(elements)


def _build_legend_html(glyph_dict: dict) -> str:
    """Build HTML legend items from projected glyphs."""
    legend_rows = []
    for arrow in glyph_dict.get("arrows", []):
        legend_rows.append(
            f'<div class="legend-row">'
            f'<div class="swatch" style="background-color: {arrow.get("color_hex")};"></div>'
            f"<span>{html.escape(arrow.get('label', ''))}: "
            f"<b>{arrow.get('magnitude', 0.0):.1f} {arrow.get('units', 'N')}</b></span>"
            f"</div>"
        )
    for arc in glyph_dict.get("torque_arcs", []):
        legend_rows.append(
            f'<div class="legend-row">'
            f'<div class="swatch" style="background-color: {arc.get("color_hex")};"></div>'
            f"<span>{html.escape(arc.get('label', ''))}: "
            f"<b>{arc.get('magnitude', 0.0):.1f} {arc.get('units', 'N·m')}</b></span>"
            f"</div>"
        )
    return "".join(legend_rows)


def _get_page_css() -> str:
    """Return styling CSS for evidence page."""
    return """
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; background: #0b0f17; color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .header { margin-bottom: 20px; }
    h1 { font-size: 20px; margin: 0 0 6px 0; color: #60a5fa; font-weight: 600; }
    .subtitle { font-size: 13px; color: #9ca3af; }
    .main-container { display: flex; gap: 24px; align-items: flex-start; }
    .video-viewport {
      position: relative; width: 840px; height: 472.5px;
      background: #111827; border-radius: 8px; overflow: hidden;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
      border: 1px solid #374151;
    }
    .simulated-frame {
      position: absolute; inset: 0;
      background: radial-gradient(circle at 50% 60%, #1e293b 0%, #0f172a 100%);
    }
    .simulated-club {
      position: absolute; inset: 0; pointer-events: none;
    }
    .overlay-svg {
      position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none;
    }
    .sidebar {
      flex: 1; display: flex; flex-direction: column; gap: 16px;
    }
    .panel {
      background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 16px;
    }
    .panel h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #9ca3af; margin: 0 0 12px 0; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
    .badge {
      display: inline-flex; align-items: center; padding: 4px 10px;
      border-radius: 9999px; font-size: 12px; font-weight: 500;
      background: #1e3a5f; color: #93c5fd; border: 1px solid #2563eb;
    }
    .control-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 13px; }
    .swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; margin-right: 8px; }
    .legend-row { display: flex; align-items: center; font-size: 13px; margin-bottom: 8px; }
    .receipts { font-size: 12px; color: #9ca3af; line-height: 1.6; }
    .receipts code { color: #e5e7eb; background: #1f2937; padding: 2px 5px; border-radius: 4px; }
    """


def _render_page_html(svg_elements: str, glyph_dict: dict) -> str:
    """Build complete HTML page for evidence preview."""
    receipt = glyph_dict.get("receipt", {})
    receipts_html = "<br>".join(
        f"<code>{html.escape(k)}</code>: {html.escape(str(v))}"
        for k, v in receipt.items()
    )
    legend_html = _build_legend_html(glyph_dict)
    css_content = _get_page_css()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Video Force/Torque Overlay Component (FTO-29, #11314)</title>
  <style>{css_content}</style>
</head>
<body>
  <div class="header">
    <h1>Video Force/Torque Overlay Component</h1>
    <div class="subtitle">FTO-29 (#11314) &bull; SVG Projection Aligned with Video Frame &bull; Shared DRY FTO-8 Projector</div>
  </div>
  <div class="main-container">
    <div class="video-viewport">
      <div class="simulated-frame"></div>
      <svg class="simulated-club" viewBox="0 0 1920 1080">
        <!-- Ground reference line -->
        <line x1="200" y1="920" x2="1720" y2="920" stroke="#334155" stroke-width="2" stroke-dasharray="12,12" />
        <text x="220" y="905" fill="#64748b" font-size="20" font-family="sans-serif">Ground plane (z=0)</text>
        <!-- Golfer / shaft reference silhouette -->
        <line x1="960" y1="450" x2="1100" y2="820" stroke="#475569" stroke-width="6" stroke-linecap="round" />
        <circle cx="1100" cy="821" r="18" fill="#64748b" />
        <circle cx="960" cy="450" r="14" fill="#64748b" />
        <text x="980" y="445" fill="#94a3b8" font-size="18" font-family="sans-serif">Grip / wrist joint</text>
        <text x="1130" y="830" fill="#94a3b8" font-size="18" font-family="sans-serif">Clubhead impact point</text>
      </svg>
      <svg class="overlay-svg" viewBox="0 0 1920 1080">
{svg_elements}
      </svg>
    </div>
    <div class="sidebar">
      <div class="panel">
        <h2>Overlay Controls (VideoAnalyzer.tsx)</h2>
        <div class="toolbar">
          <span class="badge">&check; Forces Active</span>
          <span class="badge">&check; Torques Active</span>
          <span class="badge">25.0 FPS</span>
        </div>
        <div class="control-row">
          <span>Time / Frame:</span>
          <span><b>00:00.60</b> (Frame #15 @ 25 fps)</span>
        </div>
        <div class="control-row">
          <span>SVG viewBox:</span>
          <span><code>0 0 1920 1080</code> (Dynamic metadata)</span>
        </div>
        <div class="control-row">
          <span>Projection Engine:</span>
          <span><code>project_glyphs()</code> (Server FTO-8)</span>
        </div>
      </div>
      <div class="panel">
        <h2>Active Wrenches & Legend</h2>
        {legend_html}
      </div>
      <div class="panel">
        <h2>Server Receipts & Calibration</h2>
        <div class="receipts">
          {receipts_html}
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def generate_video_force_overlay_html(output_html: Path) -> None:
    """Generate HTML evidence visualizer for video force overlay."""
    camera = build_evidence_camera()
    frame = build_evidence_frame()
    style = ForceGlyphStyle(
        force_scale_m_per_n=0.002,
        torque_scale_m_per_nm=0.015,
    )
    glyph_set = build_glyphs(frame, style)
    projector = PinholeProjector(camera)
    projected = project_glyphs(glyph_set, projector, image_size_px=(1920, 1080))
    glyph_dict = projected.to_dict()

    svg_elements = render_svg_elements(glyph_dict)
    html_content = _render_page_html(svg_elements, glyph_dict)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")


async def capture_screenshot(html_path: Path, png_path: Path) -> None:
    """Capture headless screenshot of evidence HTML using Chromium/Edge."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel="msedge")
        except Exception:  # noqa: BLE001
            browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(html_path.as_uri())
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(png_path))
        await browser.close()


def main() -> None:
    """CLI entrypoint."""
    repo_root = Path(__file__).resolve().parents[2]
    evidence_dir = repo_root / "docs" / "development" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    html_path = evidence_dir / "video_force_overlay_evidence.html"
    png_path = evidence_dir / "video_force_overlay_evidence.png"

    generate_video_force_overlay_html(html_path)
    print(f"Generated HTML evidence: {html_path}")  # noqa: T201

    try:
        asyncio.run(capture_screenshot(html_path, png_path))
        print(f"Captured screenshot evidence: {png_path}")  # noqa: T201
    except Exception as exc:  # noqa: BLE001
        print(f"Playwright screenshot capture fallback: {exc}")  # noqa: T201


if __name__ == "__main__":
    main()
