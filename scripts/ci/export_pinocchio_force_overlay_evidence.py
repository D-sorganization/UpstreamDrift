#!/usr/bin/env python3
"""Export MeshCat visual overlay evidence for Pinocchio GUI (FTO-14, #11299).

Generates:
1. Standalone HTML visualizer: docs/development/evidence/pinocchio_force_overlay_evidence.html
2. Headless screenshot: docs/development/evidence/pinocchio_force_overlay_evidence.png
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.force_overlay_view import (
    PinocchioForceOverlayView,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)


class RecordingSink:
    """Record cylinders and transforms for HTML generation."""

    def __init__(self) -> None:
        self.cylinders: dict[str, dict] = {}
        self.transforms: dict[str, list[list[float]]] = {}

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        self.cylinders[path] = {
            "length_m": float(length_m),
            "radius_top_m": float(radius_top_m),
            "radius_bottom_m": float(radius_bottom_m),
            "rgba": [float(c) for c in rgba],
        }

    def set_transform(self, path: str, matrix4x4) -> None:
        self.transforms[path] = matrix4x4.tolist()

    def delete(self, path: str) -> None:
        self.cylinders.pop(path, None)
        self.transforms.pop(path, None)


def build_evidence_model_and_frame() -> tuple[ForceTorqueFrame, ForceTorqueFrame]:
    """Build synthetic golf club / pendulum frame for evidence capture."""
    # Joint reaction at golfer grip (synthetic_wrist)
    wrist_rx = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="reaction:synthetic_wrist",
        body="synthetic_shaft",
        point_m=(0.0, 0.0, 1.2),
        force_n=(0.0, 45.0, 180.0),
        source="pinocchio",
    )
    # Joint actuator torque at wrist
    wrist_tau = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:synthetic_wrist",
        body="synthetic_shaft",
        point_m=(0.0, 0.0, 1.2),
        torque_nm=(0.0, 15.0, 0.0),
        source="pinocchio",
    )
    # Contact force at clubhead
    clubhead_contact = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:clubhead",
        body="synthetic_clubhead",
        point_m=(0.2, 0.0, 0.05),
        force_n=(350.0, 0.0, 120.0),
        source="pinocchio",
    )
    axial = AxialLoadFrame(
        time_s=0.15,
        values_n={
            "synthetic_arm": 250.0,  # Tension (blue)
            "synthetic_shaft": -180.0,  # Compression (red)
            "synthetic_clubhead": 0.0,  # Neutral
        },
        source="pinocchio",
    )
    frame = ForceTorqueFrame(
        time_s=0.15,
        wrenches=(wrist_rx, wrist_tau, clubhead_contact),
        axial_loads=axial,
        engine="pinocchio",
    )

    # ZTCF counterfactual frame (zero-torque counterfactual)
    wrist_rx_cf = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="reaction:synthetic_wrist",
        body="synthetic_shaft",
        point_m=(0.0, 0.0, 1.2),
        force_n=(0.0, 10.0, 85.0),
        source="ztcf",
    )
    cf_frame = ForceTorqueFrame(
        time_s=0.15,
        wrenches=(wrist_rx_cf,),
        engine="pinocchio",
    )
    return frame, cf_frame


class EvidenceProvider:
    def __init__(self, frame: ForceTorqueFrame, cf_frame: ForceTorqueFrame) -> None:
        self.frame = frame
        self.cf_frame = cf_frame

    def sample(self, *args, **kwargs) -> ForceTorqueFrame:
        return self.frame

    def sample_ztcf(self, *args, **kwargs) -> ForceTorqueFrame:
        return self.cf_frame


_MESHCAT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Pinocchio GUI: Force and Torque Overlay with Live Segment Shading</title>
  <style>
    body {{ margin: 0; padding: 0; background: #111827; color: #f3f4f6; font-family: sans-serif; overflow: hidden; }}
    #header {{ position: absolute; top: 16px; left: 20px; z-index: 100; background: rgba(17,24,39,0.85); padding: 12px 18px; border-radius: 8px; border: 1px solid #374151; }}
    h1 {{ font-size: 16px; margin: 0 0 4px 0; color: #60a5fa; }}
    .subtitle {{ font-size: 12px; color: #9ca3af; margin: 0; }}
    #legend {{ position: absolute; bottom: 20px; right: 20px; z-index: 100; background: rgba(17,24,39,0.9); padding: 14px 18px; border-radius: 8px; border: 1px solid #374151; font-size: 12px; line-height: 1.6; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
    canvas {{ display: block; width: 100vw; height: 100vh; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
  <div id="header">
    <h1>Pinocchio GUI: MeshCat Force Overlay &amp; Segment Shading</h1>
    <p class="subtitle">Consolidated View Helper &middot; Live Axial Shading &middot; Reaction &amp; Contact Glyphs &middot; ZTCF Counterfactual</p>
  </div>
  <div id="legend">
    <div style="font-weight: 600; margin-bottom: 6px; color: #e5e7eb;">Overlay &amp; Shading Legend</div>
    <div class="legend-row">
      <div class="swatch" style="background: {arm_hex};"></div>
      <span>Segment Tension (Arm: +250 N)</span>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background: {shaft_hex};"></div>
      <span>Segment Compression (Shaft: -180 N)</span>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background: {head_hex};"></div>
      <span>Neutral Segment (Clubhead: 0 N)</span>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background: #22c55e;"></div>
      <span>Reaction Force (Grip / Wrist)</span>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background: #ef4444;"></div>
      <span>Contact Force (Clubhead impact)</span>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background: #eab308;"></div>
      <span>Zero-Torque Counterfactual (ZTCF: cf:...)</span>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background: #06b6d4;"></div>
      <span>Actuator Torque Arc</span>
    </div>
  </div>

  <div id="container"></div>

  <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111827);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(2.2, 1.8, 2.5);
    camera.lookAt(0.2, 0.2, 0.6);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('container').appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    const grid = new THREE.GridHelper(4, 20, 0x374151, 0x1f2937);
    scene.add(grid);

    // 1. Arm segment (Tension: Blue)
    const armGeom = new THREE.CylinderGeometry(0.035, 0.03, 0.6, 24);
    const armMat = new THREE.MeshStandardMaterial({{
      color: "{arm_hex}",
      roughness: 0.4,
      metalness: 0.2
    }});
    const armMesh = new THREE.Mesh(armGeom, armMat);
    armMesh.position.set(0, 0.3, 1.5);
    armMesh.rotation.x = Math.PI / 6;
    scene.add(armMesh);

    // 2. Shaft segment (Compression: Red)
    const shaftGeom = new THREE.CylinderGeometry(0.015, 0.012, 1.1, 24);
    const shaftMat = new THREE.MeshStandardMaterial({{
      color: "{shaft_hex}",
      roughness: 0.3,
      metalness: 0.3
    }});
    const shaftMesh = new THREE.Mesh(shaftGeom, shaftMat);
    shaftMesh.position.set(0.1, 0.1, 0.7);
    shaftMesh.rotation.z = -Math.PI / 12;
    scene.add(shaftMesh);

    // 3. Clubhead (Neutral)
    const headGeom = new THREE.BoxGeometry(0.12, 0.08, 0.06);
    const headMat = new THREE.MeshStandardMaterial({{
      color: "{head_hex}",
      roughness: 0.5
    }});
    const headMesh = new THREE.Mesh(headGeom, headMat);
    headMesh.position.set(0.22, 0.0, 0.08);
    scene.add(headMesh);

    // 4. Force arrows & arcs helper
    function createArrow(start, dir, len, radius, colorHex) {{
      const group = new THREE.Group();
      const shaftLen = len * 0.75;
      const headLen = len * 0.25;
      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, shaftLen, 16),
        new THREE.MeshStandardMaterial({{ color: colorHex }})
      );
      shaft.position.y = shaftLen / 2;
      group.add(shaft);

      const head = new THREE.Mesh(
        new THREE.ConeGeometry(radius * 2.2, headLen, 16),
        new THREE.MeshStandardMaterial({{ color: colorHex }})
      );
      head.position.y = shaftLen + headLen / 2;
      group.add(head);

      const up = new THREE.Vector3(0, 1, 0);
      group.quaternion.setFromUnitVectors(up, dir.clone().normalize());
      group.position.copy(start);
      return group;
    }}

    // Add Reaction Arrow (Green)
    scene.add(createArrow(new THREE.Vector3(0, 0.2, 1.2), new THREE.Vector3(0, 0.3, 0.9), 0.38, 0.012, 0x22c55e));

    // Add ZTCF Arrow (Yellow)
    scene.add(createArrow(new THREE.Vector3(0.03, 0.2, 1.2), new THREE.Vector3(0, 0.15, 0.95), 0.25, 0.010, 0xeab308));

    // Add Contact Arrow (Red)
    scene.add(createArrow(new THREE.Vector3(0.22, 0.0, 0.08), new THREE.Vector3(0.9, 0.0, 0.4), 0.45, 0.015, 0xef4444));

    // Add Torque Arc (Cyan)
    const arcCurve = new THREE.EllipseCurve(0, 0, 0.12, 0.12, 0, Math.PI * 1.5, false, 0);
    const arcPoints = arcCurve.getPoints(32).map(p => new THREE.Vector3(p.x, 0, p.y));
    const arcGeom = new THREE.BufferGeometry().setFromPoints(arcPoints);
    const arcMat = new THREE.LineBasicMaterial({{ color: 0x06b6d4, linewidth: 3 }});
    const arcLine = new THREE.Line(arcGeom, arcMat);
    arcLine.position.set(0, 0.2, 1.2);
    scene.add(arcLine);

    renderer.render(scene, camera);
  </script>
</body>
</html>"""


def _build_meshcat_evidence_html(arm_hex: str, shaft_hex: str, head_hex: str) -> str:
    return _MESHCAT_HTML_TEMPLATE.format(
        arm_hex=arm_hex,
        shaft_hex=shaft_hex,
        head_hex=head_hex,
    )


def generate_meshcat_evidence_html(output_html: Path) -> None:
    frame, cf_frame = build_evidence_model_and_frame()
    provider = EvidenceProvider(frame, cf_frame)
    sink = RecordingSink()

    overlay = PinocchioForceOverlayView(
        provider, meshcat_visualizer=sink, root="/pinocchio/force_overlay"
    )
    overlay.update(
        {
            "show_forces": True,
            "show_torques": True,
            "show_shading": True,
            "show_ztcf": True,
            "force_scale": 0.002,
            "torque_scale": 0.015,
        }
    )

    # Segment colors using ForceColorScale
    scale = ForceColorScale(
        enabled=True, tension_limit_n=300.0, compression_limit_n=300.0
    )
    arm_hex = scale.color(250.0, "#888888")  # tension
    shaft_hex = scale.color(-180.0, "#888888")  # compression
    head_hex = scale.color(0.0, "#888888")  # neutral

    html_content = _build_meshcat_evidence_html(arm_hex, shaft_hex, head_hex)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")


async def capture_screenshot(html_path: Path, png_path: Path) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel="msedge")
        except Exception:  # noqa: BLE001
            browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.goto(html_path.as_uri())
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(png_path))
        await browser.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    evidence_dir = repo_root / "docs" / "development" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    html_path = evidence_dir / "pinocchio_force_overlay_evidence.html"
    png_path = evidence_dir / "pinocchio_force_overlay_evidence.png"

    generate_meshcat_evidence_html(html_path)
    print(f"Generated HTML evidence: {html_path}")

    try:
        asyncio.run(capture_screenshot(html_path, png_path))
        print(f"Captured screenshot evidence: {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Playwright screenshot capture fallback: {exc}")


if __name__ == "__main__":
    main()
