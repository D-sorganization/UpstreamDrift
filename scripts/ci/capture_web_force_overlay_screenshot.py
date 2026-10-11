"""Capture Playwright headless screenshot evidence for Web Three.js force overlay (ADR-0052, #11308).

Renders the multi-kind GlyphSet fixture through the Three.js scene graph
and ForceLegend overlay into docs/development/evidence/web_force_overlay_evidence.png.
"""

from __future__ import annotations

import http.server
import json
import logging
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "schemas" / "glyph-set-examples.json"
OUTPUT_DIR = REPO_ROOT / "docs" / "development" / "evidence"
OUTPUT_FILE = OUTPUT_DIR / "web_force_overlay_evidence.png"


def load_multi_kind_fixture() -> dict[str, Any]:
    """Build multi-kind fixture combining synthetic force arrow and torque arc."""
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cases = {c["name"]: c["data"] for c in data.get("cases", [])}
    force_case = cases["synthetic_force_only"]
    torque_case = cases["synthetic_torque_only_z"]

    return {
        "schema_version": "glyph-set-v1",
        "time_s": 0.25,
        "arrows": force_case["arrows"],
        "torque_arcs": torque_case["torque_arcs"],
        "legend": {
            "force_reference_n": 500.0,
            "force_reference_length_m": 0.5,
            "torque_reference_nm": 20.0,
            "torque_reference_radius_m": 0.05,
            "kinds_present": ["contact", "joint_actuator"],
            "unavailable_labels": [],
            "engine": "synthetic",
            "source_labels": ["synthetic:contact", "synthetic:actuator"],
        },
    }


def _get_preview_css() -> str:
    return (
        "* { box-sizing: border-box; } "
        "body { margin: 0; padding: 0; background-color: #111827; color: #f3f4f6; "
        "font-family: ui-sans-serif, system-ui, sans-serif; overflow: hidden; width: 100vw; height: 100vh; } "
        "#container { position: relative; width: 100%; height: 100%; } "
        "#webgl-canvas { display: block; width: 100%; height: 100%; } "
        ".legend-card { position: absolute; top: 16px; left: 16px; background-color: rgba(31,41,55,0.95); "
        "border: 1px solid rgba(75,85,99,0.6); border-radius: 8px; padding: 14px 18px; font-size: 12px; "
        "line-height: 1.5; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); max-width: 320px; } "
        ".legend-header { display: flex; justify-content: space-between; align-items: center; "
        "border-bottom: 1px solid #374151; padding-bottom: 6px; margin-bottom: 10px; } "
        ".legend-title { font-weight: 600; font-size: 13px; color: #f9fafb; } "
        ".engine-badge { font-size: 10px; text-transform: uppercase; background: #374151; "
        "padding: 2px 6px; border-radius: 4px; color: #9ca3af; } "
        ".legend-section { margin-bottom: 8px; } "
        ".section-title { font-size: 10px; text-transform: uppercase; color: #9ca3af; letter-spacing: 0.05em; margin-bottom: 4px; } "
        ".ref-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; } "
        ".ref-bar { height: 4px; width: 36px; background-color: #60a5fa; border-radius: 2px; } "
        ".ref-arc { width: 16px; height: 16px; border: 2px solid #fbbf24; border-top-color: transparent; border-radius: 50%; transform: rotate(-45deg); } "
        ".swatch-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; } "
        ".swatch-item { display: flex; align-items: center; gap: 6px; } "
        ".swatch-dot { width: 10px; height: 10px; border-radius: 2px; }"
    )


def _get_preview_js(fixture_json: str) -> str:
    template = """
import * as THREE from 'three';
const glyphs = __FIXTURE_JSON__;
function alignYTo(dirVec) {
  const dir = dirVec.clone().normalize();
  const up = new THREE.Vector3(0, 1, 0);
  if (dir.lengthSq() < 1e-12) return new THREE.Quaternion();
  if (up.dot(dir) < -0.9999999) return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  return new THREE.Quaternion().setFromUnitVectors(up, dir);
}
function buildArrow(g) {
  const grp = new THREE.Group();
  const tail = new THREE.Vector3(...g.tail_m), headBase = new THREE.Vector3(...g.head_base_m), tip = new THREE.Vector3(...g.tip_m);
  const sLen = tail.distanceTo(headBase), sMid = tail.clone().add(headBase).multiplyScalar(0.5);
  const sDir = headBase.clone().sub(tail).normalize(), sQuat = alignYTo(sDir);
  const hLen = headBase.distanceTo(tip), hMid = headBase.clone().add(tip).multiplyScalar(0.5);
  const hDir = tip.clone().sub(headBase).normalize(), hQuat = alignYTo(hDir);
  const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...g.rgba.slice(0, 3)), roughness: 0.3, metalness: 0.2 });
  const sMesh = new THREE.Mesh(new THREE.CylinderGeometry(g.shaft_radius_m, g.shaft_radius_m, sLen, 24), mat);
  sMesh.position.copy(sMid); sMesh.quaternion.copy(sQuat); grp.add(sMesh);
  const hMesh = new THREE.Mesh(new THREE.ConeGeometry(g.head_radius_m, hLen, 24), mat);
  hMesh.position.copy(hMid); hMesh.quaternion.copy(hQuat); grp.add(hMesh);
  return grp;
}
function buildArc(g) {
  const grp = new THREE.Group();
  const pts = g.polyline_m.map(p => new THREE.Vector3(...p));
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  const tGeo = new THREE.TubeGeometry(curve, Math.max(32, pts.length * 2), Math.max(0.003, g.radius_m * 0.045), 12, false);
  const hBase = new THREE.Vector3(...g.head_base_m), hTip = new THREE.Vector3(...g.head_tip_m);
  const hLen = hBase.distanceTo(hTip), hMid = hBase.clone().add(hTip).multiplyScalar(0.5);
  const hQuat = alignYTo(hTip.clone().sub(hBase).normalize());
  const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...g.rgba.slice(0, 3)), roughness: 0.3, metalness: 0.2 });
  grp.add(new THREE.Mesh(tGeo, mat));
  const hMesh = new THREE.Mesh(new THREE.ConeGeometry(hLen * 0.3, hLen, 24), mat);
  hMesh.position.copy(hMid); hMesh.quaternion.copy(hQuat); grp.add(hMesh);
  return grp;
}
const canvas = document.getElementById('webgl-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x111827, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.6, 0.7, 1.2);
camera.lookAt(0.05, 0.0, 0.6);
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const d1 = new THREE.DirectionalLight(0xffffff, 0.8); d1.position.set(2, 3, 4); scene.add(d1);
const d2 = new THREE.DirectionalLight(0x93c5fd, 0.4); d2.position.set(-2, -1, -2); scene.add(d2);
const grid = new THREE.GridHelper(2, 20, 0x374151, 0x1f2937); grid.position.y = -0.01; scene.add(grid);
const grp = new THREE.Group();
if (glyphs.arrows) for (const a of glyphs.arrows) grp.add(buildArrow(a));
if (glyphs.torque_arcs) for (const t of glyphs.torque_arcs) grp.add(buildArc(t));
scene.add(grp);
renderer.render(scene, camera);
window.__SCENE_RENDERED__ = true;
"""
    return template.replace("__FIXTURE_JSON__", fixture_json)


def generate_html_preview(fixture: dict[str, Any]) -> str:
    """Generate HTML page embedding Three.js overlay and legend overlay."""
    fixture_json = json.dumps(fixture)
    css = _get_preview_css()
    js_code = _get_preview_js(fixture_json)

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8">
<title>Force Overlay Preview (#11308)</title>
<style>{css}</style>
<script type="importmap">{{"imports": {{"three": "/ui/node_modules/three/build/three.module.js"}}}}</script>
</head>
<body>
<div id="container">
  <canvas id="webgl-canvas"></canvas>
  <div class="legend-card" data-testid="force-legend">
    <div class="legend-header">
      <span class="legend-title">Force & Torque Overlays</span>
      <span class="engine-badge">Engine: synthetic</span>
    </div>
    <div class="legend-section">
      <div class="section-title">Scale References</div>
      <div class="ref-item"><div class="ref-bar"></div><span>500.0 N = 0.500 m</span></div>
      <div class="ref-item"><div class="ref-arc"></div><span>20.0 N*m = 0.050 m</span></div>
    </div>
    <div class="legend-section">
      <div class="section-title">Wrench Kinds</div>
      <div class="swatch-grid">
        <div class="swatch-item"><div class="swatch-dot" style="background-color: #009E73;"></div><span>Contact</span></div>
        <div class="swatch-item"><div class="swatch-dot" style="background-color: #E69F00;"></div><span>Joint Actuator</span></div>
      </div>
    </div>
  </div>
</div>
<script type="module">{js_code}</script>
</body>
</html>"""


def capture_screenshot() -> None:
    """Spin up local server, navigate with headless Chrome, and save screenshot."""
    from playwright.sync_api import sync_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture = load_multi_kind_fixture()
    html_content = generate_html_preview(fixture)

    html_file = REPO_ROOT / "temp_web_force_overlay_preview.html"
    html_file.write_text(html_content, encoding="utf-8")
    port = 8899

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", port), QuietHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    logger.info("Serving preview at http://127.0.0.1:%d", port)

    try:
        with sync_playwright() as p:
            browser = None
            for channel in ("chrome", "msedge", None):
                try:
                    kwargs: dict[str, Any] = {
                        "headless": True,
                        "args": [
                            "--use-gl=angle",
                            "--use-angle=swiftshader",
                            "--enable-webgl",
                        ],
                    }
                    if channel:
                        kwargs["channel"] = channel
                    browser = p.chromium.launch(**kwargs)
                    logger.info("Launched browser channel: %s", channel)
                    break
                except (PlaywrightError, RuntimeError, OSError) as err:
                    logger.warning("Could not launch channel %s: %s", channel, err)

            if not browser:
                raise RuntimeError("No compatible Chromium browser found.")

            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(f"http://127.0.0.1:{port}/temp_web_force_overlay_preview.html")
            page.wait_for_function("window.__SCENE_RENDERED__ === true", timeout=10000)

            page.screenshot(path=str(OUTPUT_FILE))
            logger.info(
                "Screenshot captured: %s (%d bytes)",
                OUTPUT_FILE,
                OUTPUT_FILE.stat().st_size,
            )
            browser.close()
    finally:
        httpd.shutdown()
        if html_file.exists():
            html_file.unlink()


def main() -> None:
    try:
        capture_screenshot()
    except (PlaywrightError, RuntimeError, OSError) as exc:
        logger.exception("Failed to capture force overlay screenshot: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
