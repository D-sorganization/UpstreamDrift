"""Headless Chromium capture of a MeshCat web page (Playwright)."""

from __future__ import annotations

from importlib.util import find_spec
import math
import os
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.golf_view_presets import VIEWER_FOV_Y_RAD

Image8 = NDArray[np.uint8]
CHROMIUM_ARGS = (
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--no-sandbox",
)
SCREENSHOT_TIMEOUT_MS = 180_000
_HIDE_CONTROLS_CSS = (
    ".dg, .dg.main, #stats, #stats-plot, .stats{display:none !important}"
)
_HIDE_FIXED_JS = (
    "()=>{for(const e of document.querySelectorAll('div')){"
    "const s=e.getAttribute('style')||'';"
    "if(s.includes('fixed')&&e.children.length<=2&&e.clientWidth<120)"
    "e.style.display='none';}}"
)
_ONE_FRAME_JS = "()=>new Promise(r=>requestAnimationFrame(r))"
# three.js defaults to 75 deg, which left the golfer at about a quarter of the
# frame height (NV-9, #11697); every viewer uses the shared field of view.
_SET_FOV_JS = (
    "d=>{if(typeof viewer==='undefined'||!viewer.camera)return false;"
    "viewer.camera.fov=d;viewer.camera.updateProjectionMatrix();return true;}"
)


def preferred_chromium() -> str | None:
    """Chromium executable to launch: the env override, else a full Playwright build.

    The full Chromium build renders SwiftShader WebGL several times faster than
    Playwright's default headless shell; ``None`` lets Playwright choose.
    """
    override = os.environ.get("NATIVE_VIEWER_CHROMIUM")
    if override:
        return override
    cache = Path(
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        or Path.home() / ".cache" / "ms-playwright"
    )
    for pattern in ("chromium-*/chrome-linux*/chrome",):
        found = sorted(cache.glob(pattern))
        if found:
            return str(found[-1])
    return None


def playwright_unavailable_reason() -> str | None:
    """Why headless Chromium capture cannot run here, or ``None``."""
    if find_spec("playwright") is None:
        return "playwright is not installed (pip install playwright; playwright install chromium)"
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            exe = preferred_chromium() or p.chromium.executable_path
    except (PlaywrightError, OSError, RuntimeError) as exc:
        return f"playwright driver failed to start: {exc}"
    if not Path(exe).exists():
        return f"chromium is not installed at {exe} (playwright install chromium)"
    return None


class MeshcatPage:
    """Context manager around a headless Chromium page showing a MeshCat URL.

    The viewer camera gets the vertical field of view ``fov_y_rad`` (default
    the shared ``VIEWER_FOV_Y_RAD``) on entry.
    """

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        settle_ms: int = 2500,
        fov_y_rad: float = VIEWER_FOV_Y_RAD,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if not (math.isfinite(fov_y_rad) and 0.0 < fov_y_rad < math.pi):
            raise ValueError(f"fov_y_rad must lie in (0, pi), got {fov_y_rad}")
        self._url, self._w, self._h, self._settle = url, width, height, settle_ms
        self.fov_y_rad = fov_y_rad
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None

    def __enter__(self) -> MeshcatPage:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        chromium = self._pw.chromium
        self._browser = chromium.launch(
            executable_path=preferred_chromium(), args=list(CHROMIUM_ARGS)
        )
        self._page = self._browser.new_page(
            viewport={"width": self._w, "height": self._h}
        )
        self._page.goto(self._url)
        self._page.wait_for_timeout(self._settle)
        self._page.add_style_tag(content=_HIDE_CONTROLS_CSS)
        self._page.evaluate(_HIDE_FIXED_JS)
        self.apply_fov()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for closer in (
            getattr(self._browser, "close", None),
            getattr(self._pw, "stop", None),
        ):
            if closer is not None:
                closer()

    def apply_fov(self) -> None:
        """Set the viewer camera's vertical field of view to ``fov_y_rad``.

        Raises ``RuntimeError`` when the page has no MeshCat viewer camera.
        """
        if not self._page.evaluate(_SET_FOV_JS, math.degrees(self.fov_y_rad)):
            raise RuntimeError("the MeshCat page has no viewer camera to set")

    def look_at(self, target_three: tuple[float, float, float]) -> None:
        """Aim the orbit controls at ``target_three`` (viewer Y-up coordinates)."""
        self._page.evaluate(
            "t=>{viewer.controls.target.set(t[0],t[1],t[2]);viewer.controls.update();}",
            list(target_three),
        )

    def settle(self, ms: int = 100) -> None:
        """Wait for pending scene messages, then an animation frame."""
        self._page.wait_for_timeout(ms)
        self._page.evaluate(_ONE_FRAME_JS)

    def screenshot(self) -> Image8:
        """The current viewport as an RGB uint8 array."""
        import cv2

        raw = self._page.screenshot(timeout=SCREENSHOT_TIMEOUT_MS)
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Chromium returned an undecodable screenshot")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
