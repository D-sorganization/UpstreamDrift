"""Headless Chromium capture of a MeshCat web page (Playwright)."""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            exe = preferred_chromium() or p.chromium.executable_path
    except Exception as exc:  # noqa: BLE001 - any driver failure means unavailable
        return f"playwright driver failed to start: {exc}"
    if not Path(exe).exists():
        return f"chromium is not installed at {exe} (playwright install chromium)"
    return None


class MeshcatPage:
    """Context manager around a headless Chromium page showing a MeshCat URL."""

    def __init__(
        self, url: str, width: int, height: int, settle_ms: int = 2500
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        self._url, self._w, self._h, self._settle = url, width, height, settle_ms
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
