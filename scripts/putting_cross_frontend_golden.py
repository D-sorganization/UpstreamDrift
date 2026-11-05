"""Generate the cross-frontend putting golden fixture and parity report (#14449).

Runs each putt through the shared core twice -- the ``/simulate-3d`` route the
React and Tauri frontends call, and the PyQt6 scene builder -- and freezes both
for ``ui/src/api/puttingCoreParity.test.ts``. No physics lives here.

Usage::

    python3 -m scripts.putting_cross_frontend_golden
    (cd ui && npx prettier --write src/api/puttingCore.golden.json)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.rate_limit import limiter
from src.api.routes.putting_green import router
from src.tools.putting_green_gui._scene_builder import PuttConfig, build_putt_scene

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "ui" / "src" / "api" / "puttingCore.golden.json"
REPORT_PATH = REPO_ROOT / "docs" / "development" / "putting_cross_frontend_parity.md"
ROUTE = "/tools/putting-green/simulate-3d"

#: Both frontends show the core's numbers, so distances agree to float noise.
#: Launch speed differs by definition (PyQt6 shows the first integrated
#: sample, the route the impact model's ball speed); 2e-3 covers the measured
#: ~0.09 % gap and nothing larger.
TOLERANCE = {"distance_m": 1e-6, "launch_speed_rel": 2e-3}

#: (name, putter speed m/s, hole distance m, stimp, cross slope deg), taken
#: from lane 6's 2026-11-04 gap record on #14449.
CASES = (
    ("flat-10ft", 2.0, 3.0, 10.0, 0.0),
    ("flat-long-slow", 4.0, 9.0, 8.0, 0.0),
    ("cross-slope-1deg", 2.0, 3.0, 10.0, 1.0),
)


def slope_request(slope_deg: float) -> dict[str, float]:
    """PyQt6's cross slope (downhill toward -y) as the route's grade and aspect."""
    if not 0.0 <= slope_deg < 90.0:
        raise ValueError(f"slope_deg must be in [0, 90); got {slope_deg}")
    if slope_deg == 0.0:
        return {}
    return {
        "grade_percent": 100.0 * math.tan(math.radians(slope_deg)),
        "downhill_aspect_deg": -90.0,
    }


def route_response(request: dict) -> dict:
    """POST to the real route (no rate limit); keep the read fields and rest."""
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    enabled, limiter.enabled = limiter.enabled, False
    try:
        reply = TestClient(app).post(ROUTE, json=request)
    finally:
        limiter.enabled = enabled
    reply.raise_for_status()
    body = reply.json()
    rest = {key: body["samples"][-1][key] for key in ("t_s", "x_m", "y_m")}
    return {
        "samples": [rest],
        "collision": {"ball_speed_mps": body["collision"]["ball_speed_mps"]},
        "surface": {key: body["surface"][key] for key in ("hole_x_m", "hole_y_m")},
        "holed": body["holed"],
        "total_distance_m": body["total_distance_m"],
    }


def route_values(response: dict) -> dict:
    """What the frontends derive from a route response (the ball starts at 0)."""
    rest, surface = response["samples"][-1], response["surface"]
    leave = math.hypot(
        rest["x_m"] - surface["hole_x_m"], rest["y_m"] - surface["hole_y_m"]
    )
    return {
        "holed": response["holed"],
        "total_roll_m": response["total_distance_m"],
        "leave_m": 0.0 if response["holed"] else leave,
        "rest_dy_m": rest["y_m"],
        "launch_speed_mps": response["collision"]["ball_speed_mps"],
    }


def pyqt6_values(speed: float, distance: float, stimp: float, slope: float) -> dict:
    """The PyQt6 scene's summary for the same putt."""
    scene = build_putt_scene(
        PuttConfig(
            putter_speed_ms=speed, cup_distance_m=distance, stimp=stimp, slope_deg=slope
        )
    )
    return {
        "holed": scene.holed,
        "total_roll_m": scene.total_roll_m,
        "leave_m": 0.0 if scene.holed else scene.final_distance_to_cup_m,
        "rest_dy_m": float(scene.trajectory_xyz[-1][1] - scene.start_xyz[1]),
        "launch_speed_mps": scene.launch_speed_ms,
    }


def build_golden() -> dict:
    """Every case's request, trimmed route response and PyQt6 summary."""
    cases = []
    for name, speed, distance, stimp, slope in CASES:
        request = {
            "putter_speed_mps": speed,
            "stimp_rating": stimp,
            "hole_x_m": distance,
            **slope_request(slope),
        }
        cases.append(
            {
                "name": name,
                "request": request,
                "response": route_response(request),
                "pyqt6": pyqt6_values(speed, distance, stimp, slope),
            }
        )
    return {
        "tolerance": TOLERANCE,
        "cases": cases,
    }


_GAPS = """
## Gaps

1. **Launch speed is defined differently.** PyQt6 shows the first integrated
   sample; React and Tauri show `collision.ball_speed_mps` (about 0.09 % apart).
2. **Slope is spelled differently.** PyQt6 takes a cross slope in degrees; the
   route takes `grade_percent = 100 * tan(slope)`, `downhill_aspect_deg = -90`.
3. **No break or miss field in the route.** PyQt6 reports `peak_break_m`; the
   frontends read only the rest point, so parity holds on the rest offset.
4. **Tauri is checked through the React build.** The shell holds no putting
   logic; the test asserts the local backend URL. A native Tauri build cannot
   run on the build host (missing WebKit libraries), so the shell is untested.
"""


def _table(rows: list[list[str]]) -> list[str]:
    """A Markdown table padded the way Prettier writes it (header first)."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        [cell.ljust(width) for cell, width in zip(row, widths, strict=True)]
        for row in rows
    ]
    lines.insert(1, ["-" * width for width in widths])
    return ["| " + " | ".join(line) + " |" for line in lines]


def render_report(golden: dict) -> str:
    """The parity report for ``golden``; postcondition: ends with a newline."""
    rows = [["Case", "Holed", "Roll (m)", "Leave (m)", "Rest y (m)", "Roll gap (m)"]]
    rows[0].append("Launch gap")
    for case in golden["cases"]:
        route, pyqt6 = route_values(case["response"]), case["pyqt6"]
        gap = route["launch_speed_mps"] / pyqt6["launch_speed_mps"] - 1.0
        rows.append(
            [
                case["name"],
                "yes" if pyqt6["holed"] else "no",
                f"{pyqt6['total_roll_m']:.3f}",
                f"{pyqt6['leave_m']:.3f}",
                f"{pyqt6['rest_dy_m']:.3f}",
                f"{abs(route['total_roll_m'] - pyqt6['total_roll_m']):.1e}",
                f"{gap * 100:+.3f} %",
            ]
        )
    tolerance = golden["tolerance"]
    return "\n".join(
        [
            "# Putting Cross-Frontend Parity (#14449)\n",
            "Generated by `python3 -m scripts.putting_cross_frontend_golden`; do not",
            "edit by hand. Every number comes from the shared core, once through the",
            "`/simulate-3d` route (React and Tauri) and once through the PyQt6 scene.\n",
            f"Tolerances: distances {tolerance['distance_m']:g} m absolute; launch",
            f"speed {tolerance['launch_speed_rel']:g} relative.\n",
            *_table(rows),
            _GAPS,
        ]
    )


def main() -> None:
    """Regenerate the fixture and the report from the core."""
    golden = build_golden()
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(golden), encoding="utf-8")


if __name__ == "__main__":
    main()
