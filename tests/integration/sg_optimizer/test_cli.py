"""End-to-end CLI smoke test."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from src.shared.python.sg_optimizer.cli import main


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_hole_spec(path: Path) -> None:
    path.write_text(
        "from src.shared.python.sg_optimizer.course.rasterize import (\n"
        "    SyntheticHole, RectFeature, CircleFeature,\n"
        ")\n"
        "HOLE = SyntheticHole(\n"
        "    name='cli_smoke', par=3,\n"
        "    tee=(0.0, 0.0), pin=(120.0, 0.0),\n"
        "    bbox=(-10.0, 140.0, -25.0, 25.0),\n"
        "    features=(\n"
        "        RectFeature('fairway', 0.0, 130.0, -10.0, 10.0),\n"
        "        CircleFeature('green', 120.0, 0.0, 8.0),\n"
        "    ),\n"
        ")\n"
    )


def _write_profile(path: Path) -> None:
    path.write_text(
        "name: cli_smoke\n"
        f"baseline: {REPO_ROOT / 'data' / 'sg_optimizer' / 'baselines' / 'pga_tour.yaml'}\n"
        "clubs: {}\n"
        "putting:\n"
        "  make_pct_multipliers: {}\n"
        "  three_putt_avoidance: 1.0\n"
        "short_game: {}\n"
        "notes: cli\n"
    )


def test_cli_end_to_end(tmp_path, monkeypatch):
    hole_spec = tmp_path / "hole.py"
    profile = tmp_path / "profile.yaml"
    _write_hole_spec(hole_spec)
    _write_profile(profile)
    baseline = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"

    monkeypatch.setattr(sys, "argv", ["sg-optimizer"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--profile",
                str(profile),
                "--baseline",
                str(baseline),
                "--hole-spec",
                str(hole_spec),
                "--conditions",
                "benign",
                "--resolution",
                "10.0",
                "--n-samples",
                "16",
                "--max-iter",
                "5",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["hole"] == "cli_smoke"
    assert payload["tee_optimal_action"]["club"] in (
        "driver",
        "3_wood",
        "5_iron",
        "7_iron",
        "9_iron",
        "pw",
        "sw",
        "lw",
    )
