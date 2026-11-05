"""``strategy`` CLI sub-command: both numbers, quick-tour toggle, HTML view (#14572)."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from src.shared.python.sg_optimizer.cli import main
from tests.integration.sg_optimizer.test_cli import (
    REPO_ROOT,
    _write_hole_spec,
    _write_profile,
)

pytestmark = pytest.mark.integration

BASELINE = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"


def _run(tmp_path, *extra: str) -> tuple[int, str]:
    hole, profile = tmp_path / "hole.py", tmp_path / "profile.yaml"
    _write_hole_spec(hole)
    _write_profile(profile)
    argv = [
        "strategy",
        "--profile", str(profile),
        "--baseline", str(BASELINE),
        "--hole-spec", str(hole),
        "--conditions", "benign",
        "--resolution", "10.0",
        "--n-samples", "16",
        "--max-iter", "5",
        *extra,
    ]  # fmt: skip
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_strategy_prints_tour_and_personal_numbers(tmp_path) -> None:
    rc, out = _run(tmp_path)
    assert rc == 0
    assert "cli_smoke" in out and "tour:" in out and "personal:" in out
    assert "n/a" not in out.split("delta:")[0]


def test_quick_tour_skips_the_personal_solve(tmp_path) -> None:
    rc, out = _run(tmp_path, "--quick-tour")
    assert rc == 0
    assert "personal: n/a" in out


def test_html_option_writes_the_view(tmp_path) -> None:
    page = tmp_path / "view.html"
    rc, _ = _run(tmp_path, "--html", str(page))
    assert rc == 0
    body = page.read_text(encoding="utf-8")
    assert "<ellipse" in body and "strokes to green" in body
