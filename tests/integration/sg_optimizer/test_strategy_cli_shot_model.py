"""``strategy --shot-table`` / ``--shot-model``: the selected model reaches the personal solve (#14744).

Wiring proof only: the fixture tables are hand-built, not physics; they show the
CLI injects the selected model and reports which one it used.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    Convergence,
    DispersionCell,
    DispersionTable,
)
from src.shared.python.sg_optimizer.shot_model.table_model import CLUB_SKILL
from src.shared.python.sg_optimizer.cli import main
from tests.integration.sg_optimizer.test_cli import _write_profile
from tests.integration.sg_optimizer.test_strategy_cli import BASELINE

pytestmark = pytest.mark.integration

# An injected model runs the scalar Bellman sweep (cost per raster cell), so the
# hole is a short fairway-to-green strip.
_HOLE = """from src.shared.python.sg_optimizer.course.rasterize import (
    SyntheticHole, RectFeature, CircleFeature,
)
HOLE = SyntheticHole(
    name='shot_model_strip', par=3, tee=(0.0, 0.0), pin=(100.0, 0.0),
    bbox=(-10.0, 125.0, -12.0, 12.0),
    features=(
        RectFeature('fairway', 0.0, 115.0, -12.0, 12.0),
        CircleFeature('green', 100.0, 0.0, 10.0),
    ),
)
"""


def _run(tmp_path, *extra: str) -> tuple[int, str]:
    hole, profile = tmp_path / "hole.py", tmp_path / "profile.yaml"
    hole.write_text(_HOLE, encoding="utf-8")
    _write_profile(profile)
    argv = [
        "strategy",
        "--profile", str(profile),
        "--baseline", str(BASELINE),
        "--hole-spec", str(hole),
        "--conditions", "benign",
        "--resolution", "10.0",
        "--n-samples", "12",
        "--max-iter", "5",
        *extra,
    ]  # fmt: skip
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def _table(path, carry: float, sd: float) -> DispersionTable:
    """Every skill lands ``carry`` yards out with ``sd`` yards of scatter."""
    cells = tuple(
        DispersionCell(
            s, s, "fairway", "calm", "flat", (carry, 0.0), (sd, sd), 0.0, 500
        )
        for s in sorted(set(CLUB_SKILL.values()))
    )
    table = DispersionTable(cells, {}).with_convergence(Convergence(True, 0.0))
    table.to_json(path)
    return table


def _numbers(out: str) -> tuple[float, float]:
    m = re.search(r"tour: ([\d.]+)\s+personal: ([\d.]+)", out)
    assert m is not None, out
    return float(m.group(1)), float(m.group(2))


def test_different_tables_change_the_personal_result_not_the_tour(tmp_path) -> None:
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    _table(good, 100.0, 2.0)
    _table(bad, 40.0, 30.0)
    _, out_good = _run(tmp_path, "--shot-table", str(good))
    _, out_bad = _run(tmp_path, "--shot-table", str(bad))
    (tour_g, personal_g), (tour_b, personal_b) = _numbers(out_good), _numbers(out_bad)
    assert tour_g == tour_b
    assert personal_b > personal_g + 0.2


def test_selected_table_provenance_is_in_text_and_html(tmp_path) -> None:
    path, page = tmp_path / "t.json", tmp_path / "view.html"
    table = _table(path, 100.0, 2.0)
    rc, out = _run(tmp_path, "--shot-table", str(path), "--html", str(page))
    assert rc == 0
    assert f"Shot model: table {table.table_hash}" in out
    assert f"table {table.table_hash}" in page.read_text(encoding="utf-8")


def test_default_reports_the_baseline_and_matches_explicit_baseline(tmp_path) -> None:
    _, default = _run(tmp_path)
    _, explicit = _run(tmp_path, "--shot-model", "baseline")
    assert "Shot model: baseline" in default and "Shot model: baseline" in explicit
    # Same model, but an injected one runs the scalar sweep with its own common
    # random draws, so only Monte Carlo noise separates the two numbers.
    assert _numbers(default)[0] == _numbers(explicit)[0]
    assert abs(_numbers(default)[1] - _numbers(explicit)[1]) < 0.5


def test_missing_table_is_rejected(tmp_path) -> None:
    with pytest.raises(ContractViolationError, match="nope.json"):
        _run(tmp_path, "--shot-table", str(tmp_path / "nope.json"))


def test_tampered_table_is_rejected(tmp_path) -> None:
    path = tmp_path / "t.json"
    _table(path, 100.0, 2.0)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cells"][0]["mean"][0] = 500.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="hash"):
        _run(tmp_path, "--shot-table", str(path))


def test_unknown_backend_is_rejected_not_replaced(tmp_path) -> None:
    with pytest.raises(ContractViolationError, match="unknown shot-model backend"):
        _run(tmp_path, "--shot-model", "nope")


def test_quick_tour_rejects_an_explicit_model(tmp_path) -> None:
    path = tmp_path / "t.json"
    _table(path, 100.0, 2.0)
    with pytest.raises(ContractViolationError, match="quick-tour"):
        _run(tmp_path, "--quick-tour", "--shot-table", str(path))


def test_table_and_backend_are_mutually_exclusive(tmp_path, capsys) -> None:
    path = tmp_path / "t.json"
    _table(path, 100.0, 2.0)
    with pytest.raises(SystemExit):
        _run(tmp_path, "--shot-table", str(path), "--shot-model", "baseline")
    assert "not allowed with" in capsys.readouterr().err
