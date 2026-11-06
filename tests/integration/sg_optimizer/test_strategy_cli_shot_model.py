"""``strategy --shot-table`` / ``--shot-model``: the selected model reaches the personal solve (#14744).

Wiring proof only: the fixture tables are hand-built, not physics; they show the
CLI injects the selected model and reports which one it used.
"""

from __future__ import annotations

import json
import re

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    Convergence,
    DispersionCell,
    DispersionTable,
)
from src.shared.python.sg_optimizer.shot_model.table_model import CLUB_SKILL
from tests.integration.sg_optimizer.test_strategy_cli import _run

pytestmark = pytest.mark.integration


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
    _table(good, 120.0, 2.0)
    _table(bad, 60.0, 40.0)
    _, out_good = _run(tmp_path, "--shot-table", str(good))
    _, out_bad = _run(tmp_path, "--shot-table", str(bad))
    (tour_g, personal_g), (tour_b, personal_b) = _numbers(out_good), _numbers(out_bad)
    assert tour_g == tour_b
    assert personal_b > personal_g + 0.2


def test_selected_table_provenance_is_in_text_and_html(tmp_path) -> None:
    path, page = tmp_path / "t.json", tmp_path / "view.html"
    table = _table(path, 120.0, 2.0)
    rc, out = _run(tmp_path, "--shot-table", str(path), "--html", str(page))
    assert rc == 0
    assert f"Shot model: table {table.table_hash}" in out
    assert f"table {table.table_hash}" in page.read_text(encoding="utf-8")


def test_default_reports_the_baseline_and_matches_explicit_baseline(tmp_path) -> None:
    _, default = _run(tmp_path)
    _, explicit = _run(tmp_path, "--shot-model", "baseline")
    assert "Shot model: baseline" in default
    assert _numbers(default) == _numbers(explicit)


def test_missing_table_is_rejected(tmp_path) -> None:
    with pytest.raises(ContractViolationError, match="nope.json"):
        _run(tmp_path, "--shot-table", str(tmp_path / "nope.json"))


def test_tampered_table_is_rejected(tmp_path) -> None:
    path = tmp_path / "t.json"
    _table(path, 120.0, 2.0)
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
    _table(path, 120.0, 2.0)
    with pytest.raises(ContractViolationError, match="quick-tour"):
        _run(tmp_path, "--quick-tour", "--shot-table", str(path))


def test_table_and_backend_are_mutually_exclusive(tmp_path, capsys) -> None:
    path = tmp_path / "t.json"
    _table(path, 120.0, 2.0)
    with pytest.raises(SystemExit):
        _run(tmp_path, "--shot-table", str(path), "--shot-model", "baseline")
    assert "not allowed with" in capsys.readouterr().err
