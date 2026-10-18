"""NSX-100: the pure Variability & Consistency view (no PyQt, no numpy)."""

from __future__ import annotations

import sys

import pytest

from src.tools.grip_wrench_plots import variability_view as vv
from src.shared.python.loop_nullspace.plot_model import NSXPlotModel
from tests.unit.tools.grip_wrench_plots.golden_support import (
    golden_model,
    variability_golden_dict,
    variability_golden_model,
    view_text_fixture,
)

pytestmark = pytest.mark.unit


def test_view_module_has_no_gui_imports() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(vv))
    names = {
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not any(str(m).split(".")[0] in {"PyQt6", "matplotlib", "numpy"} for m in names)


def test_complete_view_shows_every_expected_string() -> None:
    view = vv.variability_view(variability_golden_model())
    text = view.text()
    missing = [s for s in view_text_fixture()["complete"]["expected"] if s not in text]
    assert missing == []


def test_sections_and_unvalidated_badge_come_from_the_payload() -> None:
    raw = variability_golden_dict()
    view = vv.variability_view(variability_golden_model())
    assert [s.name for s in view.sections] == list(vv.SECTION_NAMES)
    kernel = view.section("kernel_heatmap")
    for badge in raw["sensitivity"]["kernel_heatmap"]["decision"]["badges"]:
        assert badge in kernel.badge
    assert "UNVALIDATED" in kernel.badge


def test_withheld_arm_carries_its_reason_and_no_number() -> None:
    raw = variability_golden_dict()["sensitivity"]["strategy_pareto"]["points"][0]
    row = next(
        r for r in vv.variability_view(variability_golden_model()).section("strategy_pareto").rows
        if r[0] == raw["label"]
    )
    assert raw["reason"] in row
    assert "withheld" in row
    assert not any(c.replace(".", "").isdigit() for c in row[2:6])


def test_unavailable_engine_shows_its_reason() -> None:
    fx = view_text_fixture()["unavailable"]
    model = NSXPlotModel.from_dict(fx["plot_model"])
    view = vv.variability_view(model)
    assert view.status.startswith("UNAVAILABLE: ")
    for s in fx["expected"]:
        assert s in view.text()
    assert all(sec.status.startswith("UNAVAILABLE: ") for sec in view.sections)


def test_model_without_sensitivity_says_it_was_not_requested() -> None:
    view = vv.variability_view(golden_model())
    assert view.status == vv.NOT_REQUESTED
    assert view.sections == ()


def test_input_is_type_checked() -> None:
    with pytest.raises(TypeError):
        vv.variability_view({})  # type: ignore[arg-type]
