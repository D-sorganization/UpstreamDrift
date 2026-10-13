"""Public figure entry points stay within the architecture budget."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from src.tools.shot_pattern_analysis import academic_plots

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    ["write_academic_dispersion_figure", "write_overhead_trajectory_figure"],
)
def test_figure_entry_points_have_bounded_body(name: str) -> None:
    source = inspect.getsource(getattr(academic_plots, name))
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    assert function.end_lineno is not None
    assert function.end_lineno - function.lineno + 1 <= 100


def test_dispersion_entry_point_has_bounded_parameters() -> None:
    signature = inspect.signature(academic_plots.write_academic_dispersion_figure)
    assert len(signature.parameters) <= 8
