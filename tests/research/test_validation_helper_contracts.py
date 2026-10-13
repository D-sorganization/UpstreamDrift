"""Execute isolated helper contracts without historical top-level I/O."""

from __future__ import annotations

import ast
import dataclasses
from decimal import Decimal
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any
import unittest.mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/shot_pattern_analysis/phase2/validation"
pytestmark = pytest.mark.unit


def tree(relative: str) -> ast.Module:
    return ast.parse((BASE / relative).read_text())


def run(nodes: list[ast.stmt], values: dict[str, Any]) -> dict[str, Any]:
    namespace = {
        "Decimal": Decimal,
        "Any": Any,
        "dataclasses": dataclasses,
        "np": np,
        "unittest": unittest,
        **values,
    }
    code = compile(
        ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
        "isolated-helper",
        "exec",
    )
    FunctionType(code, namespace)()
    return namespace


def function(relative: str, name: str, **values: Any) -> Any:
    node = next(
        n
        for n in ast.walk(tree(relative))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return run([node], values)[name]


def decimal_site(site: str, token: str, expected: float) -> Any:
    if site == "candidate":
        rows: list[dict[str, Any]] = []
        fn = function(
            "article_source_repair_v1/astra/audit_candidate.py",
            "check",
            rows=rows,
            failures=[],
        )
        fn("test", token, expected)
        return rows[0]
    if site == "root":
        function("article_source_repair_v1/root/verify.py", "number")(token, expected)
        return None
    relative = f"article_source_repair_v1/astra/{'audit_interactions' if site == 'interactions' else 'final_check'}.py"
    loops = [n for n in ast.walk(tree(relative)) if isinstance(n, ast.For)]
    if site == "interactions":
        node = next(n for n in loops if isinstance(n.target, ast.Tuple))
        result = run(
            [node],
            {
                "tokens": [token] * 3,
                "checks": [],
                "key": 0,
                "f": ["", "", "", "Draw"],
                "v": dict.fromkeys(["estimate", "lower_95", "upper_95"], expected),
            },
        )
        return result["checks"][0]
    node = next(
        n for n in loops if isinstance(n.target, ast.Name) and n.target.id == "f"
    )
    run([node], {"fields": [{"display": token, "value": expected}]})
    return None


@pytest.mark.parametrize("site", ["candidate", "root", "interactions", "final"])
@pytest.mark.parametrize(
    "token,value",
    [
        ("0", 0.0),
        ("-0.000", -0.0),
        ("1.230", 1.23),
        ("-2.73e-05", -2.73e-5),
        ("7.461556e-05", 7.461555935e-5),
    ],
)
def test_finite_precision_preserved(site: str, token: str, value: float) -> None:
    result = decimal_site(site, token, value)
    if result is not None:
        assert result.get("matches_display_precision", result.get("matches")) is True


@pytest.mark.parametrize("site", ["candidate", "root", "interactions", "final"])
@pytest.mark.parametrize("token", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_nonfinite_display_rejected(site: str, token: str) -> None:
    with pytest.raises(ValueError):
        decimal_site(site, token, 0.0)


@pytest.mark.parametrize("site", ["candidate", "root", "interactions", "final"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_expected_rejected(site: str, value: float) -> None:
    with pytest.raises(ValueError):
        decimal_site(site, "0.50", value)


@pytest.mark.parametrize("site", ["candidate", "root", "interactions", "final"])
def test_outside_display_precision_fails(site: str) -> None:
    if site in {"root", "final"}:
        with pytest.raises(ValueError):
            decimal_site(site, "1.23", 1.25)
    else:
        result = decimal_site(site, "1.23", 1.25)
        assert result.get("matches_display_precision", result.get("matches")) is False


def test_root_zero_cannot_hide_small_nonzero() -> None:
    with pytest.raises(ValueError, match="rounded-to-zero"):
        decimal_site("root", "0.0000", 1e-8)


@pytest.mark.parametrize("fail", [False, True])
def test_actual_instrumentation_restores_method(fail: bool) -> None:
    worker = next(
        n
        for n in ast.walk(tree("selected_renderer_representative_v1/qualify.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "worker"
    )
    node = next(n for n in ast.walk(worker) if isinstance(n, ast.With))
    calls = []

    class Figure:
        def savefig(self, path: str, **kwargs: Any) -> str:
            calls.append((path, kwargs))
            return "saved"

    original = Figure.savefig

    def save(fig: Figure, path: str, **kwargs: Any) -> str:
        return original(fig, path, **kwargs)

    def exercise() -> None:
        assert Figure.savefig is save
        assert Figure().savefig("file", dpi=123) == "saved"
        if fail:
            raise RuntimeError("sentinel")

    isolated = ast.With(
        items=node.items,
        body=[
            ast.Expr(
                ast.Call(func=ast.Name("exercise", ast.Load()), args=[], keywords=[])
            )
        ],
    )
    if fail:
        with pytest.raises(RuntimeError, match="sentinel"):
            run([isolated], {"Figure": Figure, "save": save, "exercise": exercise})
    else:
        run([isolated], {"Figure": Figure, "save": save, "exercise": exercise})
    assert Figure.savefig is original
    assert calls == [("file", {"dpi": 123})]


def test_dataclass_instance_not_class_conversion() -> None:
    fn = function("selected_renderer_representative_v1/qualify.py", "data_value")

    @dataclasses.dataclass
    class Example:
        count: int

    assert fn(Example(3)) == {"count": 3}
    assert fn(Example) is Example


@pytest.mark.parametrize("present", [False, True])
def test_scrollbar_guard_before_use(present: bool) -> None:
    node = next(
        n
        for n in ast.walk(tree("capture_bounds_gui.py"))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "scrollbar"
    )
    scrollbar = SimpleNamespace(maximum=lambda: 42) if present else None
    if present:
        run([node], {"scrollbar": scrollbar})
    else:
        with pytest.raises(RuntimeError):
            run([node], {"scrollbar": scrollbar})


@pytest.mark.parametrize("reference", [None, {"artifact_sha256": {}}])
def test_numeric_reference_guard(reference: Any) -> None:
    node = next(
        n
        for n in ast.walk(tree("v6_output_review_harness_v1/review_batch.py"))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "reference"
    )
    if reference is None:
        with pytest.raises(ValueError):
            run(
                [node],
                {
                    "reference": reference,
                    "args": SimpleNamespace(expected_numeric="test.json"),
                },
            )
    else:
        run([node], {"reference": reference})


@pytest.mark.parametrize(
    "spec", [None, SimpleNamespace(loader=None), SimpleNamespace(loader=object())]
)
def test_import_spec_guard(spec: Any) -> None:
    node = next(
        n
        for n in tree("v6_output_review_harness_v1/test_review_batch.py").body
        if isinstance(n, ast.If)
    )
    if spec is None or spec.loader is None:
        with pytest.raises(RuntimeError):
            run([node], {"spec": spec})
    else:
        run([node], {"spec": spec})


@pytest.mark.parametrize("site", ["candidate", "root", "interactions", "final"])
def test_display_rounding_boundary_preserved(site: str) -> None:
    # A 0.01 display unit permits just under half a unit, but not just over.
    result = decimal_site(site, "1.23", 1.234999)
    if result is not None:
        assert result.get("matches_display_precision", result.get("matches")) is True
    if site in {"root", "final"}:
        with pytest.raises(ValueError):
            decimal_site(site, "1.23", 1.235001)
    else:
        result = decimal_site(site, "1.23", 1.235001)
        assert result.get("matches_display_precision", result.get("matches")) is False
