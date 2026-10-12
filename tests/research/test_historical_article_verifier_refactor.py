"""Exercise Article Verification Without Historical Top-Level File Access."""

from __future__ import annotations

import ast
import collections
from decimal import Decimal
from pathlib import Path
import re
from typing import Any

import pytest

from tests.research.historical_fixture_security import read_fixture, verify_source

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/shot_pattern_analysis/phase2/validation"
SOURCE = BASE / "article_source_repair_v1/root/verify.py"
pytestmark = pytest.mark.unit


def _load(source: str, values: dict[str, Any]) -> dict[str, Any]:
    verify_source("article", source)
    nodes: list[ast.stmt] = [
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    ]
    namespace = {
        "collections": collections,
        "Decimal": Decimal,
        "re": re,
        "Any": Any,
        **values,
    }
    # Trusted fixture AST: fixed path/digests verified before extraction; no external code.
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


def _fixture() -> tuple[dict[str, Any], str]:
    values: dict[str, Any] = {
        "rows": {},
        "cells": {},
        "effects": {},
        "nominals": {},
        "expected_e": {},
        "proto": {"P": "primary"},
    }
    lines = []
    token = r"\numtext{1.0}"
    for i, (club, label) in enumerate(
        [
            ("driver", "Driver"),
            ("seven_iron", "7-Iron"),
            ("pitching_wedge", "Pitching Wedge"),
        ],
        1,
    ):
        key = f"C{i:02}"
        values["cells"][key] = {
            "cell": club + "_shaft_only",
            "source_protocol": "primary",
            "config": {
                "club": {"club_id": club},
                "curve_scale": 1,
                "noise": {"standard_deviations": [1]},
            },
            "nominals": {"Straight": {"residual_m": 1.0}},
        }
        lines.append(f"{key} & {label} & 1 & 1 & S & P")
        for pattern in ["Straight", "Draw", "Fade"]:
            values["rows"][key, pattern] = {
                "supported_n": 10,
                "scoring_paired_n": 8,
                "lateral_sd_m": 1.0,
                "downrange_sd_m": 1.0,
                "target_rmse_m": 1.0,
                "mean_sg_strokes": 1.0,
                "paired_sg_vs_straight": {
                    "estimate": 1.0,
                    "lower_95": 1.0,
                    "upper_95": 1.0,
                },
                "club": club,
                "pattern": pattern,
            }
            lines.append(f"{key} & {pattern} & 10 & " + " & ".join([token] * 4))
            if pattern != "Straight":
                lines.append(f"{key} & {pattern} & 8 & " + " & ".join([token] * 3))
                lines.append(f"{label} & {pattern} & {token} & {token} & 1/0/0")
    values["effects"]["E1"] = {
        "common_supported_n": 6,
        "course_scoring_effect_modification": {
            p: {"estimate": 1.0, "lower_95": 1.0, "upper_95": 1.0}
            for p in ["Draw", "Fade"]
        },
    }
    lines.extend(f"E1 & {p} & 6 & " + " & ".join([token] * 3) for p in ["Draw", "Fade"])
    values["nominals"]["F1A"] = {
        p: dict.fromkeys(["face_deg", "path_deg", "speed_mps", "residual_m"], 1.0)
        for p in ["Straight", "Draw", "Fade"]
    }
    lines.extend(
        f"F1A & {p} & " + " & ".join([token] * 4) for p in ["Straight", "Draw", "Fade"]
    )
    values["expected_e"]["E01"] = [
        ("primary", values["cells"][k]["cell"]) for k in ["C01", "C02"]
    ]
    lines.append("E01 & Twist & C01/P & C02/P")
    lines.append("maximum residual among the 72 selected nominal solutions is " + token)
    return values, "\n".join(lines)


def _result(fn: Any, source: str) -> Any:
    try:
        return fn(source)
    except (ValueError, KeyError) as exc:
        return type(exc).__name__, exc.args


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "count",
        "duplicate",
        "missing",
        "precision",
        "condition",
        "interaction",
        "residual",
        "synthesis",
    ],
)
def test_verifier_preserves_numeric_identity_and_error_contract(mutation: str) -> None:
    values, article = _fixture()
    before = _load(read_fixture("article", archived=True), values)
    after = _load(read_fixture("article"), values)
    if mutation == "count":
        article = article.replace("Straight & 10", "Straight & 9", 1)
    if mutation == "duplicate":
        article += "\n" + next(
            x for x in article.splitlines() if x.startswith("C01 & Straight")
        )
    if mutation == "missing":
        article = "\n".join(article.splitlines()[1:])
    if mutation == "precision":
        article = article.replace(r"\numtext{1.0}", r"\numtext{2.0}", 1)
    if mutation == "condition":
        article = article.replace("Driver & 1 & 1", "Driver & 2 & 1")
    if mutation == "interaction":
        article = article.replace("C01/P & C02/P", "C02/P & C01/P")
    if mutation == "residual":
        article = article.rsplit("\n", 1)[0]
    if mutation == "synthesis":
        article = article.replace("1/0/0", "0/1/0", 1)
    expected = _result(before["verify"], article)
    assert _result(after["verify"], article) == expected
    if mutation == "none":
        assert expected["numeric_fields"] == 85
    else:
        assert isinstance(expected, tuple)


def test_verifier_has_separate_row_and_coverage_contracts() -> None:
    functions = {
        n.name
        for n in ast.parse(read_fixture("article")).body
        if isinstance(n, ast.FunctionDef)
    }
    assert {"_numeric_row", "_verify_coverage"} <= functions
