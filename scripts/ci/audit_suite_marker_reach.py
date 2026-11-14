#!/usr/bin/env python3
"""Audit suite-marker reach and directory auto-marking coverage (#14947)."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.check_suite_marker_ratchet import (  # noqa: E402
    _decorator_suite_markers,
    _is_test_class,
    _is_test_function,
    _load_baseline,
    _module_suite_markers,
    iter_test_files,
)
from tests.helpers.suite_marker_reach import (  # noqa: E402
    OPT_OUT_MARKER,
    infer_directory_suite_marker,
    summarize_gate_selectability,
)
from tests.support.suite_markers import BASELINE_PATH  # noqa: E402

SELECTABILITY_TARGET_PCT = 85.0


def _has_opt_out(
    node: ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef,
) -> bool:
    for dec in node.decorator_list:
        attr = (
            dec.attr
            if isinstance(dec, ast.Attribute)
            else getattr(dec.func, "attr", None)
            if isinstance(dec, ast.Call)
            else None
        )
        if attr == OPT_OUT_MARKER:
            return True
    return False


def _iter_test_nodes(tree: ast.Module) -> Iterable[tuple[str, bool, bool]]:
    mod_m = bool(_module_suite_markers(tree))
    for stmt in tree.body:
        if _is_test_function(stmt):
            yield (
                stmt.name,
                mod_m or bool(_decorator_suite_markers(stmt)),
                _has_opt_out(stmt),
            )
        elif _is_test_class(stmt):
            cls_m = mod_m or bool(_decorator_suite_markers(stmt))
            cls_opt = _has_opt_out(stmt)
            for item in stmt.body:
                if _is_test_function(item):
                    yield (
                        f"{stmt.name}::{item.name}",
                        cls_m or bool(_decorator_suite_markers(item)),
                        cls_opt or _has_opt_out(item),
                    )


def audit_suite_markers(
    tests_root: pathlib.Path, baseline_path: pathlib.Path | None = None
) -> dict[str, Any]:
    base_file = BASELINE_PATH if baseline_path is None else baseline_path
    base_entries = _load_baseline(base_file) if base_file.exists() else []
    b_u = sum(1 for n in base_entries if infer_directory_suite_marker(n) == "unit")
    b_i = sum(
        1 for n in base_entries if infer_directory_suite_marker(n) == "integration"
    )

    total, explicit, a_u, a_i = 0, 0, 0, 0
    for path in sorted(iter_test_files(tests_root)):
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="replace"),
                filename=str(path),
            )
        except SyntaxError:
            continue
        p_str = path.as_posix()
        for name, marked, opt_out in _iter_test_nodes(tree):
            total += 1
            if marked:
                explicit += 1
            elif not opt_out:
                inferred = infer_directory_suite_marker(f"{p_str}::{name}")
                if inferred == "unit":
                    a_u += 1
                elif inferred == "integration":
                    a_i += 1

    after_marked = explicit + a_u + a_i
    before_sum = summarize_gate_selectability(total, explicit)
    after_sum = summarize_gate_selectability(total, after_marked)
    target_met = bool(float(after_sum["selected_pct"]) >= SELECTABILITY_TARGET_PCT)

    return {
        "baseline": {
            "total": len(base_entries),
            "unit": b_u,
            "integration": b_i,
            "covered_by_auto_marking": b_u + b_i,
            "remaining_unmarked": len(base_entries) - (b_u + b_i),
        },
        "census": {
            "total_tests": total,
            "explicitly_marked": explicit,
            "auto_marked_unit": a_u,
            "auto_marked_integration": a_i,
            "after_marked": after_marked,
            "before": before_sum,
            "after": after_sum,
            "target_met": target_met,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=pathlib.Path, default=REPO_ROOT / "tests")
    parser.add_argument("--baseline", type=pathlib.Path, default=BASELINE_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    res = audit_suite_markers(args.tests_root, args.baseline)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    b, c = res["baseline"], res["census"]
    print(
        f"=== Suite Marker Reach Audit (#14947) ===\nBaseline: {b['total']} unmarked ({b['unit']} unit, {b['integration']} integration)"
    )
    print(
        f"Covered by auto-marking: {b['covered_by_auto_marking']} | Remaining unmarked: {b['remaining_unmarked']}"
    )
    print(
        f"Test Tree Census: {c['total_tests']} total | {c['explicitly_marked']} marked | {c['auto_marked_unit'] + c['auto_marked_integration']} auto-marked"
    )
    print(
        f"Before auto-marking: {c['before']['summary_line']}\nAfter auto-marking:  {c['after']['summary_line']}\nTarget (>={SELECTABILITY_TARGET_PCT:.1f}%): {'MET' if c['target_met'] else 'NOT MET'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
