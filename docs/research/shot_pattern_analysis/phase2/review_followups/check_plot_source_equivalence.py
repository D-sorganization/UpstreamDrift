"""Verify isolated plot formatting and explicit float64-edge equivalence."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_BASELINE_COMMIT = "4fc22c694404217208534b78308d31ea8985e7d3"
_HISTORICAL_CANDIDATE_COMMIT = "ecc82796cec7e9119e2dd2fb0e8bb2582851761f"
_HISTORICAL_CANDIDATE_SHA256 = {
    "density.py": "d444779f6ba6179e0fd87348c9b60f901a853e4558afa4cae3e732ff675640ad",
    "academic_plots.py": "246b891faaa6e3512ecb9685d6cc6a84e2bea3ccadd1f0030a7afa5671978944",
    "matched_report_plots.py": "72515aef38a1c72b740f0c70a770a4c7db1124a62b4514bb424af7124d6dd8fc",
}
_DENSITY_SOURCE_SHA256 = {
    "_density_original": "127d1d249203efbddb3152e668e270dc54be5857e97377a344edc9d543c2d3ce",
    "_density_revised": (
        _HISTORICAL_CANDIDATE_SHA256["density.py"],
        "6dcef50caf4ce6dfb3244890b1fe54615a5e50936b1b56080c471e01113c5ef1",
    ),
}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _ExplicitFloat64(ast.NodeTransformer):
    """Normalize only the documented default-dtype spelling change."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "np.linspace"
        ):
            node.keywords = [
                keyword
                for keyword in node.keywords
                if not (
                    keyword.arg == "dtype"
                    and ast.unparse(keyword.value) == "np.float64"
                )
            ]
        return node


def _module(name: str, source: bytes) -> types.ModuleType:
    expected_sha256 = _DENSITY_SOURCE_SHA256.get(name)
    allowed_digests = (
        (expected_sha256,) if isinstance(expected_sha256, str) else expected_sha256
    )
    if allowed_digests is None or _sha(source) not in allowed_digests:
        raise ValueError(f"Unapproved density source digest: {name}")
    module = types.ModuleType(name)
    sys.modules[name] = module
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec(compile(source, name, "exec"), module.__dict__)  # nosec B102
    return module


def _equal(first: object, second: object) -> bool:
    if isinstance(first, np.ndarray):
        return (
            isinstance(second, np.ndarray)
            and first.dtype == second.dtype
            and np.array_equal(first, second)
        )
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_equal(value, second[key]) for key, value in first.items())
        )
    return bool(first == second)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline != _BASELINE_COMMIT:
        raise ValueError("Unapproved plot baseline commit")
    if args.candidate is not None and args.candidate != _HISTORICAL_CANDIDATE_COMMIT:
        raise ValueError("Unapproved historical candidate commit")
    files = {}
    for name in ("density.py", "academic_plots.py", "matched_report_plots.py"):
        path = Path("src/tools/shot_pattern_analysis") / name
        before = subprocess.check_output(
            ["git", "show", f"{_BASELINE_COMMIT}:{path}"], cwd=_ROOT
        )
        after = (
            subprocess.check_output(
                ["git", "show", f"{_HISTORICAL_CANDIDATE_COMMIT}:{path}"],
                cwd=_ROOT,
            )
            if args.candidate is not None
            else (_ROOT / path).read_bytes()
        )
        if args.candidate is not None:
            if _sha(after) != _HISTORICAL_CANDIDATE_SHA256[name]:
                raise ValueError(f"Unapproved historical candidate digest: {path}")
        elif (
            name == "density.py"
            and _sha(after) != _DENSITY_SOURCE_SHA256["_density_revised"][1]
        ):
            raise ValueError(f"Unapproved current density source digest: {path}")
        old_tree, new_tree = ast.parse(before), ast.parse(after)
        exact_ast = ast.dump(old_tree) == ast.dump(new_tree)
        normalized = ast.dump(_ExplicitFloat64().visit(old_tree)) == ast.dump(
            _ExplicitFloat64().visit(new_tree)
        )
        if not normalized or (name != "density.py" and not exact_ast):
            raise ValueError(f"Unexpected semantic source change: {path}")
        files[str(path)] = {
            "before_sha256": _sha(before),
            "after_sha256": _sha(after),
            "exact_ast_equal": exact_ast,
            "normalized_ast_equal": normalized,
        }
        if name == "density.py":
            original = _module("_density_original", before)
            revised = _module("_density_revised", after)
    random = np.random.default_rng(20261009)
    fixtures = [
        {"Straight": np.zeros((12, 2))},
        {"Straight": np.column_stack((np.arange(25), np.zeros(25)))},
        {"Straight": np.array([[1.0, 2.0]])},
    ]
    fixtures.extend(
        {
            name: random.normal(size=(37 + index, 2)) * (index + 1) + offset
            for name, offset in (("Straight", 0), ("Draw", -5), ("Fade", 5))
        }
        for index in range(25)
    )
    checks = 0
    for fixture in fixtures:
        for bins in (8, (13, 19), 64):
            first = original.shared_histogram_density(fixture, bins=bins)
            second = revised.shared_histogram_density(fixture, bins=bins)
            if not _equal(asdict(first), asdict(second)):
                raise ValueError("Density result or dtype differs from baseline")
            checks += 1
    args.output.write_text(
        json.dumps(
            {
                "schema": "plot-source-equivalence/1",
                "baseline_commit": args.baseline,
                "candidate_commit": args.candidate,
                "files": files,
                "density_comparisons_exact": checks,
                "numpy_version": np.__version__,
                "script_sha256": _sha(Path(__file__).read_bytes()),
                "scope": (
                    "Historical source parity only; no current qualification renewed"
                    if args.candidate is not None
                    else "Current source parity only; no original execution snapshot rewritten"
                ),
                "pass": True,
            },
            indent=2,
        )
        + "\n"
    )
    sys.stdout.write(
        f"Verified {checks} exact density comparisons and three source projections\n"
    )


if __name__ == "__main__":
    main()
