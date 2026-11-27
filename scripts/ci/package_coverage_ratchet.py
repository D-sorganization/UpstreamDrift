#!/usr/bin/env python3
"""Per-package coverage aggregation and ratchet evaluation (#14946).

Evaluates package-level coverage metrics against accountable gates configured
in scripts/config/mypy_exclusion_budget.json.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

DEFAULT_BUDGET = Path("scripts/config/mypy_exclusion_budget.json")


@dataclass(frozen=True)
class PackageCoverageSummary:
    """Aggregated coverage metrics for a single package or prefix."""

    name: str
    path: str
    covered_lines: int
    num_statements: int
    percent_covered: float
    matching_files: int


@dataclass(frozen=True)
class RatchetResult:
    """Evaluation result for an individual package ratchet gate."""

    name: str
    path: str
    covered_lines: int
    num_statements: int
    percent_covered: float
    ratchet_floor: float
    passed: bool
    reason: str
    matching_files: int


@dataclass(frozen=True)
class RatchetReport:
    """Overall outcome of per-package coverage ratchet evaluation."""

    passed: bool
    results: list[RatchetResult]
    total_covered: int
    total_statements: int
    total_percent: float
    freshness_passed: bool = True
    freshness_message: str = ""


def _normalize_path(path_str: str) -> str:
    return path_str.replace("\\", "/").removeprefix("./")


def _sum_coverage_for_path(files: dict[str, Any], prefix: str) -> tuple[int, int, int]:
    cov, stmts, matched = 0, 0, 0
    norm_pfx = _normalize_path(prefix)
    for p, info in files.items():
        if _normalize_path(p).startswith(norm_pfx):
            matched += 1
            s = info.get("summary", {}) if isinstance(info, dict) else {}
            cov += int(s.get("covered_lines", 0))
            stmts += int(s.get("num_statements", 0))
    return cov, stmts, matched


def compute_package_coverage(
    coverage_data: dict[str, Any],
    package_gates: Sequence[dict[str, Any]] | None = None,
) -> dict[str, PackageCoverageSummary]:
    if not isinstance(coverage_data, dict) or "files" not in coverage_data:
        raise ValueError("coverage_data must contain 'files' dictionary")
    files = coverage_data.get("files", {})
    if package_gates is not None:
        gates = list(package_gates)
    else:
        cfg = (
            json.loads(DEFAULT_BUDGET.read_text("utf-8"))
            if DEFAULT_BUDGET.is_file()
            else {}
        )
        gates = cfg.get("coverage_gates", [])

    if gates:
        sums: dict[str, PackageCoverageSummary] = {}
        for g in gates:
            n = str(g.get("name", "")).strip()
            p = _normalize_path(str(g.get("path", ""))).strip()
            if n and p:
                c, s, m = _sum_coverage_for_path(files, p)
                sums[n] = PackageCoverageSummary(
                    n, p, c, s, 100.0 * c / s if s > 0 else 0.0, m
                )
        return sums

    sums = {}
    for p, info in files.items():
        np = _normalize_path(str(p))
        pts = np.split("/")
        k = (
            pts[3]
            if np.startswith("src/shared/python/") and len(pts) >= 4
            else (pts[1] if len(pts) >= 2 else np)
        )
        s = info.get("summary", {}) if isinstance(info, dict) else {}
        cov, stmts = int(s.get("covered_lines", 0)), int(s.get("num_statements", 0))
        cur = sums.get(k)
        c = (cur.covered_lines if cur else 0) + cov
        sm = (cur.num_statements if cur else 0) + stmts
        m = (cur.matching_files if cur else 0) + 1
        pct = 100.0 * c / sm if sm > 0 else 0.0
        sums[k] = PackageCoverageSummary(k, k, c, sm, pct, m)
    return sums


def evaluate_package_coverage_ratchet(
    coverage_data: dict[str, Any], budget_data: dict[str, Any]
) -> RatchetReport:
    if not isinstance(budget_data, dict) or "coverage_gates" not in budget_data:
        raise ValueError("budget_data must contain a 'coverage_gates' list")
    files = coverage_data.get("files", {}) if isinstance(coverage_data, dict) else {}
    results, tot_cov, tot_stmts = [], 0, 0
    for g in budget_data.get("coverage_gates", []):
        if not isinstance(g, dict):
            continue
        name = str(g.get("name", "")).strip()
        path = _normalize_path(str(g.get("path", ""))).strip()
        val = g.get("min_coverage")
        floor = float(val if val is not None else g.get("floor_percent", 0.0))
        cov, stmts, matched = _sum_coverage_for_path(files, path)
        tot_cov += cov
        tot_stmts += stmts
        if matched == 0 or stmts == 0:
            passed, pct = False, 0.0
            item = "zero files" if matched == 0 else "0 statements"
            reason = f"Gate '{name}' matches {item} in coverage report"
        else:
            pct = 100.0 * cov / stmts
            passed = pct >= floor
            state = "meets" if passed else "dropped below"
            reason = f"Coverage {pct:.1f}% {state} floor {floor:.1f}%"
        results.append(
            RatchetResult(name, path, cov, stmts, pct, floor, passed, reason, matched)
        )
    return RatchetReport(
        bool(results) and all(r.passed for r in results),
        results,
        tot_cov,
        tot_stmts,
        100.0 * tot_cov / tot_stmts if tot_stmts > 0 else 0.0,
    )


def ratchet_from_measured(
    coverage_data: dict[str, Any],
    budget_data: dict[str, Any],
    *,
    ratchet_up_only: bool = True,
) -> dict[str, float]:
    files = coverage_data.get("files", {}) if isinstance(coverage_data, dict) else {}
    ratchets: dict[str, float] = {}
    for g in budget_data.get("coverage_gates", []):
        if not isinstance(g, dict):
            continue
        name = str(g.get("name", "")).strip()
        path = _normalize_path(str(g.get("path", ""))).strip()
        val = g.get("min_coverage")
        floor = float(val if val is not None else g.get("floor_percent", 0.0))
        cov, stmts, _ = _sum_coverage_for_path(files, path)
        m = round(math.floor(1000.0 * cov / stmts) / 10.0, 1) if stmts > 0 else 0.0
        ratchets[name] = max(floor, m) if ratchet_up_only else m
    return ratchets


def check_coverage_freshness(
    coverage_data: dict[str, Any],
    *,
    max_age_days: float = 7.0,
    now: datetime | None = None,
) -> tuple[bool, str]:
    meta = coverage_data.get("meta") if isinstance(coverage_data, dict) else None
    if not isinstance(meta, dict) or "timestamp" not in meta:
        return False, "Missing meta.timestamp in coverage report"
    try:
        raw = meta["timestamp"]
        ts = (
            datetime.fromtimestamp(raw, tz=timezone.utc)
            if isinstance(raw, (int, float))
            else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        )
    except (ValueError, OSError) as exc:
        return False, f"Invalid meta.timestamp format: {exc}"
    ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    cur = now or datetime.now(timezone.utc)
    cur = cur.replace(tzinfo=timezone.utc) if cur.tzinfo is None else cur
    age = (cur - ts).total_seconds() / 86400.0
    if age < -0.1:
        return False, f"meta.timestamp is in future ({age:.1f}d)"
    if age > max_age_days:
        return False, f"Coverage report is stale: {age:.1f}d old (> {max_age_days}d)"
    return True, f"Coverage report is fresh: {age:.1f}d old (<= {max_age_days}d)"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enforce coverage ratchet")
    p.add_argument("--coverage-json", default="coverage.json")
    p.add_argument("--budget-json", default="scripts/config/mypy_exclusion_budget.json")
    p.add_argument("--write-ratchet", action="store_true")
    p.add_argument("--check-freshness", action="store_true")
    p.add_argument("--max-age-days", type=float, default=7.0)
    p.add_argument("--summary-out")
    p.add_argument("--ratchet-up", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    c_p, b_p = Path(args.coverage_json), Path(args.budget_json)
    if not c_p.is_file() or not b_p.is_file():
        print(f"File missing: {c_p} or {b_p}", file=sys.stderr)
        return 1
    try:
        cov = json.loads(c_p.read_text(encoding="utf-8"))
        b = json.loads(b_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"JSON decode error: {exc}", file=sys.stderr)
        return 1
    if args.check_freshness:
        fresh, msg = check_coverage_freshness(cov, max_age_days=args.max_age_days)
        if not fresh:
            print(f"Freshness failed: {msg}", file=sys.stderr)
            return 1
    if args.write_ratchet:
        up = ratchet_from_measured(cov, b, ratchet_up_only=args.ratchet_up)
        for g in b.get("coverage_gates", []):
            if g.get("name") in up:
                g["min_coverage"] = up[g["name"]]
        b_p.write_text(json.dumps(b, indent=2) + "\n", encoding="utf-8")
    rep = evaluate_package_coverage_ratchet(cov, b)
    for r in rep.results:
        tag = "PASS" if r.passed else "FAIL"
        print(
            f"[{tag}] {r.name:<25} {r.percent_covered:5.1f}% (floor: {r.ratchet_floor:5.1f}%) - {r.reason}"
        )
    print(
        f"Overall: {rep.total_percent:.1f}% ({rep.total_covered}/{rep.total_statements}), Passed: {rep.passed}"
    )
    if args.summary_out:
        rows = "\n".join(
            f"- {r.name}: {r.percent_covered:.1f}% / {r.ratchet_floor:.1f}% ({'PASS' if r.passed else 'FAIL'})"
            for r in rep.results
        )
        text = f"## Coverage Ratchet Report\n- Overall: {rep.total_percent:.1f}% | Passed: {rep.passed}\n\n{rows}\n"
        Path(args.summary_out).write_text(text, encoding="utf-8")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
