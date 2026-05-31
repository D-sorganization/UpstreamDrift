"""Generate the CC-11 cross-engine differential-testing report.

The generator accepts normalized CC-11 JSON and the CC-7 harness shape
introduced by the draft conformance harness. It emits a stable machine-readable
artifact and a human-readable Markdown report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SCHEMA = "upstreamdrift.cross_engine_differential.v1"
DEFAULT_JSON = Path("docs/validation/cross_engine_v1.json")
DEFAULT_MARKDOWN = Path("docs/validation/cross_engine_v1.md")


@dataclass(frozen=True)
class Comparison:
    """One quantified cross-engine comparison row."""

    check_name: str
    phase: str
    metric_name: str
    engine1: str
    engine2: str
    dof: str
    max_deviation: float
    tolerance: float
    unit: str
    passed: bool
    severity: str
    rms_deviation_pct: float | None = None
    peak_reference: float | None = None
    divergence_id: str | None = None
    rationale: str | None = None

    @property
    def tolerance_ratio(self) -> float:
        """Return max deviation divided by tolerance."""
        if self.tolerance == 0:
            return float("inf")
        return self.max_deviation / self.tolerance


def generate_report(
    input_path: Path | None = None,
    json_path: Path = DEFAULT_JSON,
    markdown_path: Path = DEFAULT_MARKDOWN,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Generate machine-readable and Markdown report artifacts."""
    source = _load_source(input_path)
    normalized = _normalize(source, generated_at=generated_at)
    _write_json(json_path, normalized)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(normalized), encoding="utf-8")
    return normalized


def _load_source(input_path: Path | None) -> dict[str, Any]:
    if input_path is None:
        return _default_pending_source()
    return json.loads(input_path.read_text(encoding="utf-8"))


def _normalize(source: dict[str, Any], *, generated_at: str | None) -> dict[str, Any]:
    generated = (
        generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    comparisons = _extract_comparisons(source)
    status = str(source.get("status") or _status_from_comparisons(comparisons))
    dependencies = list(source.get("dependencies", _default_dependencies()))
    engines = list(source.get("engines", _engines_from_comparisons(comparisons)))

    return {
        "schema": SCHEMA,
        "generated_at": generated,
        "status": status,
        "source_shape": _source_shape(source),
        "engines": engines,
        "dependencies": dependencies,
        "summary": _summary(comparisons),
        "comparisons": [_comparison_dict(row) for row in comparisons],
    }


def _extract_comparisons(source: dict[str, Any]) -> list[Comparison]:
    if "comparisons" in source:
        return [_comparison_from_normalized(row) for row in source["comparisons"]]
    if "checks" in source:
        return [_comparison_from_cc7(row) for row in source["checks"]]
    if "results" in source:
        return [_comparison_from_cc7(row) for row in source["results"]]
    return []


def _comparison_from_normalized(row: dict[str, Any]) -> Comparison:
    return Comparison(
        check_name=str(row["check_name"]),
        phase=str(row.get("phase", "unspecified")),
        metric_name=str(row["metric_name"]),
        engine1=str(row["engine1"]),
        engine2=str(row["engine2"]),
        dof=str(row.get("dof", "aggregate")),
        max_deviation=float(row["max_deviation"]),
        tolerance=float(row["tolerance"]),
        unit=str(row.get("unit", "absolute")),
        passed=bool(row["passed"]),
        severity=str(row.get("severity", "PASSED" if row["passed"] else "ERROR")),
        rms_deviation_pct=_optional_float(row.get("rms_deviation_pct")),
        peak_reference=_optional_float(row.get("peak_reference")),
        divergence_id=_optional_str(row.get("divergence_id")),
        rationale=_optional_str(row.get("rationale")),
    )


def _comparison_from_cc7(row: dict[str, Any]) -> Comparison:
    validation = row.get("validation") or {}
    divergence = row.get("divergence") or {}
    metric_name = str(validation.get("metric_name", row.get("metric_name", "unknown")))
    return Comparison(
        check_name=str(row.get("check_name", "unknown_check")),
        phase=str(row.get("phase", _phase_for_check(str(row.get("check_name", ""))))),
        metric_name=metric_name,
        engine1=str(validation.get("engine1", row.get("engine_name", "unknown"))),
        engine2=str(validation.get("engine2", row.get("reference", "reference"))),
        dof=str(row.get("dof", row.get("joint", "aggregate"))),
        max_deviation=float(validation.get("max_deviation", 0.0)),
        tolerance=float(divergence.get("tolerance", validation.get("tolerance", 0.0))),
        unit=str(row.get("unit", _unit_for_metric(metric_name))),
        passed=bool(row.get("passed", False)),
        severity=str(
            validation.get("severity", "PASSED" if row.get("passed") else "ERROR")
        ),
        rms_deviation_pct=_optional_float(row.get("rms_deviation_pct")),
        peak_reference=_optional_float(row.get("peak_reference")),
        divergence_id=_optional_str(divergence.get("id")),
        rationale=_optional_str(divergence.get("rationale") or row.get("message")),
    )


def _summary(comparisons: list[Comparison]) -> dict[str, Any]:
    failed = [row for row in comparisons if not row.passed]
    registered = [row for row in comparisons if row.divergence_id]
    contact_free = [row for row in comparisons if row.phase == "contact_free"]
    torque = [row for row in contact_free if row.metric_name.startswith("torque")]
    return {
        "comparison_count": len(comparisons),
        "passed_count": len(comparisons) - len(failed),
        "failed_count": len(failed),
        "registered_divergence_count": len(registered),
        "contact_free_torque_max_pct_of_peak": _max_optional(
            row.rms_deviation_pct for row in torque
        ),
        "worst_tolerance_ratio": max(
            (row.tolerance_ratio for row in comparisons), default=0.0
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Cross-engine differential-testing report v1",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Source shape: `{report['source_shape']}`",
        f"- Engines: {', '.join(report['engines']) or 'none'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| comparisons | {summary['comparison_count']} |",
        f"| passed | {summary['passed_count']} |",
        f"| failed | {summary['failed_count']} |",
        f"| registered divergences | {summary['registered_divergence_count']} |",
        "| contact-free torque max RMS % of peak | "
        f"{_format_optional(summary['contact_free_torque_max_pct_of_peak'])} |",
        f"| worst tolerance ratio | {summary['worst_tolerance_ratio']:.3g} |",
        "",
        "## Dependency Status",
        "",
    ]
    lines.extend(_dependency_lines(report["dependencies"]))
    lines.extend(["", "## Comparison Rows", ""])
    if report["comparisons"]:
        lines.extend(_comparison_table(report["comparisons"]))
    else:
        lines.append(
            "No live adapter comparison rows are claimed yet. Current `origin/main` "
            "does not include the CC-7 harness or both canonical-v2 adapters."
        )
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```powershell",
            "python scripts\\validation\\cross_engine_differential_report.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _dependency_lines(dependencies: list[dict[str, Any]]) -> list[str]:
    if not dependencies:
        return ["No blocking dependencies recorded."]
    return [
        f"- {item['name']}: {item['status']} ({item['url']})" for item in dependencies
    ]


def _comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Check | Phase | DOF | Engines | Metric | Deviation | Tolerance | Result | Divergence |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        result = "pass" if row["passed"] else "fail"
        divergence = row["divergence_id"] or ""
        engines = f"{row['engine1']} vs {row['engine2']}"
        lines.append(
            f"| {row['check_name']} | {row['phase']} | {row['dof']} | {engines} | "
            f"{row['metric_name']} | {row['max_deviation']:.6g} {row['unit']} | "
            f"{row['tolerance']:.6g} | {result} | {divergence} |"
        )
    return lines


def _comparison_dict(row: Comparison) -> dict[str, Any]:
    return {
        "check_name": row.check_name,
        "phase": row.phase,
        "metric_name": row.metric_name,
        "engine1": row.engine1,
        "engine2": row.engine2,
        "dof": row.dof,
        "max_deviation": row.max_deviation,
        "tolerance": row.tolerance,
        "tolerance_ratio": row.tolerance_ratio,
        "unit": row.unit,
        "passed": row.passed,
        "severity": row.severity,
        "rms_deviation_pct": row.rms_deviation_pct,
        "peak_reference": row.peak_reference,
        "divergence_id": row.divergence_id,
        "rationale": row.rationale,
    }


def _default_pending_source() -> dict[str, Any]:
    return {
        "status": "blocked_by_draft_dependencies",
        "engines": ["mujoco-canonical-v2", "pinocchio-canonical-v2"],
        "dependencies": _default_dependencies(),
        "comparisons": [],
    }


def _default_dependencies() -> list[dict[str, str]]:
    return [
        {
            "name": "CC-7 conformance harness",
            "status": "draft PR #6826",
            "url": "https://github.com/D-sorganization/UpstreamDrift/pull/6826",
        },
        {
            "name": "CC-9 Pinocchio canonical-v2 adapter",
            "status": "draft PR #6828",
            "url": "https://github.com/D-sorganization/UpstreamDrift/pull/6828",
        },
        {
            "name": "CC-10 MuJoCo canonical-v2 adapter",
            "status": "draft PR #6829",
            "url": "https://github.com/D-sorganization/UpstreamDrift/pull/6829",
        },
    ]


def _status_from_comparisons(comparisons: list[Comparison]) -> str:
    if not comparisons:
        return "no_comparisons"
    if all(row.passed for row in comparisons):
        return "passed"
    return "failed"


def _source_shape(source: dict[str, Any]) -> str:
    if "comparisons" in source:
        return "cc11"
    if "checks" in source or "results" in source:
        return "cc7"
    return "pending"


def _engines_from_comparisons(comparisons: list[Comparison]) -> list[str]:
    engines = {row.engine1 for row in comparisons} | {
        row.engine2 for row in comparisons
    }
    return sorted(engine for engine in engines if engine != "reference")


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _phase_for_check(check_name: str) -> str:
    if "contact" in check_name:
        return "contact"
    if "dynamics" in check_name:
        return "contact_free"
    return "unspecified"


def _unit_for_metric(metric_name: str) -> str:
    if metric_name == "position":
        return "m"
    if metric_name == "acceleration":
        return "rad/s^2"
    if metric_name.startswith("torque"):
        return "percent_peak_torque"
    return "absolute"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _max_optional(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generate_report(
        input_path=args.input,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
