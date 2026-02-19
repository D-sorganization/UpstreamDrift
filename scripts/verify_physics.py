"""
Advanced Physics Verification Runner for Golf Modeling Suite.

This script performs a comprehensive check of all physics engines,
runs validation tests, and generates a detailed compliance report.
"""

# Python 3.10 compatibility: timezone.utc was added in 3.11
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

from src.shared.python.data_io.path_utils import get_src_root  # noqa: E402

# Add root to path
ROOT_DIR = get_src_root()

from src.shared.python.engine_core.engine_manager import EngineManager  # noqa: E402


class _PytestPlugin:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        """Collect test results from each pytest report hook invocation."""
        if report.when == "call":
            self.results.append(
                {
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "duration": report.duration,
                },
            )
        elif report.when == "setup" and report.outcome == "skipped":
            self.results.append(
                {
                    "nodeid": report.nodeid,
                    "outcome": "skipped",
                    "duration": 0.0,
                },
            )


def _build_report_header() -> list[str]:
    return [
        "# Physics Verification Report",
        f"**Date:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]


def _run_engine_diagnostics(report_lines: list[str]) -> None:
    manager = EngineManager(ROOT_DIR)
    probes = manager.probe_all_engines()

    report_lines.append("## 1. Engine Status")
    report_lines.append("| Engine | Status | Version | Notes |")
    report_lines.append("|---|---|---|---|")

    logger.info("-" * 60)
    logger.info("%-15s | %-15s | %-10s", "Engine", "Status", "Version")
    logger.info("-" * 60)

    for engine, result in probes.items():
        status_str = result.status.value
        version = result.version or "N/A"
        note = result.diagnostic_message

        logger.info("%-15s | %-15s | %-10s", engine.value, status_str, version)
        report_lines.append(f"| {engine.value} | {status_str} | {version} | {note} |")

    logger.info("-" * 60)
    report_lines.append("")


def _run_tests_and_report(report_lines: list[str]) -> _PytestPlugin:
    report_lines.append("## 2. Validation Test Results")
    logger.info("Running Pytest Suite...")

    plugin = _PytestPlugin()
    test_dir = ROOT_DIR / "tests" / "physics_validation"
    _ret_code = pytest.main(["-v", str(test_dir)], plugins=[plugin])

    report_lines.append("| Test Case | Outcome | Duration (s) |")
    report_lines.append("|---|---|---|")

    logger.info("Test Results:")
    for res in plugin.results:
        outcome = res["outcome"].upper()
        test_name = res["nodeid"].split("::")[-1]

        logger.info("[%s] %s (%.4fs)", outcome, test_name, res["duration"])
        report_lines.append(f"| {test_name} | {outcome} | {res['duration']:.4f} |")

    report_lines.append("")
    return plugin


def _build_recommendations(
    report_lines: list[str],
    plugin: _PytestPlugin,
) -> None:
    report_lines.append("## 3. Analysis & Recommendations")

    failed = [r for r in plugin.results if r["outcome"] == "failed"]
    skipped = [r for r in plugin.results if r["outcome"] == "skipped"]

    if failed:
        report_lines.append("### ❌ Failures Detected")
        for f in failed:
            report_lines.append(f"- **{f['nodeid']}** failed. Check logs for details.")

    if skipped:
        report_lines.append("### ⚠️ Skipped Tests")
        for s in skipped:
            report_lines.append(
                f"- **{s['nodeid']}** was skipped. Check engine availability.",
            )

    if not failed and not skipped:
        report_lines.append("### ✅ All Systems Valid")
        report_lines.append(
            "All physics engines are producing valid, energy-conserving results.",
        )


def run_verification() -> None:
    """Run the physics verification suite."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Starting Physics Verification...")

    report_lines = _build_report_header()
    _run_engine_diagnostics(report_lines)
    plugin = _run_tests_and_report(report_lines)
    _build_recommendations(report_lines, plugin)

    report_path = ROOT_DIR / "output" / "PHYSICS_VERIFICATION_REPORT.md"
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Report generated at: %s", report_path)


if __name__ == "__main__":
    run_verification()
