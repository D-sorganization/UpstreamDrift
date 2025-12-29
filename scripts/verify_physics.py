"""
Advanced Physics Verification Runner for Golf Modeling Suite.

This script performs a comprehensive check of all physics engines,
runs validation tests, and generates a detailed compliance report.
"""

import datetime
import sys
from pathlib import Path

import pytest

# Add root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from shared.python.engine_manager import EngineManager  # noqa: E402


def run_verification() -> None:
    """Run the physics verification suite."""
    report_lines = []
    report_lines.append("# Physics Verification Report")
    report_lines.append(
        f"**Date:** {datetime.datetime.now(tz=datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append("")

    print("Starting Physics Verification...")

    # 1. Engine Diagnostics
    manager = EngineManager(ROOT_DIR)
    probes = manager.probe_all_engines()

    report_lines.append("## 1. Engine Status")
    report_lines.append("| Engine | Status | Version | Notes |")
    report_lines.append("|---|---|---|---|")

    print("-" * 60)
    print(f"{'Engine':<15} | {'Status':<15} | {'Version':<10}")
    print("-" * 60)

    for engine, result in probes.items():
        status_str = result.status.value
        version = result.version or "N/A"
        note = result.diagnostic_message

        print(f"{engine.value:<15} | {status_str:<15} | {version:<10}")
        report_lines.append(f"| {engine.value} | {status_str} | {version} | {note} |")

    print("-" * 60)
    report_lines.append("")

    # 2. Run Tests
    report_lines.append("## 2. Validation Test Results")

    print("\nRunning Pytest Suite...")

    class Plugin:
        def __init__(self) -> None:
            self.results = []

        def pytest_runtest_logreport(self, report) -> None:
            if report.when == "call":
                self.results.append(
                    {
                        "nodeid": report.nodeid,
                        "outcome": report.outcome,
                        "duration": report.duration,
                    }
                )
            elif report.when == "setup" and report.outcome == "skipped":
                self.results.append(
                    {
                        "nodeid": report.nodeid,
                        "outcome": "skipped",
                        "duration": 0.0,
                    }
                )

    plugin = Plugin()
    test_dir = ROOT_DIR / "tests" / "physics_validation"

    # Run pytest
    _ret_code = pytest.main(["-v", str(test_dir)], plugins=[plugin])

    report_lines.append("| Test Case | Outcome | Duration (s) |")
    report_lines.append("|---|---|---|")

    print("\nTest Results:")
    for res in plugin.results:
        outcome = res["outcome"].upper()
        test_name = res["nodeid"].split("::")[-1]

        print(f"[{outcome}] {test_name} ({res['duration']:.4f}s)")
        report_lines.append(f"| {test_name} | {outcome} | {res['duration']:.4f} |")

    report_lines.append("")

    # 3. Recommendations
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
                f"- **{s['nodeid']}** was skipped. Check engine availability."
            )

    if not failed and not skipped:
        report_lines.append("### ✅ All Systems Valid")
        report_lines.append(
            "All physics engines are producing valid, energy-conserving results."
        )

    # Write Report
    report_path = ROOT_DIR / "output" / "PHYSICS_VERIFICATION_REPORT.md"
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nReport generated at: {report_path}")


if __name__ == "__main__":
    run_verification()
