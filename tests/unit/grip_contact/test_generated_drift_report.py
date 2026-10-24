"""``--check`` drift diagnostics for the Simscape grip generated files (#13544).

The kernel/params/golden and scenario exporters return ``1`` when a committed
generated file drifts. In merge-group CI that bare ``1`` was the only evidence
of an intermittent drift, so the check now also reports *which* file drifted
and a bounded unified diff as a WARNING log record (pytest shows it under
"Captured log call" when a test fails). These tests pin that report; they never
relax the check itself (a drifted file still returns ``1``).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from src.shared.python.grip_contact.generated_export_lock import GENERATED_EXPORT_LOCK
from src.shared.python.grip_contact.matlab_kernel import export_matlab_kernel_bundle
from src.shared.python.grip_contact.net_wrench_parity import report_generated_drift
from src.shared.python.grip_contact.simscape_grip_contact import export_scenarios_json


@pytest.mark.unit
def test_report_generated_drift_names_file_and_bounds_diff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = "".join(f"line {i}\n" for i in range(500))
    current = "".join(
        f"LINE {i}\n" if i % 10 == 0 else f"line {i}\n" for i in range(500)
    )
    report_generated_drift(Path("gen/out.json"), current, expected, max_lines=20)
    err = caplog.text
    assert "gen/out.json" in err or "gen\\out.json" in err
    assert "-LINE 0" in err
    assert "+line 0" in err
    assert "truncated" in err
    # 20 diff lines plus a header line and a truncation line, never the whole file.
    assert len(err.splitlines()) <= 24
    assert "LINE 499" not in err


@pytest.mark.unit
def test_committed_export_paths_share_process_lock(tmp_path: Path) -> None:
    """#13544: exporters serialize when targets include committed generated paths."""
    from src.shared.python.grip_contact.generated_export_lock import (
        targets_touch_committed,
    )
    from src.shared.python.grip_contact.matlab_kernel import (
        DEFAULT_GOLDEN_JSON_PATH,
        DEFAULT_KERNEL_M_PATH,
        DEFAULT_PARAMS_JSON_PATH,
    )

    committed = (
        DEFAULT_KERNEL_M_PATH,
        DEFAULT_PARAMS_JSON_PATH,
        DEFAULT_GOLDEN_JSON_PATH,
    )
    assert targets_touch_committed(committed, committed)
    local = tmp_path / "k.m", tmp_path / "p.json", tmp_path / "g.json"
    assert not targets_touch_committed(local, committed)

    paths = {
        "kernel_m_path": local[0],
        "params_json_path": local[1],
        "golden_json_path": local[2],
    }
    assert export_matlab_kernel_bundle(**paths, check=False) == 0
    held = threading.Event()

    def hold_lock() -> None:
        with GENERATED_EXPORT_LOCK:
            held.set()
            time.sleep(0.05)

    worker = threading.Thread(target=hold_lock)
    worker.start()
    assert held.wait(1.0)
    start = time.monotonic()
    assert export_scenarios_json(tmp_path / "scenarios.json", check=False) == 0
    elapsed = time.monotonic() - start
    worker.join(timeout=1.0)
    assert elapsed >= 0.04


@pytest.mark.unit
def test_report_generated_drift_missing_file(caplog: pytest.LogCaptureFixture) -> None:
    report_generated_drift(Path("gen/missing.m"), None, "x\n")
    assert "missing" in caplog.text


@pytest.mark.unit
def test_kernel_bundle_check_reports_drifted_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = {
        "kernel_m_path": tmp_path / "k.m",
        "params_json_path": tmp_path / "p.json",
        "golden_json_path": tmp_path / "g.json",
    }
    assert export_matlab_kernel_bundle(**paths, check=False) == 0
    p_path = paths["params_json_path"]
    p_path.write_text(
        p_path.read_text(encoding="utf-8").replace(
            '"schema_version"', '"schema_versionX"'
        ),
        encoding="utf-8",
    )
    caplog.clear()
    assert export_matlab_kernel_bundle(**paths, check=True) == 1
    err = caplog.text
    assert "p.json" in err
    assert '-  "schema_versionX"' in err
    assert '+  "schema_version"' in err


@pytest.mark.unit
def test_scenarios_check_reports_drifted_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "scenarios.json"
    assert export_scenarios_json(out, check=False) == 0
    out.write_text(
        out.read_text(encoding="utf-8").replace(
            '"matlab_release"', '"matlab_releaseX"'
        ),
        encoding="utf-8",
    )
    caplog.clear()
    assert export_scenarios_json(out, check=True) == 1
    err = caplog.text
    assert "scenarios.json" in err
    assert '-  "matlab_releaseX"' in err
