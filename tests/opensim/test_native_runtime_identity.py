"""Clean-process runtime identity diagnostic for OpenSim and ezc3d (#12065).

The classification is tested on recorded probe results so it runs on any
host.  The real diagnostic is also run end to end: every probe executes in
its own fresh interpreter, so the pytest process never loads either native
library.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.diagnostics.native_runtime_identity import (
    PROBE_ORDERS,
    classify_runtimes,
    exit_status,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_STATUSES = {"ok", "not-installed", "import-error"}


def _step(module: str, status: str, error: str | None = None) -> dict[str, object]:
    step: dict[str, object] = {"module": module, "status": status}
    if status == "ok":
        step.update(version="1.0", file=f"/site/{module}/__init__.py")
    else:
        step.update(error_type="ImportError", error=error or "boom")
    return step


def _probes(**steps: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"order": list(order), "clean_start": True, "steps": steps["_".join(order)]}
        for order in PROBE_ORDERS
    ]


def test_probe_orders_cover_each_runtime_alone_and_both_mixed_orders() -> None:
    assert set(PROBE_ORDERS) == {
        ("ezc3d",),
        ("opensim",),
        ("opensim", "ezc3d"),
        ("ezc3d", "opensim"),
    }


def test_windows_dll_clash_is_reported_as_an_incompatible_mixed_process() -> None:
    clash = "DLL load failed while importing _ezc3d: procedure could not be found"
    probes = _probes(
        ezc3d=[_step("ezc3d", "ok")],
        opensim=[_step("opensim", "ok")],
        opensim_ezc3d=[_step("opensim", "ok"), _step("ezc3d", "import-error", clash)],
        ezc3d_opensim=[_step("ezc3d", "ok"), _step("opensim", "ok")],
    )
    identities = classify_runtimes(probes)
    assert identities["c3d-preparation"] == "supported"
    assert identities["native-opensim"] == "supported"
    assert identities["mixed-process"].startswith("incompatible")
    assert "opensim -> ezc3d" in identities["mixed-process"]
    assert exit_status(probes) == 0


def test_missing_runtime_makes_mixed_process_not_evaluable_with_its_reason() -> None:
    probes = _probes(
        ezc3d=[_step("ezc3d", "ok")],
        opensim=[_step("opensim", "not-installed", "No module named 'opensim'")],
        opensim_ezc3d=[_step("opensim", "not-installed"), _step("ezc3d", "ok")],
        ezc3d_opensim=[_step("ezc3d", "ok"), _step("opensim", "not-installed")],
    )
    identities = classify_runtimes(probes)
    assert identities["native-opensim"] == "unavailable: opensim is not installed"
    assert identities["mixed-process"] == (
        "not-evaluable: opensim is not installed (no qualified evidence)"
    )
    assert exit_status(probes) == 0


def test_a_runtime_that_fails_alone_is_broken_and_fails_the_diagnostic() -> None:
    probes = _probes(
        ezc3d=[_step("ezc3d", "import-error", "DLL load failed: bad wheel")],
        opensim=[_step("opensim", "ok")],
        opensim_ezc3d=[_step("opensim", "ok"), _step("ezc3d", "import-error")],
        ezc3d_opensim=[_step("ezc3d", "import-error"), _step("opensim", "ok")],
    )
    identities = classify_runtimes(probes)
    assert identities["c3d-preparation"].startswith("broken: ImportError")
    assert identities["mixed-process"].startswith("not-evaluable")
    assert exit_status(probes) == 1


def test_both_orders_succeeding_is_compatible() -> None:
    probes = _probes(
        ezc3d=[_step("ezc3d", "ok")],
        opensim=[_step("opensim", "ok")],
        opensim_ezc3d=[_step("opensim", "ok"), _step("ezc3d", "ok")],
        ezc3d_opensim=[_step("ezc3d", "ok"), _step("opensim", "ok")],
    )
    assert classify_runtimes(probes)["mixed-process"] == "compatible"


def test_unclean_probe_start_is_never_classified() -> None:
    probes = _probes(
        ezc3d=[_step("ezc3d", "ok")],
        opensim=[_step("opensim", "ok")],
        opensim_ezc3d=[_step("opensim", "ok"), _step("ezc3d", "ok")],
        ezc3d_opensim=[_step("ezc3d", "ok"), _step("opensim", "ok")],
    )
    probes[0]["clean_start"] = False
    with pytest.raises(ValueError, match="clean"):
        classify_runtimes(probes)


def test_real_diagnostic_reports_explicit_identities_from_clean_processes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "identity.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.diagnostics.native_runtime_identity",
            "--python",
            sys.executable,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == exit_status(report["probes"]), result.stderr
    python = report["python"]
    assert python["executable"] and python["version"] and python["platform"]
    assert [tuple(p["order"]) for p in report["probes"]] == list(PROBE_ORDERS)
    for probe in report["probes"]:
        assert probe["clean_start"] is True
        for step in probe["steps"]:
            assert step["status"] in _STATUSES
            if step["status"] == "ok":
                assert step["file"]
                assert "ezc3d_dll" in step
            else:
                assert step["error_type"] and step["error"]
    assert report["runtime_identities"] == classify_runtimes(report["probes"])
