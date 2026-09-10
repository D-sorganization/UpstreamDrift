"""Tests for the bounded startup phase machinery (issue #8360).

Pure-Python coverage of ``src.launchers.startup_phases``: deterministic
successful, missing, exception-raising and never-completing phases; the
closed outcome/category sets; the Tools/Rate provider probe across
editable, vendored-style, absent and broken checkouts; and the pin between
the probe's Rate entry point and the launcher manifest.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.launchers.startup_phases import (
    CATEGORY_IMPORT_FAILURE,
    CATEGORY_MISSING_CHECKOUT,
    CATEGORY_MISSING_DEPENDENCY,
    CATEGORY_SUBPROCESS_FAILURE,
    CATEGORY_TIMEOUT,
    OUTCOME_DEGRADED,
    OUTCOME_FAILED,
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    RATE_STANDALONE_ENTRY_POINT,
    ProviderProbeResult,
    StartupPhaseError,
    StartupPhaseRecord,
    StartupPhaseTimeout,
    StartupTimeline,
    classify_exception,
    format_phase_diagnostics,
    probe_tools_provider,
    run_bounded,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """Gate for never-completing phases; released at teardown so no thread leaks."""
    gate = threading.Event()
    yield gate
    gate.set()


# ---------------------------------------------------------------------------
# run_bounded / run_phase: success, exception, never-completing
# ---------------------------------------------------------------------------


def test_run_bounded_returns_value_and_reraises() -> None:
    assert run_bounded(lambda: 42, 1.0, name="ok") == 42
    with pytest.raises(KeyError):
        run_bounded(lambda: {}["missing"], 1.0, name="boom")


def test_run_bounded_times_out_never_completing_provider(
    release: threading.Event,
) -> None:
    with pytest.raises(StartupPhaseTimeout) as excinfo:
        run_bounded(release.wait, 0.05, name="Checking Tools provider")
    assert excinfo.value.phase == "Checking Tools provider"
    assert "0.05s" in str(excinfo.value)


def test_run_bounded_validates_arguments() -> None:
    with pytest.raises(TypeError):
        run_bounded("not callable", 1.0, name="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_bounded(lambda: None, 0, name="x")
    with pytest.raises(ValueError):
        run_bounded(lambda: None, 1.0, name="")


def test_timeline_records_successful_phase_with_timestamp() -> None:
    timeline = StartupTimeline()
    assert timeline.run_phase("registry", lambda: "reg", timeout_s=1.0) == "reg"
    (record,) = timeline.records
    assert record.outcome == OUTCOME_OK
    assert record.name == "registry"
    assert record.category == ""
    assert record.duration_ms >= 0
    # ISO-8601 with millisecond precision, e.g. 2026-09-10T10:00:00.123
    assert "T" in record.started_at and "." in record.started_at
    assert timeline.blocking_phase() is None
    assert timeline.current_phase is None


def test_optional_phase_exception_degrades_and_returns_none() -> None:
    timeline = StartupTimeline()

    def broken() -> None:
        raise ImportError("no such provider")

    assert timeline.run_phase("engine", broken, timeout_s=1.0) is None
    (record,) = timeline.records
    assert record.outcome == OUTCOME_FAILED
    assert record.category == CATEGORY_IMPORT_FAILURE
    assert "ImportError: no such provider" in record.detail
    assert timeline.blocking_phase() is record


def test_optional_phase_timeout_degrades(release: threading.Event) -> None:
    timeline = StartupTimeline()
    assert timeline.run_phase("provider", release.wait, timeout_s=0.05) is None
    (record,) = timeline.records
    assert record.outcome == OUTCOME_TIMEOUT
    assert record.category == CATEGORY_TIMEOUT


def test_required_phase_failure_raises_structured_error() -> None:
    timeline = StartupTimeline()

    def broken() -> None:
        raise ModuleNotFoundError("No module named 'yaml'")

    with pytest.raises(StartupPhaseError) as excinfo:
        timeline.run_phase("registry", broken, timeout_s=1.0, required=True)
    assert excinfo.value.record.category == CATEGORY_MISSING_DEPENDENCY
    assert "registry failed [missing_dependency]" in str(excinfo.value)


def test_required_phase_timeout_raises_structured_error(
    release: threading.Event,
) -> None:
    timeline = StartupTimeline()
    with pytest.raises(StartupPhaseError) as excinfo:
        timeline.run_phase("registry", release.wait, timeout_s=0.05, required=True)
    assert excinfo.value.record.outcome == OUTCOME_TIMEOUT


def test_outcome_of_maps_value_to_degraded() -> None:
    timeline = StartupTimeline()
    probe = ProviderProbeResult(
        provider="tools",
        available=False,
        category=CATEGORY_MISSING_CHECKOUT,
        detail="no Tools",
    )
    value = timeline.run_phase(
        "provider",
        lambda: probe,
        timeout_s=1.0,
        outcome_of=ProviderProbeResult.phase_outcome,
    )
    assert value is probe
    (record,) = timeline.records
    assert record.outcome == OUTCOME_DEGRADED
    assert record.category == CATEGORY_MISSING_CHECKOUT
    assert record.detail == "no Tools"


def test_diagnostics_name_blocking_and_running_phase(
    release: threading.Event,
) -> None:
    timeline = StartupTimeline()
    timeline.run_phase("registry", lambda: 1, timeout_s=1.0)
    timeline.run_phase("provider", release.wait, timeout_s=0.05)
    text = timeline.format_diagnostics()
    assert text.splitlines()[0] == "UpstreamDrift startup diagnostics"
    assert "blocking phase: provider" in text
    assert "timeout  provider" in text
    assert "[timeout]" in text

    # A phase that is still running is reported as such.
    running = threading.Thread(
        target=lambda: timeline.run_phase("slow", release.wait, timeout_s=5.0),
        daemon=True,
    )
    running.start()
    for _ in range(200):
        if timeline.current_phase == "slow":
            break
        threading.Event().wait(0.005)
    assert "still running: slow" in timeline.format_diagnostics()
    release.set()
    running.join(1.0)


def test_format_phase_diagnostics_validates_total() -> None:
    with pytest.raises(ValueError):
        format_phase_diagnostics((), total_ms=-1)


def test_phase_record_format_line() -> None:
    record = StartupPhaseRecord(
        name="Docker",
        started_at="2026-09-10T10:00:00.000",
        duration_ms=12,
        outcome=OUTCOME_OK,
        detail="not available",
    )
    assert record.format_line() == (
        "2026-09-10T10:00:00.000 ok       Docker (12 ms) not available"
    )


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (StartupPhaseTimeout("p", 1.0), CATEGORY_TIMEOUT),
        (ModuleNotFoundError("x"), CATEGORY_MISSING_DEPENDENCY),
        (ImportError("x"), CATEGORY_IMPORT_FAILURE),
        (SyntaxError("x"), CATEGORY_IMPORT_FAILURE),
        (subprocess.TimeoutExpired("docker", 1.0), CATEGORY_SUBPROCESS_FAILURE),
        (
            type("SecureSubprocessError", (Exception,), {})("x"),
            CATEGORY_SUBPROCESS_FAILURE,
        ),
        (RuntimeError("x"), ""),
    ],
)
def test_classify_exception(exc: BaseException, expected: str) -> None:
    assert classify_exception(exc) == expected


def test_classify_exception_rejects_non_exception() -> None:
    with pytest.raises(TypeError):
        classify_exception("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tools / Rate provider probe
# ---------------------------------------------------------------------------


def _make_tools_checkout(root: Path, *, with_entry: bool = True) -> Path:
    tools = root / "Tools"
    package = tools / "src" / "rate_of_closure"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    if with_entry:
        (tools / RATE_STANDALONE_ENTRY_POINT).write_text(
            "raise SystemExit(0)\n", encoding="utf-8"
        )
    return tools


def test_probe_reports_missing_checkout_when_nothing_resolves(tmp_path: Path) -> None:
    repo = tmp_path / "deep" / "UpstreamDrift"
    repo.mkdir(parents=True)
    result = probe_tools_provider(repo, None)
    assert result.available is False
    assert result.category == CATEGORY_MISSING_CHECKOUT
    assert "vendor/ud-tools" in result.detail
    assert result.phase_outcome()[0] == OUTCOME_DEGRADED


def test_probe_reports_missing_checkout_for_invalid_explicit_path(
    tmp_path: Path,
) -> None:
    result = probe_tools_provider(tmp_path, str(tmp_path / "nowhere"))
    assert result.category == CATEGORY_MISSING_CHECKOUT
    assert "TOOLS_REPO_PATH" in result.detail


def test_probe_finds_editable_tools_via_explicit_path(tmp_path: Path) -> None:
    tools = _make_tools_checkout(tmp_path)
    result = probe_tools_provider(tmp_path / "UpstreamDrift", str(tools))
    assert result.available is True
    assert result.category == ""
    assert result.entry_point == tools / RATE_STANDALONE_ENTRY_POINT
    assert result.pinned is False
    assert "env, unpinned" in result.detail
    assert result.phase_outcome() == (OUTCOME_OK, "", result.detail)


def test_probe_finds_sibling_tools_checkout(tmp_path: Path) -> None:
    _make_tools_checkout(tmp_path)
    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    result = probe_tools_provider(repo, None)
    assert result.available is True
    assert "sibling, unpinned" in result.detail


def test_probe_reports_missing_rate_entry_point(tmp_path: Path) -> None:
    tools = _make_tools_checkout(tmp_path, with_entry=False)
    result = probe_tools_provider(tmp_path, str(tools))
    assert result.available is False
    assert result.category == CATEGORY_MISSING_DEPENDENCY
    assert str(tools / RATE_STANDALONE_ENTRY_POINT) in result.detail


def test_probe_reports_missing_rate_package(tmp_path: Path) -> None:
    tools = tmp_path / "Tools"
    (tools / "src").mkdir(parents=True)
    (tools / RATE_STANDALONE_ENTRY_POINT).parent.mkdir(parents=True)
    (tools / RATE_STANDALONE_ENTRY_POINT).write_text("", encoding="utf-8")
    result = probe_tools_provider(
        tmp_path, str(tools), package="definitely_not_a_rate_package"
    )
    assert result.category == CATEGORY_MISSING_DEPENDENCY
    assert "definitely_not_a_rate_package" in result.detail


def test_probe_reports_import_failure_when_spec_lookup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.machinery

    tools = _make_tools_checkout(tmp_path)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise ImportError("broken finder")

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", _explode)
    result = probe_tools_provider(tmp_path, str(tools))
    assert result.category == CATEGORY_IMPORT_FAILURE
    assert "broken finder" in result.detail


def test_probe_validates_arguments(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        probe_tools_provider("not a path", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        probe_tools_provider(tmp_path, None, entry_point=tmp_path)
    with pytest.raises(ValueError):
        probe_tools_provider(tmp_path, None, package="")


def test_probe_never_mutates_sys_path(tmp_path: Path) -> None:
    import sys

    tools = _make_tools_checkout(tmp_path)
    before = list(sys.path)
    probe_tools_provider(tmp_path, str(tools))
    assert sys.path == before


# ---------------------------------------------------------------------------
# Rate launches through its direct standalone entry point
# ---------------------------------------------------------------------------


def test_manifest_launches_rate_through_standalone_entry_point() -> None:
    """The launcher tile and the startup probe agree on Rate's entry point."""
    manifest = json.loads(
        (REPO_ROOT / "src/config/launcher_manifest.json").read_text(encoding="utf-8")
    )
    tiles = manifest["tiles"] if isinstance(manifest, dict) else manifest
    rate = next(tile for tile in tiles if tile["id"] == "rate_of_closure")
    assert rate["provider"] == "tools"
    assert rate["path"] == "tools://" + RATE_STANDALONE_ENTRY_POINT.as_posix()
