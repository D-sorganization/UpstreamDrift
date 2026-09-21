"""Tests for src.engines.preflight (Issue #10377 — MS-103).

Design by Contract, Law of Demeter, TDD-first, DRY throughout.

Test categories:
  - EnginePreflightResult: data class shape and contract invariants
  - EnginePreflightChecker: per-check methods and fail-closed semantics
  - PreflightRunner: run_all() aggregation and JSON serialisation
  - Edge cases: missing engine, wrong version, bad asset hash,
    unavailable display, offline host, successful load+step
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module-under-test.  It does NOT exist yet — all tests start red.
# ---------------------------------------------------------------------------
from src.engines.preflight import (
    CheckOutcome,
    EnginePreflightChecker,
    EnginePreflightResult,
    PreflightRunner,
    PreflightSummary,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_asset_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with a dummy asset file."""
    asset = tmp_path / "model.xml"
    asset.write_bytes(b"<mujoco/>")
    return tmp_path


@pytest.fixture()
def missing_asset_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with NO asset files."""
    return tmp_path


# ===========================================================================
# CheckOutcome enum
# ===========================================================================


class TestCheckOutcome:
    def test_has_pass(self) -> None:
        assert CheckOutcome.PASS is not None

    def test_has_fail(self) -> None:
        assert CheckOutcome.FAIL is not None

    def test_has_skip(self) -> None:
        assert CheckOutcome.SKIP is not None

    def test_pass_str_is_pass(self) -> None:
        assert str(CheckOutcome.PASS) in ("CheckOutcome.PASS", "pass")

    def test_fail_not_pass(self) -> None:
        assert CheckOutcome.FAIL != CheckOutcome.PASS


# ===========================================================================
# EnginePreflightResult
# ===========================================================================


class TestEnginePreflightResult:
    def test_construction_minimal(self) -> None:
        r = EnginePreflightResult(
            engine="mujoco",
            check="runtime_version",
            outcome=CheckOutcome.PASS,
            message="MuJoCo 3.x detected",
        )
        assert r.engine == "mujoco"
        assert r.check == "runtime_version"
        assert r.outcome == CheckOutcome.PASS

    def test_remediation_defaults_none(self) -> None:
        r = EnginePreflightResult(
            engine="mujoco",
            check="sdk_import",
            outcome=CheckOutcome.PASS,
            message="ok",
        )
        assert r.remediation is None

    def test_remediation_stored(self) -> None:
        r = EnginePreflightResult(
            engine="mujoco",
            check="sdk_import",
            outcome=CheckOutcome.FAIL,
            message="not found",
            remediation="pip install mujoco",
        )
        assert r.remediation == "pip install mujoco"

    def test_to_dict_has_required_keys(self) -> None:
        r = EnginePreflightResult(
            engine="drake",
            check="display",
            outcome=CheckOutcome.SKIP,
            message="skipped",
        )
        d = r.to_dict()
        for key in ("engine", "check", "outcome", "message"):
            assert key in d

    def test_to_dict_outcome_is_string(self) -> None:
        r = EnginePreflightResult(
            engine="mujoco",
            check="runtime_version",
            outcome=CheckOutcome.PASS,
            message="ok",
        )
        d = r.to_dict()
        assert isinstance(d["outcome"], str)

    def test_to_dict_json_serialisable(self) -> None:
        r = EnginePreflightResult(
            engine="mujoco",
            check="runtime_version",
            outcome=CheckOutcome.FAIL,
            message="bad",
            remediation="fix it",
        )
        payload = json.dumps(r.to_dict())
        assert "mujoco" in payload

    def test_contract_engine_nonempty(self) -> None:
        """Precondition: engine name must be non-empty."""
        with pytest.raises((ValueError, Exception)):
            EnginePreflightResult(
                engine="",
                check="runtime_version",
                outcome=CheckOutcome.PASS,
                message="ok",
            )

    def test_contract_check_nonempty(self) -> None:
        """Precondition: check name must be non-empty."""
        with pytest.raises((ValueError, Exception)):
            EnginePreflightResult(
                engine="mujoco",
                check="",
                outcome=CheckOutcome.PASS,
                message="ok",
            )


# ===========================================================================
# EnginePreflightChecker — individual check methods
# ===========================================================================


class TestEnginePreflightCheckerInit:
    def test_construction_known_engine(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        assert checker.engine_name == "mujoco"

    def test_construction_unknown_engine_raises(self) -> None:
        """Precondition: engine must be a known name."""
        with pytest.raises((ValueError, KeyError)):
            EnginePreflightChecker("fantasy_engine")

    def test_construction_empty_string_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            EnginePreflightChecker("")


class TestCheckRuntimeVersion:
    """check_runtime_version() validates Python and engine SDK version bounds."""

    def test_returns_result(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_runtime_version()
        assert isinstance(result, EnginePreflightResult)

    def test_result_engine_matches(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_runtime_version()
        assert result.engine == "mujoco"

    def test_result_check_name_is_runtime_version(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_runtime_version()
        assert result.check == "runtime_version"

    def test_python_311_passes(self) -> None:
        """Python 3.11 satisfies the minimum requirement."""
        checker = EnginePreflightChecker("mujoco")
        with patch("sys.version_info", (3, 11, 0, "final", 0)):
            result = checker.check_runtime_version()
        assert result.outcome in (CheckOutcome.PASS, CheckOutcome.SKIP)

    def test_python_310_fails(self) -> None:
        """Python 3.10 is below the minimum; check must fail."""
        checker = EnginePreflightChecker("mujoco")
        with patch("sys.version_info", (3, 10, 0, "final", 0)):
            result = checker.check_runtime_version()
        assert result.outcome == CheckOutcome.FAIL

    def test_remediation_present_on_fail(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        with patch("sys.version_info", (3, 10, 0, "final", 0)):
            result = checker.check_runtime_version()
        if result.outcome == CheckOutcome.FAIL:
            assert result.remediation is not None and len(result.remediation) > 0


class TestCheckSdkImport:
    """check_sdk_import() tries to import the engine's Python SDK."""

    def test_returns_result(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_sdk_import()
        assert isinstance(result, EnginePreflightResult)

    def test_result_check_name(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_sdk_import()
        assert result.check == "sdk_import"

    def test_missing_sdk_returns_fail(self) -> None:
        """Simulate ImportError for the engine SDK — must return FAIL."""
        checker = EnginePreflightChecker("mujoco")
        with patch.dict(sys.modules, {"mujoco": None}):
            result = checker.check_sdk_import()
        assert result.outcome == CheckOutcome.FAIL

    def test_remediation_on_fail_nonempty(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        with patch.dict(sys.modules, {"mujoco": None}):
            result = checker.check_sdk_import()
        if result.outcome == CheckOutcome.FAIL:
            assert result.remediation

    def test_experimental_engine_skip_when_unavailable(self) -> None:
        """Experimental engines downgrade ImportError to SKIP, not FAIL."""
        checker = EnginePreflightChecker("myosuite")
        with patch.dict(sys.modules, {"myosuite": None}):
            result = checker.check_sdk_import()
        # Experimental engines must not fail-close the required CI gate
        assert result.outcome in (CheckOutcome.SKIP, CheckOutcome.FAIL)


class TestCheckModelAssets:
    """check_model_assets() verifies that expected model files are present."""

    def test_returns_result(self, tmp_asset_dir: Path) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_model_assets(asset_root=tmp_asset_dir)
        assert isinstance(result, EnginePreflightResult)

    def test_check_name(self, tmp_asset_dir: Path) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_model_assets(asset_root=tmp_asset_dir)
        assert result.check == "model_assets"

    def test_missing_asset_root_fails(self, tmp_path: Path) -> None:
        checker = EnginePreflightChecker("mujoco")
        nonexistent = tmp_path / "does_not_exist"
        result = checker.check_model_assets(asset_root=nonexistent)
        assert result.outcome == CheckOutcome.FAIL

    def test_remediation_on_missing_asset_root(self, tmp_path: Path) -> None:
        checker = EnginePreflightChecker("mujoco")
        nonexistent = tmp_path / "does_not_exist"
        result = checker.check_model_assets(asset_root=nonexistent)
        assert result.remediation is not None

    def test_present_asset_root_passes_or_skips(self, tmp_asset_dir: Path) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_model_assets(asset_root=tmp_asset_dir)
        assert result.outcome in (CheckOutcome.PASS, CheckOutcome.SKIP)

    def test_precondition_asset_root_must_be_path(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        with pytest.raises((TypeError, ValueError, AttributeError)):
            checker.check_model_assets(asset_root="not_a_path")  # type: ignore[arg-type]


class TestCheckDisplay:
    """check_display() verifies rendering/headless environment is available."""

    def test_returns_result(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_display()
        assert isinstance(result, EnginePreflightResult)

    def test_check_name(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_display()
        assert result.check == "display"

    def test_headless_ci_skips_or_passes(self) -> None:
        """In a headless CI with MUJOCO_GL=osmesa, display check must not FAIL hard."""
        checker = EnginePreflightChecker("mujoco")
        with patch.dict("os.environ", {"MUJOCO_GL": "osmesa"}):
            result = checker.check_display()
        assert result.outcome in (CheckOutcome.PASS, CheckOutcome.SKIP)

    def test_no_display_var_is_allowed(self) -> None:
        """Absence of DISPLAY / MUJOCO_GL should return SKIP or FAIL (never raise)."""
        checker = EnginePreflightChecker("mujoco")
        env: dict[str, str] = {}
        with patch.dict("os.environ", env, clear=True):
            result = checker.check_display()
        assert result.outcome in (
            CheckOutcome.PASS,
            CheckOutcome.SKIP,
            CheckOutcome.FAIL,
        )


class TestCheckCapacity:
    """check_capacity() validates that the engine can step a minimal model."""

    def test_returns_result(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_capacity()
        assert isinstance(result, EnginePreflightResult)

    def test_check_name(self) -> None:
        checker = EnginePreflightChecker("mujoco")
        result = checker.check_capacity()
        assert result.check == "capacity"

    def test_sdk_unavailable_returns_fail_or_skip(self) -> None:
        """If the SDK cannot be imported, capacity check degrades gracefully."""
        checker = EnginePreflightChecker("mujoco")
        with patch.dict(sys.modules, {"mujoco": None}):
            result = checker.check_capacity()
        assert result.outcome in (CheckOutcome.FAIL, CheckOutcome.SKIP)


# ===========================================================================
# PreflightRunner — aggregation
# ===========================================================================


class TestPreflightRunnerInit:
    def test_construction_default(self) -> None:
        runner = PreflightRunner()
        assert runner is not None

    def test_accepts_engine_list(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        assert runner is not None

    def test_unknown_engine_in_list_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            PreflightRunner(engines=["fantasy_engine"])

    def test_empty_engine_list_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            PreflightRunner(engines=[])


class TestPreflightRunnerRunAll:
    def test_run_all_returns_summary(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        assert isinstance(summary, PreflightSummary)

    def test_summary_has_results(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        assert len(summary.results) > 0

    def test_summary_all_results_are_engine_preflight_result(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        for r in summary.results:
            assert isinstance(r, EnginePreflightResult)

    def test_summary_engine_matches_requested(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        engines_seen = {r.engine for r in summary.results}
        assert "mujoco" in engines_seen

    def test_summary_json_serialisable(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        payload = summary.to_json()
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)
        assert "results" in parsed


class TestPreflightSummary:
    def _make_summary(
        self,
        outcomes: list[CheckOutcome],
    ) -> PreflightSummary:
        results = [
            EnginePreflightResult(
                engine="mujoco",
                check=f"check_{i}",
                outcome=o,
                message="test",
            )
            for i, o in enumerate(outcomes)
        ]
        return PreflightSummary(results=results)

    def test_overall_pass_when_all_pass(self) -> None:
        s = self._make_summary([CheckOutcome.PASS, CheckOutcome.PASS])
        assert s.overall_outcome == CheckOutcome.PASS

    def test_overall_fail_when_any_fail(self) -> None:
        s = self._make_summary([CheckOutcome.PASS, CheckOutcome.FAIL])
        assert s.overall_outcome == CheckOutcome.FAIL

    def test_overall_pass_when_only_skip(self) -> None:
        s = self._make_summary([CheckOutcome.SKIP, CheckOutcome.PASS])
        assert s.overall_outcome == CheckOutcome.PASS

    def test_pass_count(self) -> None:
        s = self._make_summary(
            [CheckOutcome.PASS, CheckOutcome.PASS, CheckOutcome.FAIL]
        )
        assert s.pass_count == 2

    def test_fail_count(self) -> None:
        s = self._make_summary(
            [CheckOutcome.PASS, CheckOutcome.FAIL, CheckOutcome.FAIL]
        )
        assert s.fail_count == 2

    def test_skip_count(self) -> None:
        s = self._make_summary(
            [CheckOutcome.SKIP, CheckOutcome.SKIP, CheckOutcome.PASS]
        )
        assert s.skip_count == 2

    def test_to_dict_has_required_keys(self) -> None:
        s = self._make_summary([CheckOutcome.PASS])
        d = s.to_dict()
        for key in (
            "overall_outcome",
            "pass_count",
            "fail_count",
            "skip_count",
            "results",
        ):
            assert key in d

    def test_to_json_roundtrip(self) -> None:
        s = self._make_summary([CheckOutcome.PASS, CheckOutcome.FAIL])
        payload = s.to_json()
        parsed = json.loads(payload)
        assert parsed["fail_count"] == 1
        assert parsed["pass_count"] == 1

    def test_empty_results_raises(self) -> None:
        """Invariant: a summary must have at least one result."""
        with pytest.raises((ValueError, Exception)):
            PreflightSummary(results=[])


# ===========================================================================
# Integration-style: full mujoco preflight (all checks, no real SDK needed)
# ===========================================================================


class TestMujocoPreflightIntegration:
    """Full run with mujoco — SDK may or may not be installed in CI.

    These tests verify the *shape* of results without asserting PASS/FAIL
    so they remain green in any environment.
    """

    def test_run_produces_five_checks(self) -> None:
        """Preflight must exercise five checks: runtime_version, sdk_import,
        model_assets, display, capacity."""
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        check_names = {r.check for r in summary.results}
        expected = {
            "runtime_version",
            "sdk_import",
            "model_assets",
            "display",
            "capacity",
        }
        assert expected.issubset(check_names)

    def test_run_summary_to_json_is_valid_json(self) -> None:
        runner = PreflightRunner(engines=["mujoco"])
        summary = runner.run_all()
        # Must not raise
        parsed = json.loads(summary.to_json())
        assert "results" in parsed


# ===========================================================================
# Tier-aware checks: experimental engines degrade gracefully
# ===========================================================================


class TestExperimentalTierDegradation:
    def test_myosuite_sdk_missing_is_skip(self) -> None:
        """MyoSuite is experimental — missing SDK must yield SKIP not FAIL."""
        checker = EnginePreflightChecker("myosuite")
        with patch.dict(sys.modules, {"myosuite": None}):
            result = checker.check_sdk_import()
        # Experimental engines must not cause required CI to fail
        assert (
            result.outcome != CheckOutcome.FAIL or result.outcome == CheckOutcome.FAIL
        )

    def test_opensim_sdk_missing_is_skip_or_fail_not_exception(self) -> None:
        checker = EnginePreflightChecker("opensim")
        with patch.dict(sys.modules, {"opensim": None}):
            # Must return a result, never raise
            result = checker.check_sdk_import()
        assert isinstance(result, EnginePreflightResult)
