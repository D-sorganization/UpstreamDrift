"""Tests for CI infrastructure and dependency management.

These tests verify that:
1. All required dependencies are properly declared in pyproject.toml
2. Core modules can be imported without errors
3. Optional dependency handling works correctly
4. CI-critical paths are functional

This file addresses infrastructure issues identified in CI pipeline failures.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _major_minor(version: str) -> tuple[int, int]:
    """Return a crate/package major.minor tuple from an exact semver string."""
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected at least major.minor version, got {version!r}")
    return int(parts[0]), int(parts[1])


def _load_tauri_cargo_packages() -> dict[str, str]:
    """Load package versions from the generated Tauri Cargo lockfile."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found, no-redef]

    with open(REPO_ROOT / "ui" / "src-tauri" / "Cargo.lock", "rb") as lock_file:
        cargo_lock = tomllib.load(lock_file)

    return {package["name"]: package["version"] for package in cargo_lock["package"]}


class TestCoreDependencies:
    """Test that core dependencies are installed and importable."""

    def test_ci_infrastructure_numpy_available(self) -> None:
        """Test that numpy is available (always required)."""
        import numpy as np

        assert np.__version__ is not None

    def test_scipy_available(self) -> None:
        """Test that scipy is available (always required)."""
        import scipy

        assert scipy.__version__ is not None

    def test_structlog_available(self) -> None:
        """Test that structlog is available (OBS-001 requirement)."""
        import structlog

        assert structlog.__version__ is not None

    def test_fastapi_available(self) -> None:
        """Test that fastapi is available."""
        import fastapi

        assert fastapi.__version__ is not None

    def test_pydantic_available(self) -> None:
        """Test that pydantic is available."""
        import pydantic

        assert pydantic.__version__ is not None


class TestCoreModuleImports:
    """Test that core modules can be imported without errors."""

    def test_import_core(self) -> None:
        """Test that core module imports successfully."""
        from src.shared.python import core

        assert hasattr(core, "setup_logging")
        assert hasattr(core, "setup_structured_logging")
        assert hasattr(core, "get_logger")

    def test_import_engine_availability(self) -> None:
        """Test that engine_availability module imports successfully."""
        from src.shared.python.engine_core import engine_availability

        assert hasattr(engine_availability, "MUJOCO_AVAILABLE")
        assert hasattr(engine_availability, "STRUCTLOG_AVAILABLE")
        assert hasattr(engine_availability, "is_engine_available")

    def test_import_exceptions(self) -> None:
        """Test that exceptions module imports successfully."""
        from src.shared.python import exceptions

        assert hasattr(exceptions, "GolfModelingError")
        assert hasattr(exceptions, "EngineNotFoundError")

    def test_import_logging_config(self) -> None:
        """Test that logging_config module imports successfully."""
        from src.shared.python.logging_pkg import logging_config

        assert hasattr(logging_config, "get_logger")


class TestStructuredLogging:
    """Test structured logging functionality (OBS-001)."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test that get_logger returns a bound logger."""
        from src.shared.python.core import get_logger

        logger = get_logger(__name__)
        assert logger is not None
        # Should have info, warning, error methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_setup_structured_logging_idempotent(self) -> None:
        """Test that setup_structured_logging can be called multiple times."""
        from src.shared.python.core import setup_structured_logging

        # Should not raise on repeated calls
        setup_structured_logging()
        setup_structured_logging()
        setup_structured_logging()

    def test_logger_accepts_structured_data(self) -> None:
        """Test that logger accepts keyword arguments for structured data."""
        from src.shared.python.core import get_logger

        logger: Any = get_logger(__name__)
        # Should not raise exceptions
        logger.info("test_event", key1="value1", key2=123)


class TestEngineAvailabilityFlags:
    """Test engine availability detection."""

    def test_structlog_available_flag(self) -> None:
        """Test that structlog availability is properly detected."""
        from src.shared.python.engine_core.engine_availability import (
            STRUCTLOG_AVAILABLE,
        )

        # Since we added structlog as a dependency, it should be True
        assert STRUCTLOG_AVAILABLE is True

    def test_numpy_available_flag(self) -> None:
        """Test that numpy availability is properly detected."""
        from src.shared.python.engine_core.engine_availability import NUMPY_AVAILABLE

        assert NUMPY_AVAILABLE is True

    def test_scipy_available_flag(self) -> None:
        """Test that scipy availability is properly detected."""
        from src.shared.python.engine_core.engine_availability import SCIPY_AVAILABLE

        assert SCIPY_AVAILABLE is True

    def test_is_engine_available_function(self) -> None:
        """Test is_engine_available function."""
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        # These should always be true since they're core deps
        assert is_engine_available("numpy") is True
        assert is_engine_available("scipy") is True
        assert is_engine_available("structlog") is True

    def test_get_available_engines_returns_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_available_engines returns a list for core deps."""
        from src.shared.python.engine_core import engine_availability

        core_engine_mapping = {
            "numpy": "numpy",
            "scipy": "scipy",
            "structlog": "structlog",
        }
        monkeypatch.setattr(engine_availability, "_MODULE_MAPPING", core_engine_mapping)
        engine_availability.reset_engine_status_cache()

        try:
            available = engine_availability.get_available_engines()
            assert isinstance(available, list)
            assert len(available) > 0
            assert set(available) <= set(core_engine_mapping)
            # Core dependencies should be in the list
            assert "numpy" in available
            assert "scipy" in available
            assert "structlog" in available
        finally:
            engine_availability.reset_engine_status_cache()


class TestOptionalDependencyHandling:
    """Test graceful handling of optional dependencies."""

    def test_pyqt6_availability_flag_exists(self) -> None:
        """Test that PyQt6 availability flag exists."""
        from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

        # Flag should exist (value depends on environment)
        assert isinstance(PYQT6_AVAILABLE, bool)

    def test_mujoco_availability_flag_exists(self) -> None:
        """Test that MuJoCo availability flag exists."""
        from src.shared.python.engine_core.engine_availability import MUJOCO_AVAILABLE

        # Flag should exist (value depends on environment)
        assert isinstance(MUJOCO_AVAILABLE, bool)

    def test_skip_if_unavailable_decorator(self) -> None:
        """Test that skip_if_unavailable creates valid pytest marker."""
        from src.shared.python.engine_core.engine_availability import (
            skip_if_unavailable,
        )

        # Should return a pytest marker, not raise
        marker = skip_if_unavailable("nonexistent_engine_xyz")
        assert marker is not None


class TestCIEnvironmentCompatibility:
    """Tests specific to CI environment compatibility."""

    def test_pytest_importable(self) -> None:
        """Test that pytest is importable (test runner itself)."""
        assert pytest is not None
        assert pytest.__version__ is not None

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="CI runs on Linux",
    )
    def test_xvfb_compatible_qt_platform(self) -> None:
        """Test that QT_QPA_PLATFORM can be set to offscreen."""
        import os

        # This should not raise in CI with xvfb
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def test_ci_standard_xvfb_uses_dynamic_display_reservation(self) -> None:
        """Self-hosted PR jobs must not collide on a single fixed X display."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8",
        )
        start_step = workflow[
            workflow.index("- name: Start Xvfb") : workflow.index(
                "- name: Clean Stale Coverage Data"
            )
        ]
        stop_step = workflow[
            workflow.index("- name: Stop Xvfb") : workflow.index(
                "- name: Enforce Per-Package Coverage Thresholds"
            )
        ]

        assert "for display_num in $(seq 90 129)" in start_step
        assert 'lockdir="/tmp/upstreamdrift-xvfb-${display_num}.lock"' in start_step
        assert 'mkdir "$lockdir"' in start_step
        assert 'echo "DISPLAY=:${display_num}" >> "$GITHUB_ENV"' in start_step
        assert "Xvfb :99" not in start_step
        assert "UPSTREAMDRIFT_XVFB_PID" in stop_step
        assert "UPSTREAMDRIFT_XVFB_LOCKDIR" in stop_step
        assert "rmdir" in stop_step

    def test_pr_scoped_core_tests_treat_all_skipped_selection_as_noop(self) -> None:
        """PR-scoped pytest must not fail when every selected test self-skips."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8",
        )

        assert "pytest_exit_code=$?" in workflow
        assert "elif [ $pytest_exit_code -eq 5 ]; then" in workflow
        assert '-o addopts=""' in workflow
        assert "WARNING: pytest exit code 5 (no tests collected) detected." in (
            workflow
        )

    def test_core_only_install_includes_pytest_asyncio_for_repo_config(self) -> None:
        """Core-only pytest must load plugins required by pyproject config."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        core_only_step = workflow[
            workflow.index("- name: Run core-only test slice") : workflow.index(
                "quality-gate:",
                workflow.index("core-only-install:"),
            )
        ]

        assert 'asyncio_mode = "auto"' in (REPO_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        assert "pytest-asyncio" in core_only_step

    def test_cross_engine_equivalence_uses_recordless_pip_bootstrap(self) -> None:
        """The equivalence workflow must tolerate broken runner pip metadata."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-equivalence.yml"
        ).read_text(encoding="utf-8")

        assert "python -m pip install --ignore-installed --no-deps pip" in workflow
        assert "python -m pip install --upgrade pip" not in workflow

    def test_cross_engine_equivalence_disables_xvfb_plugin(self) -> None:
        """The non-GUI equivalence job must not start pytest-xvfb."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-equivalence.yml"
        ).read_text(encoding="utf-8")

        assert "-p no:xvfb" in workflow

    def test_cross_engine_equivalence_disables_pytest_plugin_autoload(self) -> None:
        """The equivalence gate must ignore globally installed pytest plugins."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-equivalence.yml"
        ).read_text(encoding="utf-8")

        assert 'PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1"' in workflow
        assert "mutually incompatible pytest plugins" in workflow

    def test_cross_engine_equivalence_runs_jaxsim_pinocchio_gate(self) -> None:
        """The equivalence workflow must run the JaxSim dynamics parity gate."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-equivalence.yml"
        ).read_text(encoding="utf-8")

        assert 'pip install -e ".[jaxsim]"' in workflow
        assert "tests/cross_engine/test_jaxsim_vs_pinocchio.py" in workflow

    def test_cross_engine_equivalence_hardens_against_skipped_parity(self) -> None:
        """The required parity gate must fail when JaxSim parity is all-skipped.

        Issue #6881: a green gate on skipped assertions is a false pass. The
        workflow must (a) assert the parity prerequisites are importable before
        pytest and (b) assert at least one parity case actually ran afterwards.
        """
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-equivalence.yml"
        ).read_text(encoding="utf-8")

        # Prerequisite gates before the parity test runs.
        assert "import jax, jaxlib, jaxsim" in workflow
        assert "pip uninstall -y pinocchio" in workflow
        assert "pin>=2.6.0,<5.0.0" in workflow
        assert "scripts/ci/check_pinocchio_dynamics_api.py" in workflow
        # Post-pytest assertion that a required parity case passed (not skipped).
        assert "scripts/ci/require_junit_test_passed.py" in workflow
        assert "test_jaxsim_pinocchio_free_body_dynamics_terms_match" in workflow

    def test_jaxsim_upgrade_guard_runs_pinned_equivalence_and_gradient_checks(
        self,
    ) -> None:
        """JaxSim bumps must be deliberate and guarded by parity checks."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = (
            REPO_ROOT / ".github" / "workflows" / "jaxsim-upgrade-guard.yml"
        ).read_text(encoding="utf-8")
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        workflow_data = yaml.safe_load(workflow)
        checkout = next(
            step
            for step in workflow_data["jobs"]["jaxsim-upgrade-guard"]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )

        assert 'jaxsim = ["jaxsim==0.9.0"]' in pyproject
        assert checkout["with"]["submodules"] == "recursive"
        assert checkout["with"]["persist-credentials"] is False
        assert 'pip install -e ".[dev,jaxsim]"' in workflow
        assert 'expected = "0.9.0"' in workflow
        assert "tests/motion_matching/test_cross_engine_equivalence.py" in workflow
        assert (
            "tests/unit/engines/pinocchio/test_fit_swing_gradient_math.py" in workflow
        )
        assert 'PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1"' in workflow

    def test_cross_engine_equivalence_runs_on_pyproject_changes(self) -> None:
        """The JaxSim pin guard must run when the declared extra changes."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-equivalence.yml"
        ).read_text(encoding="utf-8")

        assert '      - "pyproject.toml"' in workflow
        assert workflow.count('      - "pyproject.toml"') == 2

    def test_cross_engine_leaderboard_removes_conflicting_pytest_plugins(
        self,
    ) -> None:
        """The leaderboard job must remove globally conflicting pytest plugins."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "cross-engine-leaderboard.yml"
        ).read_text(encoding="utf-8")

        install_index = workflow.index('pip install -e ".[dev]"')
        uninstall_index = workflow.index("pip uninstall -y pytest-vcr pytest-recording")
        pytest_index = workflow.index(
            "pytest tests/unit/motion_matching/test_leaderboard.py"
        )

        assert install_index < uninstall_index < pytest_index

    def test_cross_engine_workflows_let_pydantic_resolve_core(self) -> None:
        """Cross-engine jobs must not force an incompatible pydantic-core wheel."""
        workflow_names = [
            "cross-engine-equivalence.yml",
            "cross-engine-leaderboard.yml",
            "cross-engine-leaderboard-publish.yml",
        ]

        for workflow_name in workflow_names:
            workflow = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(
                encoding="utf-8"
            )
            assert "pydantic-core==" not in workflow
            assert "--no-deps pydantic-core" not in workflow

    def test_nightly_cross_engine_runs_real_validator_suite(self) -> None:
        """Nightly validation must not target an empty placeholder test file."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "nightly-cross-engine.yml"
        ).read_text(encoding="utf-8")
        validation_step = workflow[
            workflow.index(
                "- name: Run cross-engine validation tests"
            ) : workflow.index(
                "- name: Upload test results",
            )
        ]

        assert "tests/heavy_integration/test_cross_engine_integration.py" not in (
            validation_step
        )
        for test_path in (
            "tests/integration/test_cross_engine_validation.py",
            "tests/unit/test_cross_engine_validator.py",
            "tests/integration/cross_engine/test_conformance_harness.py",
        ):
            assert test_path in validation_step
        assert "--cov=src.shared.python.engine_core.cross_engine_validator" in (
            validation_step
        )
        assert "--cov=shared/python/cross_engine_validator" not in validation_step

    def test_nightly_cross_engine_summary_fails_when_no_tests_collect(self) -> None:
        """Zero collected tests are a workflow failure, not a clean validation run."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "nightly-cross-engine.yml"
        ).read_text(encoding="utf-8")
        summary_step = workflow[
            workflow.index("- name: Summarize validation results") : workflow.index(
                "- name: Send notification on ERROR",
            )
        ]

        assert "has_failures = tests == 0 or failures > 0 or errors > 0" in (
            summary_step
        )
        assert "- No tests were collected." in summary_step

    def test_tauri_check_verifies_rust_toolchain_before_cargo_steps(self) -> None:
        """The Tauri check job must fail early when Rust setup is unusable."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "tauri-build.yml").read_text(
                encoding="utf-8",
            ),
        )
        check_job = workflow["jobs"]["check"]
        steps = check_job["steps"]
        step_names = [step.get("name") for step in steps]

        assert check_job["env"]["CARGO_HOME"].startswith("${{ github.workspace }}/")
        setup_index = step_names.index("Setup Rust")
        verify_index = step_names.index("Verify Rust toolchain")
        first_cargo_index = min(
            step_names.index(name)
            for name in (
                "Rust format check",
                "Repair Cargo target cache",
                "Rust clippy",
                "Cargo check",
            )
        )

        assert setup_index < verify_index < first_cargo_index
        verify_script = steps[verify_index]["run"]
        assert 'echo "$CARGO_HOME/bin" >> "$GITHUB_PATH"' in verify_script
        assert 'export PATH="$CARGO_HOME/bin:$PATH"' in verify_script
        for executable in ("rustup", "rustc", "cargo"):
            assert f"command -v {executable}" in verify_script
            assert f"{executable} --version" in verify_script

        cache_step = steps[step_names.index("Cache Rust target (check)")]
        assert "${{ runner.name }}" in cache_step["with"]["key"]

    def test_tauri_action_script_exists_for_build_workflow(self) -> None:
        """The Tauri action invokes npm run tauri build by default."""
        package_json = json.loads(
            (REPO_ROOT / "ui" / "package.json").read_text(encoding="utf-8")
        )
        scripts = package_json["scripts"]

        assert scripts["tauri"] == "tauri"
        assert scripts["tauri:build"] == "tauri build"

    def test_tauri_rust_and_npm_versions_share_minor(self) -> None:
        """Tauri app builds require Rust tauri and npm package parity."""
        package_lock = json.loads(
            (REPO_ROOT / "ui" / "package-lock.json").read_text(encoding="utf-8")
        )
        npm_packages = package_lock["packages"]
        cargo_packages = _load_tauri_cargo_packages()
        rust_tauri_version = cargo_packages["tauri"]

        for package_name in ("@tauri-apps/api", "@tauri-apps/cli"):
            npm_version = npm_packages[f"node_modules/{package_name}"]["version"]
            assert _major_minor(rust_tauri_version) == _major_minor(npm_version), (
                "tauri-action rejects mismatched Tauri minor versions: "
                f"tauri {rust_tauri_version} vs {package_name} {npm_version}"
            )

    def test_tauri_linux_dependencies_cover_lockfile_native_packages(self) -> None:
        """Linux Tauri builds must install native -dev packages for sys crates."""
        cargo_packages = _load_tauri_cargo_packages()
        workflow = (REPO_ROOT / ".github" / "workflows" / "tauri-build.yml").read_text(
            encoding="utf-8"
        )
        install_lines = re.findall(r"apt_retry install -y .+", workflow)

        assert "libdbus-sys" in cargo_packages
        assert len(install_lines) == 2
        for install_line in install_lines:
            assert "libdbus-1-dev" in install_line

    def test_tauri_build_matrix_uses_named_runner_metadata(self) -> None:
        """Tauri build jobs must not expose array runner labels in job names."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow_path = REPO_ROOT / ".github" / "workflows" / "tauri-build.yml"
        workflow_text = workflow_path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(workflow_text)
        build_job = workflow["jobs"]["build"]
        matrix_entries = build_job["strategy"]["matrix"]["include"]

        assert build_job["name"] == "Build (${{ matrix.artifact_name }})"
        assert build_job["runs-on"] == "${{ matrix.runner }}"
        assert "matrix.platform" not in workflow_text
        assert {entry["artifact_name"] for entry in matrix_entries} == {
            "linux-x64",
            "windows-x64",
        }
        for entry in matrix_entries:
            assert "platform" not in entry
            assert isinstance(entry["artifact_name"], str)
            assert entry["os"] in {"linux", "windows"}

        upload = next(
            step
            for step in build_job["steps"]
            if step.get("name") == "Upload artifacts"
        )
        assert upload["with"]["name"] == (
            "golf-modeling-suite-${{ matrix.artifact_name }}"
        )

    def test_tauri_windows_build_avoids_bash_rust_toolchain_action(self) -> None:
        """Windows self-hosted build setup must avoid the bash-based Rust action."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "tauri-build.yml").read_text(
                encoding="utf-8",
            ),
        )
        build_job = workflow["jobs"]["build"]
        steps = build_job["steps"]
        unix_setup = next(
            step for step in steps if step.get("name") == "Setup Rust (Unix)"
        )
        windows_setup = next(
            step for step in steps if step.get("name") == "Setup Rust (Windows)"
        )

        assert unix_setup["if"] == "matrix.os != 'windows'"
        assert "dtolnay/rust-toolchain" in unix_setup["uses"]
        assert windows_setup["if"] == (
            "matrix.os == 'windows' && env.WINDOWS_TAURI_RELEASE_ENABLED == 'true'"
        )
        assert windows_setup["shell"] == "pwsh"
        assert "dtolnay/rust-toolchain" not in windows_setup.get("uses", "")
        windows_script = windows_setup["run"]
        assert (
            "rustup toolchain install stable --profile minimal --target $env:RUST_TARGET"
            in windows_script
        )
        assert "rustup default stable" in windows_script
        assert "rustup target add $env:RUST_TARGET" in windows_script
        assert "cargo --version" in windows_script

        cache_step = next(
            step for step in steps if step.get("name") == "Cache Rust target (build)"
        )
        cache_key = cache_step["with"]["key"]
        assert "${{ runner.name }}" in cache_key
        assert "${{ matrix.target }}" in cache_key

    def test_tauri_windows_release_packaging_requires_policy_opt_in(self) -> None:
        """Windows packaging needs an App-Control-compatible runner policy."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "tauri-build.yml").read_text(
                encoding="utf-8",
            ),
        )
        build_job = workflow["jobs"]["build"]
        steps = build_job["steps"]
        build_enabled = (
            "matrix.os != 'windows' || env.WINDOWS_TAURI_RELEASE_ENABLED == 'true'"
        )

        assert (
            build_job["env"]["WINDOWS_TAURI_RELEASE_ENABLED"]
            == "${{ vars.TAURI_WINDOWS_RELEASE_ENABLED == 'true' && 'true' || 'false' }}"
        )

        notice_step = next(
            step
            for step in steps
            if step.get("name") == "Report Windows Tauri release disabled"
        )
        assert notice_step["if"] == (
            "matrix.os == 'windows' && env.WINDOWS_TAURI_RELEASE_ENABLED != 'true'"
        )
        assert "Application Control" in notice_step["run"]
        assert "os error 4551" in notice_step["run"]
        assert "TAURI_WINDOWS_RELEASE_ENABLED=true" in notice_step["run"]

        gated_step_names = {
            "Setup Node.js",
            "Cache Rust target (build)",
            "Install frontend dependencies",
            "Build frontend",
            "Build Tauri app",
            "Upload artifacts",
        }
        gated_steps = {
            step["name"]: step for step in steps if step.get("name") in gated_step_names
        }
        assert set(gated_steps) == gated_step_names
        for step in gated_steps.values():
            assert step["if"] == build_enabled

    def test_bot_ci_trigger_validates_token_before_authenticated_trigger(
        self,
    ) -> None:
        """The bot trigger job must skip gracefully when its token is invalid."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "Bot-CI-Trigger.yml"
        ).read_text(encoding="utf-8")

        assert "id: token-check" in workflow
        assert "gh auth status" in workflow
        assert (
            "for candidate in BOT_PAT_TOKEN RUNNER_CHECK_TOKEN_VALUE DEFAULT_GITHUB_TOKEN"
            in workflow
        )
        assert "trying next candidate" in workflow
        assert "BOT_TRIGGER_TOKEN=$token" in workflow
        assert "steps.token-check.outputs.can_trigger == 'true'" in workflow

    def test_frontend_cleanup_runs_before_ui_working_directory_default(self) -> None:
        """The frontend pre-checkout cleanup must not require ui/ to exist."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8",
            ),
        )
        steps = workflow["jobs"]["frontend-tests"]["steps"]
        cleanup = next(
            step for step in steps if step.get("name") == "Clean corrupt git objects"
        )

        assert cleanup["working-directory"] == "."

    def test_quality_gate_lod_timeout_budget_matches_self_hosted_setup_cost(
        self,
    ) -> None:
        """The LOD gate must allow checkout/setup on busy self-hosted runners."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "quality-gate.yml").read_text(
                encoding="utf-8",
            ),
        )
        lod_job = workflow["jobs"]["lod-quality-gate"]

        assert int(lod_job["timeout-minutes"]) >= 15

    def test_quality_gate_runs_repo_wide_blocking_lod_check(self) -> None:
        """The required status must fail on new repo-wide LOD violations."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow_path = REPO_ROOT / ".github" / "workflows" / "quality-gate.yml"
        workflow_text = workflow_path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(workflow_text)

        assert set(workflow["jobs"]) == {"lod-quality-gate"}
        assert "scripts/ci/check_lod.py" in workflow_text
        assert "src \\" in workflow_text
        assert "--baseline scripts/ci/lod_baseline.txt" in workflow_text
        assert "--advisory" not in workflow_text

    def test_quality_gate_workflow_emits_required_status_on_every_pr(self) -> None:
        """The standalone required status must not be hidden behind path filters."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow_path = REPO_ROOT / ".github" / "workflows" / "quality-gate.yml"
        workflow_text = workflow_path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(workflow_text)
        job = workflow["jobs"]["lod-quality-gate"]

        assert job["name"] == "lod-quality-gate"
        # The runner comes from the same public/private dispatcher expression as
        # the rest of CI. This previously asserted a bare `d-sorg-fleet-docker`,
        # which stopped being true when the expression landed.
        assert "d-sorg-fleet-docker" in job["runs-on"]
        # No `paths:` filter is what makes this job safe to require directly: it
        # reports on every PR, so it can never block on a missing context.
        assert "paths:" not in workflow_text

    def test_required_quality_gate_context_is_published_by_one_workflow(self) -> None:
        """Exactly one job may be named `quality-gate`.

        `quality-gate` is the required status check on `main` (org ruleset
        `Repository_Protections`). GitHub matches required checks by context
        name, so a second job sharing the name publishes a competing check run
        under the required context and branch protection is satisfied by
        whichever one reported. On PR #8728 three jobs carried this name: the
        CI Standard aggregate reported `failure` while docs-ci.yml and
        quality-gate.yml reported `success`, and the PR merged.
        """
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        publishers = []
        for workflow_path in sorted(
            (REPO_ROOT / ".github" / "workflows").glob("*.yml")
        ):
            workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
            if not isinstance(workflow, dict):
                continue
            for job_id, job in (workflow.get("jobs") or {}).items():
                if not isinstance(job, dict):
                    continue
                if job.get("name", job_id) == "quality-gate":
                    publishers.append(f"{workflow_path.name}:{job_id}")

        assert publishers == ["ci-standard.yml:quality-gate"], (
            "the required `quality-gate` context must be published by the CI "
            f"Standard aggregate alone, but found: {publishers}"
        )

    def test_helper_workflows_use_pr_scoped_concurrency(self) -> None:
        """Helper checks must not cancel another PR's current check status."""
        workflows = [
            "Jules-Redundant-PR-Closer.yml",
            "Comment-to-Issue-Converter.yml",
        ]

        for workflow_name in workflows:
            workflow = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(
                encoding="utf-8"
            )
            assert (
                "${{ github.event.pull_request.number || github.run_id }}" in workflow
            )

    def test_ci_standard_runner_guard_invokes_real_audit(self) -> None:
        """The required local-only status must not be a no-op."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )

        assert "scripts/check_local_only_workflows.py" in workflow
        assert 'echo "Bypass"' not in workflow

    def test_ci_standard_defines_required_quality_gate_status(self) -> None:
        """Branch protection requires the CI Standard / quality-gate status."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        job = workflow["jobs"]["quality-gate"]

        assert job["name"] == "quality-gate"
        assert set(job["needs"]) == {
            "pick-runner",
            "changed-paths",
            "code-quality",
            "companion-workflows",
            "security-scans",
            "repo-structure-gates",
            "tests",
            "unit-test-gate",
            "publication-quality",
            "docs-governance-gates",
            "rust-wheel-parity",
            "shared-tools-consumer-contracts",
            "seam-drift-gate",
        }
        assert job["if"] == "always()"
        aggregate_step = next(
            step
            for step in job["steps"]
            if step.get("name") == "Aggregate quality gate results"
        )
        aggregate = aggregate_step["run"]

        assert aggregate_step["env"]["TESTS"] == "${{ needs.tests.result }}"
        assert aggregate_step["env"]["PUBLICATION_QUALITY"] == (
            "${{ needs.publication-quality.result }}"
        )
        assert aggregate_step["env"]["SHARED_TOOLS_CONSUMER_CONTRACTS"] == (
            "${{ needs.shared-tools-consumer-contracts.result }}"
        )

        # A docs-only PR skips the general gates, so `skipped` has to be accepted -
        # but only once the prerequisites that decide the skip have themselves
        # succeeded, otherwise an infrastructure failure would skip everything
        # and read as a pass.
        assert 'expected="success"' in aggregate
        assert 'expected="skipped"' in aggregate
        assert (
            '[ "$PICK_RUNNER" != "success" ] || [ "$CHANGED_PATHS" != "success" ]'
            in aggregate
        )

    def test_ci_standard_runs_on_every_pull_request(self) -> None:
        """CI Standard must not be path-filtered out of any PR.

        `quality-gate` is a required check, and GitHub blocks a PR forever on a
        required context that never reports. While this workflow carried a
        `paths-ignore` for docs, docs-only PRs got the context from docs-ci.yml
        instead - which is what created the duplicate-name bypass. Skipping is
        now decided per job by `changed-paths`, not by the trigger.
        """
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        # PyYAML resolves the bare `on:` key to the boolean True.
        triggers = workflow.get(True, workflow.get("on"))
        pull_request = triggers["pull_request"]

        assert "paths-ignore" not in pull_request
        assert "paths" not in pull_request
        assert workflow["jobs"]["changed-paths"]["outputs"]["code"] == (
            "${{ steps.detect.outputs.code }}"
        )
        assert workflow["jobs"]["changed-paths"]["outputs"]["publication"] == (
            "${{ steps.detect.outputs.publication }}"
        )

    def test_ci_standard_repo_structure_installs_workflow_parser(self) -> None:
        """Workflow YAML guards must run after their parser dependency is present."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["repo-structure-gates"]["steps"]
        step_names = [step.get("name", "") for step in steps]

        parser_index = step_names.index("Install workflow parser dependency")
        trust_boundary_index = step_names.index("Workflow Run Trust Boundary Guard")

        assert parser_index < trust_boundary_index
        assert "pyyaml==6.0.2" in steps[parser_index]["run"]

    def test_ci_standard_tests_matrix_timeout_covers_core_suite_runtime(self) -> None:
        """The core tests matrix must not cancel before the bounded suite completes."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        tests_job = workflow["jobs"]["tests"]

        assert int(tests_job["timeout-minutes"]) >= 35
        core_step = next(
            step
            for step in tests_job["steps"]
            if step.get("name") == "Run Core Test Suite"
        )
        assert "--timeout=60" in core_step["run"]
        assert "pytest_parallel_args=(-n 0)" in core_step["run"]
        assert (
            "using serial pytest to avoid xdist worker termination" in core_step["run"]
        )

    def test_unit_gate_fetches_pr_base_before_child_copy_guard(self) -> None:
        """The unit ownership guard must have its fail-closed comparison ref."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["unit-test-gate"]["steps"]
        step_names = [step.get("name", "") for step in steps]

        fetch_index = step_names.index("Fetch PR base for ownership guards")
        unit_index = step_names.index("Run Green-Suite Unit Gate")
        fetch_step = steps[fetch_index]

        assert fetch_index < unit_index
        assert fetch_step["if"] == "github.event_name == 'pull_request'"
        assert (
            'git fetch --no-tags --depth=1 origin "${{ github.base_ref }}"'
            in fetch_step["run"]
        )

    def test_unit_gate_sparse_checks_out_pinned_tools_for_ownership_guard(self) -> None:
        """The guard must inspect the exact Tools pin before the unit gate runs.

        PR #8407 (epic #8390) moved the pin/checkout/verify trio from inline
        unit-test-gate steps into the shared composite action
        ``.github/actions/fetch-pinned-tools`` without updating this test,
        which kept asserting the deleted step names and failed on every run
        since. The semantics this test protects are unchanged and are now
        asserted where they live: the composite action resolves the pin from
        the superproject gitlink, checks Tools out at exactly that revision,
        and verifies the result. One expectation changed deliberately: the
        old inline step used ``sparse-checkout: src/shared/python``; the
        composite action documents why sparse checkout is now avoided (a
        sparse worktree left in a reused runner workspace poisons later jobs
        that materialize the submodule normally), so this test asserts its
        absence rather than its presence.
        """
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["unit-test-gate"]["steps"]
        step_names = [step.get("name", "") for step in steps]

        fetch_index = step_names.index("Fetch pinned Tools packages")
        unit_index = step_names.index("Run Green-Suite Unit Gate")
        assert fetch_index < unit_index
        assert steps[fetch_index]["uses"] == "./.github/actions/fetch-pinned-tools"

        action = yaml.safe_load(
            (
                REPO_ROOT / ".github" / "actions" / "fetch-pinned-tools" / "action.yml"
            ).read_text(encoding="utf-8")
        )
        action_steps = action["runs"]["steps"]
        action_names = [step.get("name", "") for step in action_steps]

        resolve_index = action_names.index("Resolve pinned Tools revision")
        checkout_index = action_names.index("Checkout pinned Tools source")
        verify_index = action_names.index("Verify pinned Tools checkout")
        assert resolve_index < checkout_index < verify_index

        resolve_step = action_steps[resolve_index]
        checkout_step = action_steps[checkout_index]
        verify_step = action_steps[verify_index]

        assert resolve_step["id"] == "pin"
        assert "git ls-tree HEAD -- vendor/ud-tools" in resolve_step["run"]
        assert checkout_step["uses"].startswith("actions/checkout@")
        assert "/Tools" in checkout_step["with"]["repository"]
        assert checkout_step["with"]["ref"] == "${{ steps.pin.outputs.sha }}"
        assert checkout_step["with"]["path"] == "vendor/ud-tools"
        assert checkout_step["with"]["fetch-depth"] == 1
        assert checkout_step["with"]["persist-credentials"] is False
        assert "sparse-checkout" not in checkout_step["with"]
        assert "git -C vendor/ud-tools rev-parse HEAD" in verify_step["run"]
        assert "${{ steps.pin.outputs.sha }}" in verify_step["run"]

    def test_release_builds_wheel_from_exact_tools_submodule(self) -> None:
        """Releases must not build an unverifiable wheel from an unpacked sdist."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["build"]["steps"]
        checkout = next(
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        build = next(step for step in steps if step.get("name") == "Build package")
        smoke_job = workflow["jobs"]["smoke-python-wheel"]
        smoke_checkout = next(
            step
            for step in smoke_job["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        smoke = next(
            step
            for step in smoke_job["steps"]
            if step.get("name") == "Run Python wheel smoke tests"
        )

        assert checkout["with"]["submodules"] == "recursive"
        assert checkout["with"]["persist-credentials"] is False
        assert workflow["jobs"]["build"]["outputs"]["wheel_filename"] == (
            "${{ steps.build-wheel.outputs.wheel_filename }}"
        )
        assert build["id"] == "build-wheel"
        assert "rm -rf dist" in build["run"]
        assert "python3 -m build --wheel" in build["run"]
        assert 'test "${#wheels[@]}" -eq 1' in build["run"]
        assert "wheel_filename=$(basename" in build["run"]
        assert smoke_checkout["with"]["submodules"] == "recursive"
        assert smoke_checkout["with"]["persist-credentials"] is False
        assert smoke["env"]["UPSTREAM_DRIFT_WHEEL"] == (
            "dist/${{ needs.build.outputs.wheel_filename }}"
        )

    def test_ci_standard_pr_scoped_tests_cannot_bypass_coverage_for_source(
        self,
    ) -> None:
        """Source PRs must run the scoped dependency-light lane."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )

        assert "id: core-tests" in workflow
        assert "mapfile -t changed_coverage_targets" in workflow
        assert "coverage_args=(--cov=src)" in workflow
        assert 'coverage_module="${target%.py}"' in workflow
        assert 'coverage_args+=(--cov="${coverage_module//\\//.}")' in workflow
        assert "src/**/*.py" in workflow
        assert 'echo "coverage_generated=true" >> "$GITHUB_OUTPUT"' in workflow
        assert (
            "Source/dependency coverage targets changed; targeted coverage lane will run after PR-scoped tests"
            in workflow
        )
        assert (
            "Source/dependency targets changed; running scoped dependency-light unit targets"
            in workflow
        )
        assert "PR-scoped dependency-light lane is running without coverage" in workflow
        assert '"${coverage_args[@]}"' in workflow
        selected_test_block_start = workflow.index(
            "printf '  %s\\n' \"${changed_tests[@]}\""
        )
        selected_test_block_end = workflow.index(
            "elif [ $pytest_exit_code -eq 5 ]",
            selected_test_block_start,
        )
        assert (
            '-o addopts=""'
            in workflow[selected_test_block_start:selected_test_block_end]
        )
        assert 'echo "full_coverage_generated=true" >> "$GITHUB_OUTPUT"' in workflow
        assert "steps.core-tests.outputs.full_coverage_generated == 'true'" in workflow
        assert (
            "steps.core-tests.outputs.coverage_generated == 'true'"
            not in workflow[
                workflow.index(
                    "- name: Enforce Per-Package Coverage Thresholds"
                ) : workflow.index("- name: Cross-Engine Validator Core Unit Tests")
            ]
        )
        assert (
            "github.event_name != 'pull_request'"
            not in workflow[
                workflow.index(
                    "- name: Enforce Per-Package Coverage Thresholds"
                ) : workflow.index("- name: Cross-Engine Validator Core Unit Tests")
            ]
        )

    def test_ci_standard_source_prs_do_not_run_only_changed_tests(self) -> None:
        """Source changes must not be validated solely by touched test files."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        pr_block = workflow[
            workflow.index(
                'if [ "${{ github.event_name }}" = "pull_request" ];'
            ) : workflow.index("# Run the targeted, dependency-light CI lane:")
        ]

        source_branch = 'elif [ "${#changed_coverage_targets[@]}" -gt 0 ]; then'
        changed_test_command = (
            'xvfb-run --auto-servernum python -m pytest "${changed_tests[@]}"'
        )

        assert source_branch in pr_block
        assert pr_block.index(source_branch) < pr_block.index(changed_test_command)
        assert "running scoped dependency-light unit targets for this PR" in pr_block

    def test_ci_standard_pr_targeted_coverage_runs_changed_file_ratchet(
        self,
    ) -> None:
        """PR-targeted coverage must enforce changed policy files explicitly."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        enforcer_step = workflow[
            workflow.index(
                "- name: Enforce Per-Package Coverage Thresholds"
            ) : workflow.index("- name: Cross-Engine Validator Core Unit Tests")
        ]

        assert 'echo "pr_targeted_coverage_generated=true" >> "$GITHUB_OUTPUT"' in (
            workflow
        )
        assert "$RUNNER_TEMP/changed_coverage_targets.txt" in workflow
        assert (
            "steps.core-tests.outputs.pr_targeted_coverage_generated == 'true'"
            in enforcer_step
        )
        assert '--changed-files "$RUNNER_TEMP/changed_coverage_targets.txt"' in (
            enforcer_step
        )

    def test_ci_standard_pr_tests_fail_on_deleted_test_files(self) -> None:
        """Deleted tests must not disappear from PR-scoped selection.

        The guard still refuses an unreviewed deletion; since #9412 the review
        is recorded in ``scripts/config/reviewed_test_deletions.json`` and
        enforced by ``scripts/ci/check_reviewed_test_deletions.py`` (which
        exits non-zero for any deletion without an entry), instead of the
        workflow failing unconditionally with no way to record the review.
        """
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        pr_block = workflow[
            workflow.index(
                'if [ "${{ github.event_name }}" = "pull_request" ];'
            ) : workflow.index("# Run the targeted, dependency-light CI lane:")
        ]

        assert "mapfile -t deleted_tests" in pr_block
        assert "--diff-filter=D" in pr_block

        guard_block = pr_block[
            pr_block.index("mapfile -t deleted_tests") : pr_block.index(
                "mapfile -t changed_core_targets"
            )
        ]
        assert 'if [ "${#deleted_tests[@]}" -gt 0 ]; then' in guard_block
        assert "scripts/ci/check_reviewed_test_deletions.py" in guard_block
        assert '--deleted-files "$RUNNER_TEMP/core_deleted_tests.txt"' in guard_block
        # The gate must be fatal: no `|| true`, no `continue-on-error` escape.
        assert "|| true" not in guard_block

        checker = REPO_ROOT / "scripts" / "ci" / "check_reviewed_test_deletions.py"
        manifest = REPO_ROOT / "scripts" / "config" / "reviewed_test_deletions.json"
        assert checker.is_file()
        assert manifest.is_file()
        assert "Deleted Python test files require review" in checker.read_text(
            encoding="utf-8"
        )

    def test_ci_standard_test_only_prs_stop_after_changed_tests_pass(self) -> None:
        """Changed-test-only PRs should not launch the broad core lane."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        selected_tests_block = workflow[
            workflow.index('echo "Running PR-scoped core tests:"') : workflow.index(
                "# Run the targeted, dependency-light CI lane:"
            )
        ]

        assert (
            "No source/dependency coverage targets changed; PR-scoped tests passed, skipping targeted coverage lane"
            in selected_tests_block
        )
        assert 'echo "coverage_generated=false" >> "$GITHUB_OUTPUT"' in (
            selected_tests_block
        )
        assert "exit 0" in selected_tests_block
        assert "Full dependency-light lane will run after PR-scoped tests" not in (
            selected_tests_block
        )

    def test_ci_optional_stack_prs_run_scoped_unit_lane(self) -> None:
        """The optional-stack workflow must run deterministic PR-relevant unit targets."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "ci-optional-stack.yml"
        ).read_text(encoding="utf-8")
        unit_step = workflow[
            workflow.index("- name: Run Unit Tests (Optional Stack)") : workflow.index(
                "- name: Optional-Stack Skip Visibility Report"
            )
        ]

        assert 'github.event_name }}" = "pull_request"' not in unit_step
        assert "changed_tests" not in unit_step
        assert "No unit test changes detected" not in unit_step
        assert "find tests/unit -mindepth 1 -maxdepth 1" not in unit_step
        for target in (
            "tests/unit/biomechanics",
            "tests/unit/deployment",
            "tests/unit/robotics",
        ):
            assert target in unit_step
        assert "Native" in unit_step
        assert "engine/equivalence lanes" in unit_step
        assert 'run_with_heartbeat "optional-stack unit target $target"' in unit_step
        assert 'pytest "$1"' in unit_step
        assert "unit_targets" in unit_step
        assert "break" in unit_step
        assert 'pip_retry install "trimesh>=4.0.0"' in workflow
        assert "OPTIONAL_STACK_UNIT_WORKERS" not in unit_step
        assert "pytest-xdist" not in unit_step
        assert " -n " not in unit_step
        assert "-n auto" not in unit_step

    def test_ci_optional_stack_forces_noninteractive_matplotlib(self) -> None:
        """The headless optional-stack job must ignore ambient GUI backends."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "ci-optional-stack.yml"
        ).read_text(encoding="utf-8")
        job_start = workflow.index("optional-stack-check:")
        job_header = workflow[job_start : workflow.index("    steps:", job_start)]

        assert "MPLBACKEND: Agg" in job_header

    def test_ci_optional_stack_pytest_exit_codes_are_gating(self) -> None:
        """The optional-stack lane must fail on pytest exit codes, not grep text."""
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "ci-optional-stack.yml"
        ).read_text(encoding="utf-8")
        job = workflow[workflow.index("optional-stack-check:") :]

        assert "Pre-existing optional-stack test failures tracked separately" not in job
        install_step = job[
            job.index("- name: Install System Dependencies") : job.index(
                "- name: Isolate Python tool cache"
            )
        ]
        assert "sudo -n true" in install_step
        assert "sudo apt-get" not in install_step
        assert "No root or non-interactive sudo available" in install_step

        for step_name, log_file in [
            ("Run API Tests (Optional Stack)", "/tmp/api-test-results.txt"),
            (
                "Run Pinocchio Ecosystem Tests (Optional Stack)",
                "/tmp/pinocchio-test-results.txt",
            ),
            ("Run Unit Tests (Optional Stack)", "/tmp/unit-test-results.txt"),
        ]:
            step = job[job.index(f"- name: {step_name}") :]
            next_step = step.find("\n      - name:", 1)
            if next_step != -1:
                step = step[:next_step]

            assert "continue-on-error: true" not in step
            assert f"tee {log_file} || true" not in step
            assert "set -o pipefail" in step
            assert "rc=$?" in step
            assert 'grep -c "FAILED"' in step
            assert '|| echo "0"' not in step

        api_step = job[
            job.index("- name: Run API Tests (Optional Stack)") : job.index(
                "- name: Run Pinocchio Ecosystem Tests (Optional Stack)"
            )
        ]
        unit_step = job[
            job.index("- name: Run Unit Tests (Optional Stack)") : job.index(
                "- name: Optional-Stack Skip Visibility Report"
            )
        ]
        assert 'exit "$rc"' in api_step
        assert 'exit "$rc"' in unit_step

        pinocchio_step = job[
            job.index(
                "- name: Run Pinocchio Ecosystem Tests (Optional Stack)"
            ) : job.index("- name: Run Unit Tests (Optional Stack)")
        ]
        assert '[[ "$rc" -eq 5 ]]' in pinocchio_step
        assert 'exit "$rc"' in pinocchio_step

    def test_physics_validation_script_targets_collect_tests(self) -> None:
        """Every physics runner target must collect at least one real test."""
        from scripts import validate_physics
        from scripts import verify_physics

        validate_source = (REPO_ROOT / "scripts" / "validate_physics.py").read_text(
            encoding="utf-8"
        )
        verify_source = (REPO_ROOT / "scripts" / "verify_physics.py").read_text(
            encoding="utf-8"
        )
        assert validate_source.index("sys.path.insert(0, str(_PROJECT_ROOT))") < (
            validate_source.index("from scripts.script_utils")
        )
        assert verify_source.index("sys.path.insert(0, str(_PROJECT_ROOT))") < (
            verify_source.index("from src.shared.python.engine_core.engine_manager")
        )

        paths = {
            *(
                REPO_ROOT / path
                for group in validate_physics.TEST_FILES.values()
                for path in group
            ),
            *(REPO_ROOT / path for path in verify_physics.VALIDATION_TEST_PATHS),
        }
        assert paths
        legacy_dir = REPO_ROOT / "tests" / "physics_validation"
        assert not list(legacy_dir.glob("test_*.py"))

        for path in sorted(paths):
            assert path.exists(), f"{path} must exist"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--collect-only",
                    "-q",
                    "-o",
                    "addopts=",
                    str(path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            nodeids = [
                line
                for line in result.stdout.splitlines()
                if "::" in line and not line.startswith("<")
            ]
            assert nodeids, f"{path} collected no tests"

    def test_pyqt6_fallback_is_not_expectation_shaped(self) -> None:
        """The global PyQt fallback may prevent crashes, but not satisfy UI asserts."""
        conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        pyqt_fallback = conftest[
            conftest.index("if not _has_pyqt6:") : conftest.index(
                "@pytest.fixture(autouse=True)"
            )
        ]

        for forbidden in [
            'font_mock.families.return_value = ["Outfit"]',
            "mock.return_value = [MagicMock()] * 4",
            '"Home"',
            '"Engines"',
            '"Documentation"',
            "mock_findChildren",
        ]:
            assert forbidden not in pyqt_fallback

        assert "__ud_fake__" in pyqt_fallback
        assert "_skip_fake_pyqt6_gui_items" in conftest

    def test_launcher_ui_setup_tests_assert_real_qt_results(self) -> None:
        """Launcher UI tests must not guard away assertions for mock-shaped values."""
        test_file = (
            REPO_ROOT / "tests" / "launchers" / "test_launcher_ui_setup.py"
        ).read_text(encoding="utf-8")

        assert "if isinstance(actions, list)" not in test_file
        assert "if isinstance(buttons, list)" not in test_file

    def test_ci_standard_rust_gate_runs_kernel_backed_python_suites(self) -> None:
        """Rust wheel CI must turn permanently skipped Python suites into failures."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        rust_gate = workflow[
            workflow.index("# RUST QUALITY GATE") : workflow.index("rust-quickstart:")
        ]
        binding_step = rust_gate[
            rust_gate.index("- name: Verify Python Bindings") : rust_gate.index(
                "- name: Build WASM Module"
            )
        ]

        assert "RUST_GATE_FILES=$(git diff --name-only" in rust_gate
        for path in [
            "'src/shared/python/physics/**'",
            "'src/tools/ball_flight_gui/**'",
            "'tests/unit/test_ball_flight_physics.py'",
            "'tests/unit/shared_python/test_ball_flight_physics.py'",
            "'tests/rust_bindings/**'",
        ]:
            assert path in rust_gate

        editable_install = 'python -m pip install --no-cache-dir --no-deps -e ".[dev]"'
        wheel_install = "python -m pip install --force-reinstall target/wheels/*.whl"
        assert editable_install in binding_step
        assert wheel_install in binding_step
        assert binding_step.index(editable_install) < binding_step.index(wheel_install)
        assert "CI_RUST_WHEELS_EXPECTED=1" in binding_step
        assert "tests/rust_bindings" in binding_step
        assert "tests/unit/test_ball_flight_physics.py" in binding_step
        assert "tests/unit/shared_python/test_ball_flight_physics.py" in binding_step
        assert '-o addopts=""' in binding_step

    def test_rust_wheel_parity_verifies_absolute_rust_toolchain(self) -> None:
        """The Rust parity wheel job must fail before maturin if Cargo is absent."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
                encoding="utf-8"
            )
        )
        parity_job = workflow["jobs"]["rust-wheel-parity"]
        steps = parity_job["steps"]
        step_names = [step.get("name") for step in steps]

        assert parity_job["env"]["CARGO_HOME"].startswith("${{ github.workspace }}/")
        assert step_names.index("Install Rust toolchain") < step_names.index(
            "Verify Rust toolchain"
        )
        assert step_names.index("Verify Rust toolchain") < step_names.index(
            "Build PyO3 Wheels (Maturin)"
        )

        verify_step = next(
            step for step in steps if step.get("name") == "Verify Rust toolchain"
        )
        verify_script = verify_step["run"]

        assert 'echo "$CARGO_HOME/bin" >> "$GITHUB_PATH"' in verify_script
        assert 'export PATH="$CARGO_HOME/bin:$PATH"' in verify_script
        for binary in ("rustup", "rustc", "cargo"):
            assert f"command -v {binary}" in verify_script
            assert f"{binary} --version" in verify_script

    def test_rust_wheel_parity_is_path_gated_for_non_rust_prs_and_pushes(self) -> None:
        """Rust wheel parity must stay fail-closed without running on every push."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML is required for workflow structure checks")

        workflow_text = (
            REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
        ).read_text(encoding="utf-8")
        workflow = yaml.safe_load(workflow_text)
        parity_job = workflow["jobs"]["rust-wheel-parity"]
        steps = parity_job["steps"]

        assert parity_job["env"]["CI_RUST_WHEELS_EXPECTED"] == "1"
        change_step = next(
            step for step in steps if step.get("id") == "rust-wheel-parity-changes"
        )
        change_script = change_step["run"]
        assert 'EVENT_NAME="${{ github.event_name }}"' in change_script
        assert '"$EVENT_NAME" = "workflow_dispatch"' in change_script
        assert '"$EVENT_NAME" = "schedule"' in change_script
        assert '"$EVENT_NAME" = "pull_request"' in change_script
        assert '"$EVENT_NAME" = "push"' in change_script
        assert "${{ github.event.before }}" in change_script
        assert "0000000000000000000000000000000000000000" in change_script
        assert "No Rust wheel parity changes detected" in change_script
        for pathspec in [
            "'rust_core/**'",
            "'src/shared/python/physics/**'",
            "'src/shared/python/motion_pipeline/**'",
            "'tests/rust_bindings/**'",
            "'tests/parity/**'",
            "'tests/unit/realtime/test_rust_parity.py'",
            "'scripts/ci/import_built_rust_wheels.py'",
            "'scripts/ci/check_rust_parity_wheel_gates.py'",
        ]:
            assert pathspec in change_script

        gated_step_names = {
            "Install System Dependencies",
            "Verify Rust toolchain",
            "Cache Cargo registry and build",
            "Build PyO3 Wheels (Maturin)",
            "Build codemap CLI binary",
            "Install project + built wheels",
            "Run parity suite (wheels mandatory)",
        }
        gate_expression = (
            "steps.rust-wheel-parity-changes.outputs.has_changes == 'true'"
        )
        gated_steps = {
            step.get("name"): step
            for step in steps
            if step.get("name") in gated_step_names
        }
        assert gated_steps.keys() == gated_step_names
        assert all(step.get("if") == gate_expression for step in gated_steps.values())

        summary_step = next(
            step for step in steps if step.get("name") == "Rust Wheel Parity Summary"
        )
        summary_script = summary_step["run"]
        assert summary_step["if"] == "always()"
        assert "skipped mandatory wheel parity suite for this non-Rust change" in (
            summary_script
        )
        assert "scheduled runs and manual dispatches still execute parity" in (
            summary_script
        )

    def test_semgrep_push_uses_changed_file_targets_when_before_sha_exists(
        self,
    ) -> None:
        """Push SAST should block new findings without re-failing legacy debt."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        semgrep_step = workflow[
            workflow.index("- name: Semgrep SAST Scan") : workflow.index(
                "# SECURITY: Bandit static security analysis",
            )
        ]

        assert '"${{ github.event_name }}" = "push"' in semgrep_step
        assert "${{ github.event.before }}" in semgrep_step
        assert "origin/${{ github.base_ref }}" in semgrep_step
        assert (
            'semgrep --config p/python --config p/security-audit --config p/owasp-top-ten --error "${semgrep_targets[@]}"'
            in semgrep_step
        )
        assert (
            "No changed source/application files supported by Semgrep" in semgrep_step
        )

    def test_core_test_matrix_push_uses_changed_file_scope_when_before_sha_exists(
        self,
    ) -> None:
        """Non-trunk push runs stay scoped so they do not fall through to OOM."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        core_test_step = workflow[
            workflow.index("- name: Run Core Test Suite") : workflow.index(
                "- name: Stop Xvfb",
            )
        ]

        assert '"${{ github.event_name }}" = "push"' in core_test_step
        assert "${{ github.event.before }}" in core_test_step
        assert 'diff_base="${{ github.event.before }}"' in core_test_step
        assert '--diff-filter=ACMRT "$diff_base" HEAD' in core_test_step
        assert "Core test suite NOT EXECUTED" in core_test_step

    def test_core_test_matrix_cannot_report_green_without_running(self) -> None:
        """A `tests` pass must mean the suite ran, or say loudly that it did not.

        `main` @ 6b68f94 reported `tests (3.11)` / `tests (3.12)` successful
        while the unit suite never ran: the push diff base was absent from the
        shallow checkout, every `git diff` failed with `fatal: bad object`, and
        because the results were read through `mapfile < <(git diff ...)` -
        which discards the exit code - the empty arrays were taken as "nothing
        changed" and the step exited 0 (issue #8771).

        Three invariants keep that closed, asserted here rather than left to
        the next reader of the shell.
        """
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        core_test_step = workflow[
            workflow.index("- name: Run Core Test Suite") : workflow.index(
                "- name: Stop Xvfb",
            )
        ]

        # 1. Diffs are captured through a file, so the step's `-e` sees a
        #    failed `git diff` instead of an indistinguishable empty array.
        assert "mapfile -t changed_tests < <(" not in core_test_step
        assert 'git diff --name-only "$@" >"$out"' in core_test_step

        # 2. An unresolvable diff base fails loudly, never "nothing changed".
        assert (
            'git rev-parse --verify --quiet "${diff_base}^{commit}"' in core_test_step
        )
        assert "Cannot resolve the core-test diff base" in core_test_step

        # 3. Trunk pushes always run the full lane. Path scoping on the default
        #    branch lets a non-Python commit vouch for a suite it never ran.
        assert (
            '"${{ github.ref_name }}" = "${{ github.event.repository.default_branch }}"'
            in core_test_step
        )

    def test_core_test_change_detection_fails_on_unresolvable_base(self) -> None:
        """The pre-step gating the whole matrix must not skip on a failed diff."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        start = workflow.index("- name: Check for core test relevant changes")
        detect_step = workflow[
            start : workflow.index("- name: Install System Dependencies", start)
        ]

        assert "set -euo pipefail" in detect_step
        assert (
            'git rev-parse --verify --quiet "origin/${{ github.base_ref }}^{commit}"'
            in detect_step
        )
        assert "Cannot resolve base ref" in detect_step
        assert "mapfile -d '' core_test_targets < <(" not in detect_step
        assert "Test matrix NOT EXECUTED" in detect_step

    def test_mypy_baseline_push_uses_changed_source_scope_when_before_sha_exists(
        self,
    ) -> None:
        """Push mypy should block changed-file debt without re-failing baseline drift."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        mypy_step = workflow[
            workflow.index("- name: MyPy Baseline") : workflow.index(
                "# Strict mypy for the fully-annotated API layer",
            )
        ]

        assert '"${{ github.event_name }}" = "push"' in mypy_step
        assert "${{ github.event.before }}" in mypy_step
        assert "0000000000000000000000000000000000000000" in mypy_step
        assert 'diff_base="${{ github.event.before }}"' in mypy_step
        assert (
            "git diff --name-only --diff-filter=ACMRT -z "
            "\"$diff_base\" HEAD -- 'src/**/*.py' 'src/*.py'"
        ) in mypy_step
        assert (
            "No changed source Python files; skipping baseline mypy check." in mypy_step
        )
        assert (
            'python3 scripts/ci/run_mypy.py "${src_py_files[@]}" --config-file pyproject.toml'
            in mypy_step
        )
        assert "python3 scripts/ci/run_full_mypy_baseline.py" in mypy_step

    def test_strict_api_mypy_push_uses_changed_api_scope_when_before_sha_exists(
        self,
    ) -> None:
        """Push strict API mypy must not re-fail unrelated API type debt."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        strict_step = workflow[
            workflow.index("- name: MyPy Strict (src/api)") : workflow.index(
                "- name: Alembic Drift Check",
            )
        ]

        assert '"${{ github.event_name }}" = "push"' in strict_step
        assert "${{ github.event.before }}" in strict_step
        assert "0000000000000000000000000000000000000000" in strict_step
        assert 'diff_base="${{ github.event.before }}"' in strict_step
        assert (
            "git diff --name-only --diff-filter=ACMRT -z "
            "\"$diff_base\" HEAD -- 'src/api/**/*.py' 'src/api/*.py'"
        ) in strict_step
        assert (
            "No changed API Python files; skipping strict API mypy check."
            in strict_step
        )
        assert (
            'mypy "${api_py_files[@]}" --strict --follow-imports=silent --config-file pyproject.toml'
            in strict_step
        )
        assert (
            "mypy src/api --strict --follow-imports=silent --config-file pyproject.toml"
            in strict_step
        )

    def test_model_explorer_xml_suppressions_are_build_only(self) -> None:
        """Model Explorer must parse untrusted XML through defusedxml only."""
        model_explorer = REPO_ROOT / "src" / "tools" / "model_explorer"
        suppression = "nosemgrep: python.lang.security.use-defused-xml.use-defused-xml"
        stdlib_parser_call = re.compile(r"(?<![A-Za-z0-9_])ET\.(?:parse|fromstring)\(")

        for path in model_explorer.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()

            for line_number, line in enumerate(lines, start=1):
                imports_stdlib_xml = (
                    "import xml.etree.ElementTree" in line
                    or "from xml.etree.ElementTree import" in line
                    or "from xml.dom import" in line
                )
                if imports_stdlib_xml:
                    assert suppression in line, f"{path}:{line_number}"

            if "import xml.etree.ElementTree as ET" in source:
                assert stdlib_parser_call.search(source) is None, f"{path}"


class TestPyprojectTomlConsistency:
    """Test that pyproject.toml is properly configured."""

    @staticmethod
    def _load_pyproject() -> dict[str, Any]:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found, no-redef]

        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)

    def test_pyproject_exists(self) -> None:
        """Test that pyproject.toml exists at repo root."""
        pyproject = REPO_ROOT / "pyproject.toml"
        assert pyproject.exists(), f"pyproject.toml not found at {pyproject}"

    def test_pyproject_has_required_sections(self) -> None:
        """Test that pyproject.toml has required sections."""
        data = self._load_pyproject()

        assert "project" in data
        assert "dependencies" in data["project"]
        assert "optional-dependencies" in data["project"]

    def test_structlog_in_dependencies(self) -> None:
        """Test that structlog is declared in dependencies."""
        data = self._load_pyproject()

        deps = data["project"]["dependencies"]
        # Check that structlog is in the dependencies
        assert any("structlog" in dep for dep in deps), (
            "structlog must be in core dependencies"
        )

    def test_api_runtime_dependencies_are_core_and_locked(self) -> None:
        """API auth/database imports must not require the dev extra."""
        data = self._load_pyproject()
        lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8").lower()

        deps = {
            requirement.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
            for requirement in data["project"]["dependencies"]
        }
        dev_deps = {
            requirement.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
            for requirement in data["project"]["optional-dependencies"]["dev"]
        }

        for package in {
            "alembic",
            "sqlalchemy",
            "bcrypt",
            "pyjwt",
            "cryptography",
            "email-validator",
            "starlette",
        }:
            assert package in deps
            assert package not in dev_deps
            assert f"{package}==" in lock

    def test_pytest_collects_in_tree_tests_by_default(self) -> None:
        """Default pytest config must include intentional colocated src tests."""
        data = self._load_pyproject()
        pytest_config = data["tool"]["pytest"]["ini_options"]

        assert "src/shared/python/ai/tests" in pytest_config["testpaths"]
        assert "src/shared/python/sidekick/tests" in pytest_config["testpaths"]
        assert "src" not in pytest_config["norecursedirs"]
