"""Tests for CI infrastructure and dependency management.

These tests verify that:
1. All required dependencies are properly declared in pyproject.toml
2. Core modules can be imported without errors
3. Optional dependency handling works correctly
4. CI-critical paths are functional

This file addresses infrastructure issues identified in CI pipeline failures.
"""

import sys
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    def test_get_available_engines_returns_list(self) -> None:
        """Test that get_available_engines returns a list."""
        from src.shared.python.engine_core.engine_availability import (
            get_available_engines,
        )

        available = get_available_engines()
        assert isinstance(available, list)
        assert len(available) > 0
        # Core dependencies should be in the list
        assert "numpy" in available
        assert "scipy" in available


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
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "jaxsim-upgrade-guard.yml"
        ).read_text(encoding="utf-8")
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        assert 'jaxsim = ["jaxsim==0.9.0"]' in pyproject
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
        lod_job = workflow["jobs"]["quality-gate"]

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

        assert set(workflow["jobs"]) == {"quality-gate"}
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
        job = workflow["jobs"]["quality-gate"]

        assert job["name"] == "quality-gate"
        assert job["runs-on"] == "d-sorg-fleet-docker"
        assert "paths:" not in workflow_text

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
            "code-quality",
            "security-scans",
            "repo-structure-gates",
            "tests",
            "unit-test-gate",
        }
        assert job["if"] == "always()"
        aggregate = next(
            step
            for step in job["steps"]
            if step.get("name") == "Aggregate quality gate results"
        )["run"]

        assert "tests:              ${{ needs.tests.result }}" in aggregate
        assert '${{ needs.tests.result }}" != "success"' in aggregate

    def test_ci_standard_pr_scoped_tests_cannot_bypass_coverage_for_source(
        self,
    ) -> None:
        """PR-scoped tests must collect targeted coverage for source changes."""
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
        assert "Full dependency-light lane will run after PR-scoped tests" in workflow
        assert (
            "Source/dependency targets changed; running the dependency-light unit lane"
            in workflow
        )
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
        assert (
            "running the dependency-light unit lane instead of only changed tests"
            in pr_block
        )

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
        """Deleted tests must not disappear from PR-scoped selection."""
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
        assert "Deleted Python test files require review" in pr_block
        assert (
            "exit 1"
            in pr_block[
                pr_block.index("mapfile -t deleted_tests") : pr_block.index(
                    "mapfile -t changed_core_targets"
                )
            ]
        )

    def test_ci_standard_test_only_prs_fall_through_to_full_lane(self) -> None:
        """Changed tests may run first, but must not be the only PR coverage."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
            encoding="utf-8"
        )
        selected_tests_block = workflow[
            workflow.index('echo "Running PR-scoped core tests:"') : workflow.index(
                "# Run the targeted, dependency-light CI lane:"
            )
        ]

        assert "Full dependency-light lane will run after PR-scoped tests" in (
            selected_tests_block
        )
        assert (
            "No source/dependency coverage targets changed; skipping targeted coverage lane"
            not in selected_tests_block
        )
        assert "exit 0" not in selected_tests_block

    def test_ci_optional_stack_prs_run_unscoped_unit_lane(self) -> None:
        """The optional-stack workflow must not skip PRs with no changed tests."""
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
        assert "pytest tests/unit/" in unit_step
        assert 'OPTIONAL_STACK_UNIT_WORKERS: "4"' in unit_step
        assert '-n "$OPTIONAL_STACK_UNIT_WORKERS"' in unit_step
        assert "-n auto" not in unit_step

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
            conftest.index(
                'if not _has_pyqt6 and "PyQt6" not in sys.modules:'
            ) : conftest.index("@pytest.fixture(autouse=True)")
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

        editable_install = 'python -m pip install --no-cache-dir -e ".[dev]"'
        wheel_install = "python -m pip install --force-reinstall target/wheels/*.whl"
        assert editable_install in binding_step
        assert wheel_install in binding_step
        assert binding_step.index(editable_install) < binding_step.index(wheel_install)
        assert "CI_RUST_WHEELS_EXPECTED=1" in binding_step
        assert "tests/rust_bindings" in binding_step
        assert "tests/unit/test_ball_flight_physics.py" in binding_step
        assert "tests/unit/shared_python/test_ball_flight_physics.py" in binding_step
        assert '-o addopts=""' in binding_step


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
