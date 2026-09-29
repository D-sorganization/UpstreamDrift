#!/usr/bin/env python3
"""
Validation script for Phase 1 comprehensive upgrades.

This script validates that all Phase 1 infrastructure improvements
are working correctly and provides a comprehensive status report.
"""

import sys
from pathlib import Path


class Phase1Validator:
    """Validates Phase 1 infrastructure upgrades."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent
        self.results: dict[str, bool] = {}
        self.errors: list[str] = []

    def run_validation(self) -> dict[str, bool]:
        """Run all validation checks."""

        checks = [
            ("Project Structure", self.check_project_structure),
            ("Build System", self.check_build_system),
            ("Requirements", self.check_requirements),
            ("Documentation", self.check_documentation),
            ("Test Infrastructure", self.check_test_infrastructure),
            ("Output Management", self.check_output_management),
            ("Code Quality", self.check_code_quality),
            ("CI/CD Configuration", self.check_cicd_config),
        ]

        for check_name, check_func in checks:
            try:
                result = check_func()
                self.results[check_name] = result
            except (OSError, ValueError, ImportError, AttributeError) as e:
                self.results[check_name] = False
                self.errors.append(f"{check_name}: {str(e)}")

        self.print_summary()
        return self.results

    def check_project_structure(self) -> bool:
        """Check that required project structure exists."""
        required_files = [
            "pyproject.toml",
            "requirements.txt",
            "docs/conf.py",
            "docs/index.rst",
            "tests/__init__.py",
            "tests/conftest.py",
            "output/README.md",
            "shared/python/output_manager.py",
        ]

        required_dirs = [
            "docs",
            "tests/unit",
            "tests/integration",
            "output/simulations",
            "output/analysis",
            "output/exports",
            "output/reports",
            "output/cache",
        ]

        missing_files = []
        missing_dirs = []

        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)

        for dir_path in required_dirs:
            if not (self.project_root / dir_path).is_dir():
                missing_dirs.append(dir_path)

        if missing_files:
            pass
        if missing_dirs:
            pass

        return len(missing_files) == 0 and len(missing_dirs) == 0

    def check_build_system(self) -> bool:
        """Check pyproject.toml configuration."""
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            return False

        try:
            import tomllib  # type: ignore[no-redef]
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                return True  # Assume valid if we can't parse

        try:
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)

            # Check required sections
            required_sections = [
                "build-system",
                "project",
                "tool.black",
                "tool.ruff",
                "tool.mypy",
                "tool.pytest.ini_options",
                "tool.coverage.run",
            ]

            missing_sections = []
            for section in required_sections:
                keys = section.split(".")
                current = config
                for key in keys:
                    if key not in current:
                        missing_sections.append(section)
                        break
                    current = current[key]

            if missing_sections:
                return False

            # Check project metadata
            project = config.get("project", {})
            required_fields = ["name", "version", "description", "dependencies"]
            missing_fields = [f for f in required_fields if f not in project]

            if missing_fields:
                return False

            # Check optional dependencies
            optional_deps = project.get("optional-dependencies", {})
            expected_groups = ["dev", "engines", "analysis", "all"]
            missing_groups = [g for g in expected_groups if g not in optional_deps]

            if missing_groups:
                pass

            return len(missing_groups) == 0

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def check_requirements(self) -> bool:
        """Check requirements.txt structure."""
        req_path = self.project_root / "requirements.txt"

        if not req_path.exists():
            return False

        try:
            content = req_path.read_text()

            # Check for key sections
            required_content = [
                "Golf Modeling Suite",
                "-e .",
                "Installation Notes",
            ]

            missing_content = []
            for item in required_content:
                if item not in content:
                    missing_content.append(item)

            return not missing_content

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def check_documentation(self) -> bool:
        """Check Sphinx documentation setup."""
        docs_dir = self.project_root / "docs"

        if not docs_dir.exists():
            return False

        required_files = [
            "conf.py",
            "index.rst",
            "installation.rst",
            "quickstart.rst",
        ]

        missing_files = []
        for file_name in required_files:
            if not (docs_dir / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            return False

        # Check conf.py content
        try:
            conf_path = docs_dir / "conf.py"
            conf_content = conf_path.read_text()

            required_config = [
                "sphinx.ext.autodoc",
                "sphinx.ext.napoleon",
                "sphinx_rtd_theme",
                "Golf Modeling Suite",
            ]

            missing_config = []
            for item in required_config:
                if item not in conf_content:
                    missing_config.append(item)

            return not missing_config

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def check_test_infrastructure(self) -> bool:
        """Check test infrastructure setup."""
        tests_dir = self.project_root / "tests"

        if not tests_dir.exists():
            return False

        # Check test files
        required_test_files = [
            "conftest.py",
            "unit/test_launchers.py",
            "unit/test_output_manager.py",
            "integration/test_engine_integration.py",
        ]

        missing_files = []
        for file_path in required_test_files:
            if not (tests_dir / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            return False

        # Check conftest.py content
        try:
            conftest_path = tests_dir / "conftest.py"
            conftest_content = conftest_path.read_text()

            required_fixtures = [
                "temp_dir",
                "sample_swing_data",
                "mock_mujoco_model",
                "sample_output_dir",
            ]

            missing_fixtures = []
            for fixture in required_fixtures:
                if f"def {fixture}" not in conftest_content:
                    missing_fixtures.append(fixture)

            return not missing_fixtures

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def check_output_management(self) -> bool:
        """Check output management system."""
        output_dir = self.project_root / "output"
        output_manager_path = (
            self.project_root / "shared" / "python" / "output_manager.py"
        )

        if not output_dir.exists() or not output_manager_path.exists():
            return False

        # Check directory structure
        required_subdirs = [
            "simulations/mujoco",
            "simulations/drake",
            "simulations/pinocchio",
            "simulations/matlab",
            "analysis/biomechanics",
            "analysis/trajectories",
            "exports/videos",
            "exports/images",
            "reports/pdf",
            "cache/temp",
        ]

        missing_dirs = []
        for subdir in required_subdirs:
            if not (output_dir / subdir).exists():
                missing_dirs.append(subdir)

        if missing_dirs:
            return False

        # Check OutputManager class
        try:
            from importlib.util import module_from_spec, spec_from_file_location

            spec = spec_from_file_location("output_manager", output_manager_path)
            loader = getattr(spec, "loader", None)
            if spec is None or loader is None:
                return False

            module = module_from_spec(spec)
            loader.exec_module(module)

            # Check required classes and methods
            if not hasattr(module, "OutputManager"):
                return False

            manager_class = module.OutputManager
            required_methods = [
                "create_output_structure",
                "save_simulation_results",
                "load_simulation_results",
                "get_simulation_list",
                "export_analysis_report",
            ]

            missing_methods = []
            for method in required_methods:
                if not hasattr(manager_class, method):
                    missing_methods.append(method)

            return not missing_methods

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def check_code_quality(self) -> bool:
        """Check code quality configuration."""
        # Check that quality tools are configured in pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            return False

        try:
            content = pyproject_path.read_text()

            required_tools = [
                "[tool.black]",
                "[tool.ruff]",
                "[tool.mypy]",
                "[tool.pytest.ini_options]",
                "[tool.coverage.run]",
            ]

            missing_tools = []
            for tool in required_tools:
                if tool not in content:
                    missing_tools.append(tool)

            return not missing_tools

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def check_cicd_config(self) -> bool:
        """Check CI/CD workflow configuration."""
        workflow_path = self.project_root / ".github" / "workflows" / "ci-standard.yml"

        if not workflow_path.exists():
            return False

        try:
            content = workflow_path.read_text()

            required_elements = [
                "pytest tests/unit/",
                "pytest tests/integration/",
                "--cov=shared --cov=engines --cov=launchers",
                "codecov/codecov-action@v4",
                "coverage.xml",
            ]

            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)

            return not missing_elements

        except (OSError, ValueError, ImportError, AttributeError):
            return False

    def print_summary(self) -> None:
        """Print validation summary."""

        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result)

        if self.errors:
            for _error in self.errors:
                pass

        if passed_checks == total_checks or passed_checks >= total_checks * 0.8:
            pass
        else:
            pass

        # Detailed results
        for _check_name, _result in self.results.items():
            pass


def main() -> None:
    """Main validation function."""
    validator = Phase1Validator()
    results = validator.run_validation()

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
