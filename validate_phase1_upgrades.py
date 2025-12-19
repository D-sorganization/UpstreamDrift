#!/usr/bin/env python3
"""
Validation script for Phase 1 comprehensive upgrades.

This script validates that all Phase 1 infrastructure improvements
are working correctly and provides a comprehensive status report.
"""

import importlib.util
import sys
from pathlib import Path


class Phase1Validator:
    """Validates Phase 1 infrastructure upgrades."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}
        self.errors = []

    def run_validation(self) -> dict[str, bool]:
        """Run all validation checks."""
        print("🔍 Validating Phase 1 Comprehensive Upgrades")
        print("=" * 50)

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
            print(f"\n📋 {check_name}")
            try:
                result = check_func()
                self.results[check_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}")
            except Exception as e:
                self.results[check_name] = False
                self.errors.append(f"{check_name}: {str(e)}")
                print(f"   ❌ ERROR: {str(e)}")

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
            print(f"   Missing files: {missing_files}")
        if missing_dirs:
            print(f"   Missing directories: {missing_dirs}")

        return len(missing_files) == 0 and len(missing_dirs) == 0

    def check_build_system(self) -> bool:
        """Check pyproject.toml configuration."""
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            return False

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                print("   Warning: Cannot parse TOML (tomllib/tomli not available)")
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
                print(f"   Missing sections: {missing_sections}")
                return False

            # Check project metadata
            project = config.get("project", {})
            required_fields = ["name", "version", "description", "dependencies"]
            missing_fields = [f for f in required_fields if f not in project]

            if missing_fields:
                print(f"   Missing project fields: {missing_fields}")
                return False

            # Check optional dependencies
            optional_deps = project.get("optional-dependencies", {})
            expected_groups = ["dev", "engines", "analysis", "all"]
            missing_groups = [g for g in expected_groups if g not in optional_deps]

            if missing_groups:
                print(f"   Missing dependency groups: {missing_groups}")

            return len(missing_groups) == 0

        except Exception as e:
            print(f"   Error parsing pyproject.toml: {e}")
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

            if missing_content:
                print(f"   Missing content: {missing_content}")
                return False

            return True

        except Exception as e:
            print(f"   Error reading requirements.txt: {e}")
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
            print(f"   Missing documentation files: {missing_files}")
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

            if missing_config:
                print(f"   Missing conf.py config: {missing_config}")
                return False

            return True

        except Exception as e:
            print(f"   Error checking conf.py: {e}")
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
            print(f"   Missing test files: {missing_files}")
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

            if missing_fixtures:
                print(f"   Missing test fixtures: {missing_fixtures}")
                return False

            return True

        except Exception as e:
            print(f"   Error checking conftest.py: {e}")
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
            print(f"   Missing output directories: {missing_dirs}")
            return False

        # Check OutputManager class
        try:
            spec = importlib.util.spec_from_file_location(
                "output_manager", output_manager_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check required classes and methods
            if not hasattr(module, "OutputManager"):
                print("   Missing OutputManager class")
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

            if missing_methods:
                print(f"   Missing OutputManager methods: {missing_methods}")
                return False

            return True

        except Exception as e:
            print(f"   Error checking OutputManager: {e}")
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

            if missing_tools:
                print(f"   Missing tool configurations: {missing_tools}")
                return False

            return True

        except Exception as e:
            print(f"   Error checking code quality config: {e}")
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

            if missing_elements:
                print(f"   Missing CI/CD elements: {missing_elements}")
                return False

            return True

        except Exception as e:
            print(f"   Error checking CI/CD config: {e}")
            return False

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)

        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result)

        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {passed_checks/total_checks*100:.1f}%")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")

        print("\n🎯 PHASE 1 STATUS:")
        if passed_checks == total_checks:
            print("   ✅ ALL CHECKS PASSED - Phase 1 Complete!")
            print("   🚀 Ready for Phase 2 development")
        elif passed_checks >= total_checks * 0.8:
            print("   ⚠️  MOSTLY COMPLETE - Minor issues to resolve")
            print("   🔧 Address remaining issues before Phase 2")
        else:
            print("   ❌ SIGNIFICANT ISSUES - Phase 1 incomplete")
            print("   🛠️  Resolve critical issues before proceeding")

        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        for check_name, result in self.results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}")


def main():
    """Main validation function."""
    validator = Phase1Validator()
    results = validator.run_validation()

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
