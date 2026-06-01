import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "require_junit_test_passed.py"
)
spec = importlib.util.spec_from_file_location("require_junit_test_passed", SCRIPT_PATH)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
require_test_passed = module.require_test_passed


def test_require_test_passed_fails_when_required_case_only_skipped(
    tmp_path: Path,
) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<testsuite>
  <testcase name="test_jaxsim_pinocchio_free_body_dynamics_terms_match[case0]">
    <skipped message="missing jax" />
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    assert (
        require_test_passed(
            junit,
            "test_jaxsim_pinocchio_free_body_dynamics_terms_match",
        )
        == 1
    )


def test_require_test_passed_accepts_at_least_one_passed_case(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<testsuite>
  <testcase name="test_jaxsim_pinocchio_free_body_dynamics_terms_match[case0]" />
  <testcase name="test_jaxsim_pinocchio_free_body_dynamics_terms_match[case1]">
    <skipped message="advisory" />
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    assert (
        require_test_passed(
            junit,
            "test_jaxsim_pinocchio_free_body_dynamics_terms_match",
        )
        == 0
    )
