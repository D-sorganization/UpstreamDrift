import json
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

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
CLASS = "tests.cross_engine.test_jaxsim_vs_pinocchio"
CASES = [
    "test_jaxsim_pinocchio_free_body_dynamics_terms_match[identity-angular-only]",
    "test_jaxsim_pinocchio_free_body_dynamics_terms_match[identity-mixed-velocity]",
    "test_jaxsim_pinocchio_free_body_dynamics_terms_match[rotated-world-origin]",
]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(f"<testsuite>{body}</testsuite>", encoding="utf-8")
    return path


def _exact(path: Path, name: str = "test_required_case") -> int:
    return require_test_passed(path, name, exact=True, classname=CLASS)


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


def test_exact_case_guard_uses_the_full_node_identity(tmp_path: Path) -> None:
    assert (
        _exact(
            _write(
                tmp_path,
                f'<testcase classname="{CLASS}" name="test_required_case_extra" />',
            )
        )
        == 1
    )
    junit = _write(
        tmp_path,
        f'<testcase classname="{CLASS}" name="test_required_case_extra" />'
        f'<testcase classname="{CLASS}" name="test_required_case">'
        '<properties><property name="case" value="present" /></properties>'
        "<system-out>ordinary output is not a failure</system-out></testcase>",
    )
    assert _exact(junit) == 0


@pytest.mark.parametrize("result_tag", ["failure", "error", "skipped"])
def test_exact_case_guard_rejects_explicit_nonpass_results(
    tmp_path: Path, result_tag: str
) -> None:
    junit = _write(
        tmp_path,
        f'<testcase classname="{CLASS}" name="test_required_case">'
        f'<{result_tag} message="not passed" /></testcase>',
    )
    assert _exact(junit) == 1


@pytest.mark.parametrize(
    ("body", "name", "classname"),
    [
        (
            f'<testcase classname="{CLASS}" name="test_required_case" />' * 2,
            "test_required_case",
            CLASS,
        ),
        ('<testcase classname="tests.module" name="case" />', "case", None),
    ],
)
def test_exact_case_guard_requires_unambiguous_class_identity(
    tmp_path: Path, body: str, name: str, classname: str | None
) -> None:
    assert (
        require_test_passed(
            _write(tmp_path, body), name, exact=True, classname=classname
        )
        == 1
    )


def test_required_cases_write_receipt_from_unique_passing_junit_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = _write(
        tmp_path,
        "".join(f'<testcase classname="{CLASS}" name="{name}" />' for name in CASES),
    )
    output = tmp_path / "receipt.json"
    args = [str(xml), "--required-classname", CLASS]
    for name in CASES:
        args.extend(["--required-case", name])
    args.extend(["--receipt-output", str(output), "--source-revision", "abc123"])
    monkeypatch.setattr(module.sys, "argv", ["guard", *args])
    monkeypatch.setattr(
        module, "version", lambda name: {"jaxsim": "0.9.0", "pin": "4.0"}[name]
    )
    monkeypatch.setenv("RUNNER_OS", "Linux")
    assert module.main() == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["source_revision"] == "abc123"
    assert receipt["executed_case_count"] == len(CASES)
    assert receipt["executed_cases"] == [f"{CLASS}::{name}" for name in CASES]
    assert receipt["runner_os"] == "Linux"


@pytest.mark.parametrize("status", ["failure", "error", "skipped"])
def test_required_cases_do_not_write_receipt_for_nonpassing_junit_nodes(
    tmp_path: Path, status: str
) -> None:
    body = "".join(
        f'<testcase classname="{CLASS}" name="{name}">'
        + (f"<{status} />" if name == CASES[0] else "")
        + "</testcase>"
        for name in CASES
    )
    output = tmp_path / "receipt.json"
    assert (
        module.require_junit_cases_and_write_receipt(
            _write(tmp_path, body), CLASS, CASES, output, "abc123", "Linux", {}
        )
        == 1
    )
    assert not output.exists()
