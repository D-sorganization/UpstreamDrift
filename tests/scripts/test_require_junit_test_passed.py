import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
CLASS = "tests.cross_engine.test_jaxsim_vs_pinocchio"
CASES = [
    "test_jaxsim_pinocchio_free_body_dynamics_terms_match[identity-angular-only]",
    "test_jaxsim_pinocchio_free_body_dynamics_terms_match[identity-mixed-velocity]",
    "test_jaxsim_pinocchio_free_body_dynamics_terms_match[rotated-world-origin]",
]
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


def _case(name: str, classname: str = CLASS, result: str = "") -> str:
    return f'<testcase classname="{classname}" name="{name}">{result}</testcase>'


def _run_receipt(tmp_path: Path, monkeypatch, nodes, cases=CASES):
    xml, output = tmp_path / "junit.xml", tmp_path / "receipt.json"
    xml.write_text(f"<testsuite>{''.join(nodes)}</testsuite>", encoding="utf-8")
    argv = ["guard", str(xml), "--required-classname", CLASS]
    for case in cases:
        argv.extend(("--required-case", case))
    argv.extend(("--receipt-output", str(output), "--source-revision", "abc123"))
    monkeypatch.setattr(module.sys, "argv", argv)
    monkeypatch.setattr(
        module, "version", {"jaxsim": "0.9.0", "pin": "4.0"}.__getitem__
    )
    monkeypatch.setenv("RUNNER_OS", "Linux")
    return module.main(), output


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


def test_receipt_uses_unique_exact_ids_and_records_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, output = _run_receipt(
        tmp_path,
        monkeypatch,
        [_case(case) for case in CASES + [CASES[0] + "-prefix-collision"]],
    )
    assert result == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["source_revision"] == "abc123"
    assert receipt["runner_os"] == "Linux" and receipt["python"] and receipt["platform"]
    assert receipt["packages"] == {"jaxsim": "0.9.0", "pin": "4.0"}
    assert receipt["executed_case_count"] == 3
    assert receipt["executed_cases"] == [f"{CLASS}::{case}" for case in CASES]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-class",
        "missing",
        "duplicate",
        "prefix-only",
        "failure",
        "error",
        "skipped",
        "duplicate-request",
    ],
)
def test_receipt_rejects_ambiguous_or_nonpassing_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    nodes = [_case(case) for case in CASES]
    cases = CASES
    if mutation == "wrong-class":
        nodes[0] = _case(CASES[0], "other.module")
    elif mutation == "missing":
        nodes.pop()
    elif mutation == "duplicate":
        nodes.append(nodes[0])
    elif mutation in {"failure", "error", "skipped"}:
        nodes[0] = _case(CASES[0], result=f"<{mutation}/>")
    else:
        if mutation == "prefix-only":
            nodes[0] = _case(CASES[0] + "-suffix")
        else:
            cases = [CASES[0]] * 2
    result, output = _run_receipt(tmp_path, monkeypatch, nodes, cases)
    assert result == 1 and not output.exists()
