from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.ci.require_junit_test_passed import require_test_passed


def _write_junit(path: Path, cases: list[tuple[str, str | None]]) -> None:
    suite = ET.Element("testsuite")
    for name, outcome in cases:
        case = ET.SubElement(suite, "testcase", name=name)
        if outcome == "skipped":
            ET.SubElement(case, "skipped")
        elif outcome == "failed":
            ET.SubElement(case, "failure")
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def test_require_test_passed_fails_when_required_case_only_skipped(tmp_path: Path) -> None:
    junit = tmp_path / "results.xml"
    _write_junit(junit, [("test_jaxsim_pinocchio_free_body_dynamics_terms_match", "skipped")])

    assert require_test_passed(
        junit,
        "test_jaxsim_pinocchio_free_body_dynamics_terms_match",
    ) == 1


def test_require_test_passed_accepts_at_least_one_passed_case(tmp_path: Path) -> None:
    junit = tmp_path / "results.xml"
    _write_junit(
        junit,
        [
            ("test_jaxsim_pinocchio_free_body_dynamics_terms_match[case0]", "skipped"),
            ("test_jaxsim_pinocchio_free_body_dynamics_terms_match[case1]", None),
        ],
    )

    assert require_test_passed(
        junit,
        "test_jaxsim_pinocchio_free_body_dynamics_terms_match",
    ) == 0
