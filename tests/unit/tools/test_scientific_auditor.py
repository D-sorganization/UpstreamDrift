"""Tests for src.shared.python.tools.scientific_auditor (Issues #1949, #1744)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.shared.python.tools.scientific_auditor import ScienceAuditor, run_audit


def _audit_code(source: str) -> list[dict]:
    """Audit a code snippet string and return detected risks."""
    import ast

    auditor = ScienceAuditor()
    tree = ast.parse(textwrap.dedent(source))
    auditor.visit(tree)
    return auditor.risks


class TestScienceAuditorDivision:
    def test_no_risks_for_safe_code(self) -> None:
        risks = _audit_code("x = 1 + 2")
        assert risks == []

    def test_division_by_variable_is_flagged(self) -> None:
        risks = _audit_code("y = a / b")
        assert len(risks) == 1
        assert risks[0]["type"] == "Singularity Risk"

    def test_division_by_nonzero_constant_not_flagged(self) -> None:
        risks = _audit_code("y = a / 2")
        assert risks == []

    def test_division_by_zero_is_flagged(self) -> None:
        # Dividing by the constant 0 is flagged (not non-zero)
        risks = _audit_code("y = a / 0")
        assert len(risks) == 1
        assert risks[0]["type"] == "Singularity Risk"

    def test_multiple_divisions_reported(self) -> None:
        risks = _audit_code("y = a / b\nz = c / d\n")
        singularity_risks = [r for r in risks if r["type"] == "Singularity Risk"]
        assert len(singularity_risks) == 2

    def test_risk_dict_has_line_key(self) -> None:
        risks = _audit_code("y = a / b")
        assert "line" in risks[0]
        assert isinstance(risks[0]["line"], int)

    def test_risk_dict_has_msg_key(self) -> None:
        risks = _audit_code("y = a / b")
        assert "msg" in risks[0]
        assert len(risks[0]["msg"]) > 0


class TestScienceAuditorTrigFunctions:
    def test_sin_with_numeric_constant_flagged(self) -> None:
        risks = _audit_code("import math; x = math.sin(45)")
        trig_risks = [r for r in risks if r["type"] == "Unit Ambiguity"]
        assert len(trig_risks) >= 1

    def test_cos_with_numeric_constant_flagged(self) -> None:
        risks = _audit_code("import math; x = math.cos(3.14)")
        trig_risks = [r for r in risks if r["type"] == "Unit Ambiguity"]
        assert len(trig_risks) >= 1

    def test_tan_with_numeric_constant_flagged(self) -> None:
        risks = _audit_code("import math; x = math.tan(90)")
        trig_risks = [r for r in risks if r["type"] == "Unit Ambiguity"]
        assert len(trig_risks) >= 1

    def test_trig_with_variable_not_flagged(self) -> None:
        risks = _audit_code("import math; x = math.sin(angle)")
        trig_risks = [r for r in risks if r["type"] == "Unit Ambiguity"]
        assert len(trig_risks) == 0

    def test_trig_risk_msg_mentions_function(self) -> None:
        risks = _audit_code("import math; x = math.sin(45)")
        trig_risks = [r for r in risks if r["type"] == "Unit Ambiguity"]
        assert any("sin" in r["msg"] for r in trig_risks)

    def test_standalone_sin_call_flagged(self) -> None:
        risks = _audit_code("from math import sin\nx = sin(1.0)")
        trig_risks = [r for r in risks if r["type"] == "Unit Ambiguity"]
        assert len(trig_risks) >= 1


class TestRunAudit:
    def test_run_audit_on_file_returns_list(self, tmp_path: Path) -> None:
        code = "x = a / b\n"
        py_file = tmp_path / "sample.py"
        py_file.write_text(code)
        risks = run_audit(py_file)
        assert isinstance(risks, list)
        assert len(risks) >= 1

    def test_run_audit_on_clean_file_returns_empty(self, tmp_path: Path) -> None:
        code = "x = 1 + 2\n"
        py_file = tmp_path / "clean.py"
        py_file.write_text(code)
        risks = run_audit(py_file)
        assert risks == []

    def test_run_audit_on_directory(self, tmp_path: Path) -> None:
        risky = tmp_path / "risky.py"
        risky.write_text("y = a / b\n")
        risks = run_audit(tmp_path)
        assert isinstance(risks, list)
        assert len(risks) >= 1

    def test_run_audit_skips_test_files(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test_something.py"
        test_file.write_text("y = a / b\n")
        risks = run_audit(tmp_path)
        assert risks == []
