from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import check_coverage_gates as checker

REPO_ROOT = Path(__file__).resolve().parents[3]
BUDGET_PATH = REPO_ROOT / "scripts" / "config" / "mypy_exclusion_budget.json"


def _write_budget(tmp_path: Path, gates: list[dict[str, object]]) -> Path:
    budget_file = tmp_path / "budget.json"
    data = {
        "schema_version": 1,
        "schedule": [{"effective_on": "2026-01-01", "max_exclusions": 10}],
        "coverage_gates": gates,
        "exclusions": [],
    }
    budget_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return budget_file


def _write_coverage_json(tmp_path: Path, files: dict[str, tuple[int, int]]) -> Path:
    report_file = tmp_path / "coverage.json"
    files_data = {}
    for filename, (covered, total) in files.items():
        files_data[filename] = {
            "summary": {
                "covered_lines": covered,
                "num_statements": total,
                "percent_covered": (covered / total * 100.0) if total > 0 else 0.0,
            }
        }
    data = {"files": files_data}
    report_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return report_file


def _write_coverage_xml(
    tmp_path: Path, files: dict[str, tuple[int, int]], source: str | None = None
) -> Path:
    report_file = tmp_path / "coverage.xml"
    classes_xml = []
    for filename, (covered, total) in files.items():
        lines_xml = []
        for line_num in range(1, total + 1):
            hits = 1 if line_num <= covered else 0
            lines_xml.append(f'            <line number="{line_num}" hits="{hits}"/>')
        lines_str = "\n".join(lines_xml)
        classes_xml.append(
            f'        <class name="{Path(filename).stem}" filename="{filename}">\n'
            f"          <lines>\n{lines_str}\n          </lines>\n"
            f"        </class>"
        )
    body = "\n".join(classes_xml)
    xml_content = (
        '<?xml version="1.0" ?>\n'
        '<coverage version="7.0.0">\n'
        + (f"  <sources><source>{source}</source></sources>\n" if source else "")
        + "  <packages>\n"
        '    <package name="pkg">\n'
        f"      <classes>\n{body}\n      </classes>\n"
        "    </package>\n"
        "  </packages>\n"
        "</coverage>\n"
    )
    report_file.write_text(xml_content, encoding="utf-8")
    return report_file


@pytest.mark.unit
def test_budget_gates_read_exactly_names_from_budget_file() -> None:
    """The checker must read gate definitions directly from the budget file."""
    gates = checker.load_budget_gates(BUDGET_PATH)
    gate_names = [g.name for g in gates]
    raw_budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    expected_names = [g["name"] for g in raw_budget["coverage_gates"]]
    assert gate_names == expected_names
    assert len(gate_names) == 6


@pytest.mark.unit
def test_dbc_validation_rejects_path_without_trailing_slash(tmp_path: Path) -> None:
    """Gate path must end with a trailing slash to avoid prefix collisions."""
    budget = _write_budget(
        tmp_path,
        [
            {
                "name": "api-routes",
                "path": "src/api/routes",  # missing trailing slash
                "min_coverage": 30.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="path must end with '/'"):
        checker.load_budget_gates(budget)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_coverage", [-1.0, 101.0, "high"])
def test_dbc_validation_rejects_invalid_min_coverage(
    tmp_path: Path, invalid_coverage: object
) -> None:
    """Gate min_coverage must be a float/int in [0, 100]."""
    budget = _write_budget(
        tmp_path,
        [
            {
                "name": "api-routes",
                "path": "src/api/routes/",
                "min_coverage": invalid_coverage,
            }
        ],
    )
    with pytest.raises(ValueError, match="min_coverage"):
        checker.load_budget_gates(budget)


@pytest.mark.unit
def test_gate_at_or_above_floor_exits_0(tmp_path: Path) -> None:
    """All gates meeting or exceeding threshold return exit code 0."""
    budget = _write_budget(
        tmp_path,
        [
            {"name": "api", "path": "src/api/", "min_coverage": 50.0},
            {"name": "core", "path": "src/core/", "min_coverage": 60.0},
        ],
    )
    report = _write_coverage_xml(
        tmp_path,
        {
            "src/api/routes.py": (5, 10),  # 50.0% == 50.0%
            "src/core/main.py": (8, 10),  # 80.0% >= 60.0%
        },
    )
    exit_code = checker.main(["--report", str(report), "--budget", str(budget)])
    assert exit_code == 0


@pytest.mark.unit
def test_gate_below_floor_exits_1(tmp_path: Path) -> None:
    """A gate falling below floor returns exit code 1."""
    budget = _write_budget(
        tmp_path,
        [
            {"name": "api", "path": "src/api/", "min_coverage": 50.0},
            {"name": "core", "path": "src/core/", "min_coverage": 60.0},
        ],
    )
    report = _write_coverage_xml(
        tmp_path,
        {
            "src/api/routes.py": (4, 10),  # 40.0% < 50.0% (fails)
            "src/core/main.py": (8, 10),  # 80.0% >= 60.0%
        },
    )
    exit_code = checker.main(["--report", str(report), "--budget", str(budget)])
    assert exit_code == 1


@pytest.mark.unit
def test_gate_with_no_matching_files_exits_2(tmp_path: Path) -> None:
    """A gate matching zero files is misconfigured and returns exit code 2."""
    budget = _write_budget(
        tmp_path,
        [
            {"name": "api", "path": "src/api/", "min_coverage": 50.0},
            {"name": "unmatched", "path": "src/nonexistent/", "min_coverage": 30.0},
        ],
    )
    report = _write_coverage_xml(
        tmp_path,
        {
            "src/api/routes.py": (8, 10),
        },
    )
    exit_code = checker.main(["--report", str(report), "--budget", str(budget)])
    assert exit_code == 2


@pytest.mark.unit
def test_xml_and_json_reports_yield_identical_percentages(tmp_path: Path) -> None:
    """XML and JSON representations of the same coverage data produce identical gate percentages."""
    budget = _write_budget(
        tmp_path,
        [
            {"name": "api", "path": "src/api/", "min_coverage": 40.0},
            {"name": "shared", "path": "src/shared/", "min_coverage": 70.0},
        ],
    )
    file_stats = {
        "src/api/v1.py": (15, 25),  # 60.0%
        "src/api/v2.py": (5, 15),  # 33.3% -> api total: 20/40 = 50.0%
        "src/shared/utils.py": (75, 100),  # 75.0%
    }
    report_json = _write_coverage_json(tmp_path, file_stats)
    report_xml = _write_coverage_xml(tmp_path, file_stats)

    gates = checker.load_budget_gates(budget)
    cov_json = checker.load_coverage_report(report_json)
    cov_xml = checker.load_coverage_report(report_xml)

    results_json, code_json = checker.evaluate_gates(gates, cov_json)
    results_xml, code_xml = checker.evaluate_gates(gates, cov_xml)

    assert code_json == code_xml == 0
    for name in results_json:
        res_j = results_json[name]
        res_x = results_xml[name]
        assert res_j.covered == res_x.covered
        assert res_j.total == res_x.total
        assert pytest.approx(res_j.percent, 1e-4) == res_x.percent
        assert res_j.status == res_x.status


def _ci_shaped_report(tmp_path: Path, source: str) -> tuple[Path, Path]:
    """Budget + Cobertura report shaped like ``--cov=src``: names relative to <source>."""
    budget = _write_budget(
        tmp_path, [{"name": "api", "path": "src/api/", "min_coverage": 50.0}]
    )
    report = _write_coverage_xml(tmp_path, {"api/routes.py": (6, 10)}, source=source)
    return budget, report


@pytest.mark.unit
def test_xml_filenames_resolve_through_absolute_source(tmp_path: Path) -> None:
    """``--cov=src`` names files relative to <source>; gates see repo paths (#10965)."""
    budget, report = _ci_shaped_report(tmp_path, str(tmp_path / "src"))
    files = checker.load_coverage_report(report, repo_root=tmp_path)
    assert files == {"src/api/routes.py": (6, 10)}
    argv = ["--report", str(report), "--budget", str(budget)]
    assert checker.main([*argv, "--repo-root", str(tmp_path)]) == 0


@pytest.mark.unit
def test_xml_filenames_resolve_through_relative_source(tmp_path: Path) -> None:
    """A relative <source> is taken relative to the repository root."""
    _, report = _ci_shaped_report(tmp_path, "src")
    files = checker.load_coverage_report(report, repo_root=tmp_path)
    assert files == {"src/api/routes.py": (6, 10)}


@pytest.mark.unit
def test_xml_source_outside_repo_root_is_config_error(tmp_path: Path) -> None:
    """A <source> outside the repository cannot be mapped: exit 2, never a pass."""
    repo = tmp_path / "repo"
    repo.mkdir()
    budget, report = _ci_shaped_report(tmp_path, str(tmp_path / "elsewhere" / "src"))
    argv = ["--report", str(report), "--budget", str(budget)]
    assert checker.main([*argv, "--repo-root", str(repo)]) == 2


def test_xml_entity_expansion_is_rejected(tmp_path: Path) -> None:
    """Coverage XML is parsed XXE-safe: entity declarations are refused (#10989)."""
    report = tmp_path / "coverage.xml"
    report.write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE coverage [<!ENTITY lol "lol">]>\n'
        "<coverage><sources><source>&lol;</source></sources></coverage>\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        checker.load_coverage_xml(report, repo_root=tmp_path)
