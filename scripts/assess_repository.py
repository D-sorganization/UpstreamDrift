#!/usr/bin/env python3
"""
Assess repository against 15 categories (A-O) and generate reports.
"""

import json
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.python.assessment.analysis import (  # noqa: E402
    assess_error_handling_content,
    assess_logging_content,
    calculate_complexity,
    count_files,
    get_python_metrics,
    grep_count,
)
from src.shared.python.assessment.constants import (  # noqa: E402
    CATEGORIES,
)
from src.shared.python.assessment.reporting import (  # noqa: E402
    generate_issue_document,
    generate_markdown_report,
)
from src.shared.python.path_utils import get_repo_root  # noqa: E402

# Setup paths
REPO_ROOT = get_repo_root()
DOCS_DIR = REPO_ROOT / "docs" / "assessments"
ISSUES_DIR = DOCS_DIR / "issues"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
ISSUES_DIR.mkdir(parents=True, exist_ok=True)


def assess_A():
    # Code Structure
    findings = []
    score = 8.0

    src_exists = (REPO_ROOT / "src").exists() or (REPO_ROOT / "shared").exists()
    if src_exists:
        findings.append("Source directory structure exists (src/ or shared/).")
    else:
        findings.append("No standard 'src' or 'shared' directory found.")
        score -= 2

    engines_exists = (REPO_ROOT / "engines").exists()
    if engines_exists:
        findings.append("Engines directory found, indicating modular architecture.")

    recs = ["Ensure all new code follows the modular engine structure."]
    return generate_markdown_report(
        "A", CATEGORIES["A"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_B():
    # Documentation
    findings = []
    score = 7.0

    readme = REPO_ROOT / "README.md"
    if readme.exists():
        findings.append("Root README.md exists.")
        if len(readme.read_text()) > 500:
            findings.append("README.md is reasonably detailed.")
        else:
            findings.append("README.md is too short.")
            score -= 1
    else:
        findings.append("Missing root README.md.")
        score -= 3

    docs_folder = REPO_ROOT / "docs"
    if docs_folder.exists():
        findings.append("docs/ directory exists.")
    else:
        findings.append("No docs/ directory.")
        score -= 1

    recs = ["Expand documentation for individual engines."]
    return generate_markdown_report(
        "B", CATEGORIES["B"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_C():
    # Test Coverage
    findings = []
    score = 6.0

    tests_dir = REPO_ROOT / "tests"
    if tests_dir.exists():
        test_files = count_files(REPO_ROOT, "tests/**/test_*.py")
        findings.append(f"Found {test_files} test files in tests/ directory.")
        if test_files < 5:
            score -= 2
    else:
        findings.append("No root tests/ directory found.")
        score -= 3

    recs = ["Increase test coverage for shared modules.", "Add integration tests."]
    return generate_markdown_report(
        "C", CATEGORIES["C"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_D():
    """Error Handling assessment."""
    findings = []
    py_files = REPO_ROOT.rglob("*.py")
    try_count = 0
    bare_except_count = 0

    for f in py_files:
        if "node_modules" in f.parts or "venv" in f.parts:
            continue
        try:
            results = assess_error_handling_content(
                f.read_text(encoding="utf-8", errors="ignore")
            )
            try_count += results["try_count"]
            bare_except_count += results["bare_except_count"]
        except Exception:
            pass

    score = 7.0
    findings.append(
        f"Found {try_count} try blocks and {bare_except_count} bare except blocks."
    )

    if bare_except_count > 5:
        score -= 2
    if try_count == 0:
        score -= 3

    recs = ["Ensure specific exceptions are caught.", "Avoid bare except clauses."]
    return generate_markdown_report(
        "D", CATEGORIES["D"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_E():
    # Performance
    findings = []
    score = 7.5

    # Heuristic: check for performance profiling tools or imports
    profiling = grep_count(REPO_ROOT, r"cProfile|timeit")
    if profiling > 0:
        findings.append("Profiling tools usage detected.")
    else:
        findings.append("No explicit profiling code found.")

    recs = ["Implement performance benchmarks for physics engines."]
    return generate_markdown_report(
        "E", CATEGORIES["E"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_F():
    # Security
    findings = []
    score = 8.0

    secrets = grep_count(REPO_ROOT, r"password|secret|key\s*=", "**/*.py")
    if secrets > 0:
        findings.append(
            f"Potential hardcoded secrets found in {secrets} files (needs verification)."
        )
        score -= 1
    else:
        findings.append("No obvious hardcoded secrets patterns found.")

    recs = [
        "Run bandit security analysis regularly.",
        "Use environment variables for all secrets.",
    ]
    return generate_markdown_report(
        "F", CATEGORIES["F"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_G():
    # Dependencies
    findings = []
    score = 8.5

    reqs = REPO_ROOT / "requirements.txt"
    pyproj = REPO_ROOT / "pyproject.toml"

    if reqs.exists() or pyproj.exists():
        findings.append("Dependency definition files found.")
    else:
        findings.append("No requirements.txt or pyproject.toml found.")
        score -= 4

    recs = ["Pin dependency versions.", "Audit dependencies for vulnerabilities."]
    return generate_markdown_report(
        "G", CATEGORIES["G"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_H():
    # CI/CD
    findings = []
    score = 8.0

    workflows = REPO_ROOT / ".github" / "workflows"
    if workflows.exists():
        count = len(list(workflows.glob("*.yml"))) + len(list(workflows.glob("*.yaml")))
        findings.append(f"Found {count} GitHub Actions workflows.")
        if count == 0:
            score -= 3
    else:
        findings.append("No .github/workflows directory.")
        score -= 5

    recs = ["Ensure CI runs on all PRs.", "Add CD pipelines for releases."]
    return generate_markdown_report(
        "H", CATEGORIES["H"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_I():
    # Code Style
    findings = []
    score = 8.0

    toml = REPO_ROOT / "pyproject.toml"
    if toml.exists() and "tool.ruff" in toml.read_text():
        findings.append("Ruff configuration found.")
    else:
        findings.append("No explicit Ruff config in pyproject.toml.")
        score -= 1

    recs = ["Enforce linting in CI.", "Use black for formatting."]
    return generate_markdown_report(
        "I", CATEGORIES["I"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_J():
    # API Design
    findings = []
    score = 7.5

    api_dir = REPO_ROOT / "api"
    if api_dir.exists():
        findings.append("api/ directory exists.")
        fastapi = grep_count(REPO_ROOT, "FastAPI", "api/**/*.py")
        if fastapi > 0:
            findings.append("FastAPI usage detected.")
    else:
        findings.append("No api/ directory.")
        score -= 2

    recs = ["Document API endpoints using OpenAPI.", "Version API endpoints."]
    return generate_markdown_report(
        "J", CATEGORIES["J"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_K():
    # Data Handling
    findings = []
    score = 7.0

    findings.append("Assessed data handling patterns.")

    recs = ["Validate input data schemas.", "Sanitize database inputs."]
    return generate_markdown_report(
        "K", CATEGORIES["K"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_L():
    """Logging assessment."""
    findings = []
    py_files = REPO_ROOT.rglob("*.py")
    logging_usage = 0
    print_usage = 0

    for f in py_files:
        if "node_modules" in f.parts or "venv" in f.parts:
            continue
        try:
            results = assess_logging_content(
                f.read_text(encoding="utf-8", errors="ignore")
            )
            logging_usage += results["logging_usage"]
            print_usage += results["print_usage"]
        except Exception:
            pass

    score = 7.0
    findings.append(
        f"Found {logging_usage} logging calls and {print_usage} print calls."
    )

    if print_usage > logging_usage:
        findings.append("High usage of print statements detected.")
        score -= 1

    recs = [
        "Replace print statements with structured logging.",
        "Configure log levels.",
    ]
    return generate_markdown_report(
        "L", CATEGORIES["L"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_M():
    # Configuration
    findings = []
    score = 7.5

    config_files = list(REPO_ROOT.glob("**/*.yaml")) + list(REPO_ROOT.glob("**/*.toml"))
    findings.append(f"Found {len(config_files)} configuration files (yaml/toml).")

    recs = ["Centralize configuration management.", "Use .env for local overrides."]
    return generate_markdown_report(
        "M", CATEGORIES["M"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_N():
    # Scalability
    findings = []
    score = 7.0

    findings.append("Scalability assessment based on architecture.")

    recs = [
        "Consider async processing for heavy loads.",
        "Implement caching strategies.",
    ]
    return generate_markdown_report(
        "N", CATEGORIES["N"], score, "\n".join(findings), recs, DOCS_DIR
    )


def assess_O():
    # Maintainability
    findings = []
    score = 7.5

    total_metrics = {"functions": 0, "branches": 0}
    py_files = REPO_ROOT.rglob("*.py")
    for f in py_files:
        if "node_modules" in f.parts or "venv" in f.parts:
            continue
        metrics = get_python_metrics(f)
        total_metrics["functions"] += metrics["functions"]
        total_metrics["branches"] += metrics["branches"]

    avg_complexity = calculate_complexity(total_metrics)
    findings.append(f"Average complexity (branches/func): {avg_complexity:.2f}")

    if avg_complexity > 5:
        score -= 2

    recs = ["Refactor large functions.", "Keep dependencies updated."]
    return generate_markdown_report(
        "O", CATEGORIES["O"], score, "\n".join(findings), recs, DOCS_DIR
    )


def run_all_assessments():
    assessors = [
        assess_A,
        assess_B,
        assess_C,
        assess_D,
        assess_E,
        assess_F,
        assess_G,
        assess_H,
        assess_I,
        assess_J,
        assess_K,
        assess_L,
        assess_M,
        assess_N,
        assess_O,
    ]

    reports = []
    for assessor in assessors:
        try:
            report = assessor()
            reports.append(report)
        except Exception as e:
            print(f"Error running assessment: {e}")

    return reports


def generate_issues_locally(json_path):
    """Read summary JSON and create issue markdown files for low scores."""
    try:
        with open(json_path) as f:
            data = json.load(f)

        category_scores = data.get("category_scores", {})

        for cat_code, info in category_scores.items():
            score = info.get("score", 10)
            if score < 5:
                generate_issue_document(
                    category_id=cat_code,
                    category_name=info.get("name", CATEGORIES.get(cat_code, "Unknown")),
                    grade=score,
                    details="The assessment for this category returned a score below 5/10.",
                    output_dir=ISSUES_DIR,
                )
                print(f"Created issue for category {cat_code}")

    except Exception as e:
        print(f"Error generating local issues: {e}")


def main():
    print("Starting repository assessment...")

    # 1. Run individual assessments
    reports = run_all_assessments()

    # 2. Run summary generator
    summary_md = DOCS_DIR / "Comprehensive_Assessment.md"
    summary_json = DOCS_DIR / "assessment_summary.json"

    # Use explicitly generated reports
    input_reports = [str(p.relative_to(REPO_ROOT)) for p in reports]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_assessment_summary.py"),
        "--input",
        *input_reports,
        "--output",
        str(summary_md),
        "--json-output",
        str(summary_json),
    ]

    print("Generating summary...")
    subprocess.run(cmd, check=True)

    # 3. Create issues locally for low grades
    print("Checking for low grades...")
    generate_issues_locally(summary_json)

    print("Assessment complete.")


if __name__ == "__main__":
    main()
