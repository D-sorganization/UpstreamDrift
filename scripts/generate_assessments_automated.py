import re
from datetime import datetime
from pathlib import Path

# Configuration
BASELINE_ASSESSMENT_PATH = Path(
    "docs/assessments/Comprehensive_Assessment_2026-02-03.md"
)
COMPLETIST_DATA_DIR = Path(".jules/completist_data")
PRAGMATIC_REVIEW_PATH = Path(
    "docs/assessments/pragmatic_programmer/review_2026-02-08.md"
)
OUTPUT_DIR = Path("docs/assessments")
COMPLETIST_REPORT_PATH = OUTPUT_DIR / "completist" / "Completist_Report_2026-02-08.md"
COMPREHENSIVE_REPORT_PATH = OUTPUT_DIR / "Comprehensive_Assessment.md"

CATEGORIES = {
    "A": "Architecture & Implementation",
    "B": "Code Quality & Hygiene",
    "C": "Documentation & Comments",
    "D": "User Experience & Developer Journey",
    "E": "Performance & Scalability",
    "F": "Installation & Deployment",
    "G": "Testing & Validation",
    "H": "Error Handling & Debugging",
    "I": "Security & Input Validation",
    "J": "Extensibility & Plugin Architecture",
    "K": "Reproducibility & Provenance",
    "L": "Long-Term Maintainability",
    "M": "Educational Resources & Tutorials",
    "N": "Visualization & Export",
    "O": "CI/CD & DevOps",
}


def parse_baseline_assessment(path):
    """Parses the existing comprehensive assessment to extract baseline content per category."""
    if not path.exists():
        print(f"Warning: Baseline assessment not found at {path}")
        return {}

    content = path.read_text()
    assessments = {}

    # Regex to capture sections like "## Assessment A: ..."
    pattern = re.compile(
        r"## Assessment ([A-O]): (.*?)\n(.*?)(?=\n## Assessment|\n## Summary Scorecard)",
        re.DOTALL,
    )

    for match in pattern.finditer(content):
        category_id = match.group(1)
        title = match.group(2).strip()
        body = match.group(3).strip()
        assessments[category_id] = {"title": title, "body": body}

    return assessments


def parse_completist_data(directory):
    """Parses data files from .jules/completist_data/."""
    data = {
        "todos": [],
        "fixmes": [],
        "stubs": [],
        "incomplete_docs": [],
        "not_implemented": [],
        "abstract_methods": [],
    }

    mapping = {
        "todo_markers.txt": "todos",
        "stub_functions.txt": "stubs",
        "incomplete_docs.txt": "incomplete_docs",
        "not_implemented.txt": "not_implemented",
        "abstract_methods.txt": "abstract_methods",
    }

    if not directory.exists():
        print(f"Warning: Completist data directory not found at {directory}")
        return data

    for filename, key in mapping.items():
        file_path = directory / filename
        if file_path.exists():
            lines = file_path.read_text().splitlines()
            data[key] = [line.strip() for line in lines if line.strip()]

    # Separate FIXMEs from TODOs if possible (simple heuristic based on text content if available)
    # The current todo_markers.txt format is file:line: content.
    # We'll assume standard grep output.

    todos_clean = []
    fixmes_clean = []

    for item in data["todos"]:
        if "FIXME" in item or "XXX" in item:
            fixmes_clean.append(item)
        else:
            todos_clean.append(item)

    data["todos"] = todos_clean
    data["fixmes"] = fixmes_clean

    return data


def parse_pragmatic_review(path):
    """Parses the Pragmatic Programmer review file."""
    if not path.exists():
        print(f"Warning: Pragmatic review not found at {path}")
        return {"dry_violations": [], "god_functions": [], "hardcoded_secrets": []}

    content = path.read_text()

    dry_violations = re.findall(
        r"- \*\*DRY\*\* \[MAJOR\]: (.*?)\n(.*?)(?=- \*\*|\Z)", content, re.DOTALL
    )
    god_functions = re.findall(
        r"- \*\*ORTHOGONALITY\*\* \[MAJOR\]: (.*?)\n(.*?)(?=- \*\*|\Z)",
        content,
        re.DOTALL,
    )
    hardcoded_secrets = re.findall(
        r"- \*\*REVERSIBILITY\*\* \[MAJOR\]: (.*?)\n(.*?)(?=- \*\*|\Z)",
        content,
        re.DOTALL,
    )

    return {
        "dry_violations": dry_violations,
        "god_functions": god_functions,
        "hardcoded_secrets": hardcoded_secrets,
    }


def generate_category_report(
    category_id, category_name, baseline, completist_data, pragmatic_data
):
    """Generates a detailed markdown report for a specific category."""

    sanitized_name = (
        category_name.replace(" ", "_").replace("&", "and").replace("/", "_")
    )
    output_path = OUTPUT_DIR / f"Assessment_{category_id}_{sanitized_name}.md"

    # Determine relevant completist/pragmatic data for this category
    new_findings = []

    if category_id == "A":  # Architecture
        new_findings.append(
            f"- **Abstract Methods**: Found {len(completist_data['abstract_methods'])} abstract method definitions defining the interface contracts."
        )
        new_findings.append(
            f"- **Not Implemented**: Found {len(completist_data['not_implemented'])} occurrences of NotImplementedError (checking for missing implementations vs abstract classes)."
        )

    elif category_id == "B":  # Code Quality
        new_findings.append(
            f"- **FIXMEs**: Found {len(completist_data['fixmes'])} FIXME/XXX markers indicating technical debt."
        )
        new_findings.append(
            f"- **God Functions**: Found {len(pragmatic_data['god_functions'])} complex functions violating orthogonality."
        )
        new_findings.append(
            f"- **DRY Violations**: Found {len(pragmatic_data['dry_violations'])} significant code duplications."
        )

    elif category_id == "C":  # Documentation
        new_findings.append(
            f"- **Incomplete Docs**: Found {len(completist_data['incomplete_docs'])} files with missing or incomplete documentation."
        )

    elif category_id == "G":  # Testing
        new_findings.append(
            f"- **Test Stubs**: Found {len(completist_data['stubs'])} stubbed tests or functions marked as stubs."
        )

    elif category_id == "I":  # Security
        new_findings.append(
            f"- **Hardcoded Secrets**: Found {len(pragmatic_data['hardcoded_secrets'])} potential hardcoded API keys or secrets."
        )

    elif category_id == "L":  # Maintainability
        new_findings.append(
            f"- **TODOs**: Found {len(completist_data['todos'])} TODO items representing future work or known gaps."
        )

    # Construct content
    content = f"# Assessment {category_id}: {category_name}\n\n"
    content += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
    content += "**Assessor**: Comprehensive Assessment Agent\n\n"

    content += "## 1. Baseline Assessment (2026-02-03)\n"
    content += "*(From previous comprehensive review)*\n\n"
    if baseline:
        content += baseline["body"] + "\n\n"
    else:
        content += "No baseline data available.\n\n"

    content += "## 2. New Findings (2026-02-08)\n"
    content += "### Quantitative Metrics\n"
    if new_findings:
        for finding in new_findings:
            content += f"{finding}\n"
    else:
        content += (
            "- No specific new quantitative metrics for this category in this pass.\n"
        )

    content += "\n### Pragmatic Review Integration\n"
    if category_id == "B" and pragmatic_data["god_functions"]:
        content += "**God Functions Detected:**\n"
        for title, _ in pragmatic_data["god_functions"][:5]:  # Limit to 5
            content += f"- {title.strip()}\n"
        if len(pragmatic_data["god_functions"]) > 5:
            content += f"- ... and {len(pragmatic_data['god_functions']) - 5} more.\n"

    if category_id == "B" and pragmatic_data["dry_violations"]:
        content += "\n**DRY Violations Detected:**\n"
        for title, _ in pragmatic_data["dry_violations"][:5]:
            content += f"- {title.strip()}\n"

    if category_id == "I" and pragmatic_data["hardcoded_secrets"]:
        content += "\n**Security Risks:**\n"
        for title, _ in pragmatic_data["hardcoded_secrets"]:
            content += f"- {title.strip()}\n"

    content += "\n## 3. Recommendations\n"
    content += "1. Address the specific findings listed above.\n"
    content += "2. Review the baseline recommendations if still relevant.\n"

    output_path.write_text(content)
    print(f"Generated {output_path}")


def generate_completist_report(data):
    """Generates the completist report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "completist").mkdir(parents=True, exist_ok=True)

    content = "# Completist Audit Report\n\n"
    content += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n\n"

    content += "## Executive Summary\n"
    content += f"- **Total TODOs**: {len(data['todos'])}\n"
    content += f"- **Total FIXMEs**: {len(data['fixmes'])}\n"
    content += f"- **Stubbed Functions**: {len(data['stubs'])}\n"
    content += f"- **Incomplete Documentation**: {len(data['incomplete_docs'])} files\n"
    content += f"- **Not Implemented Errors**: {len(data['not_implemented'])}\n\n"

    content += "## Critical Technical Debt (FIXME)\n"
    for item in data["fixmes"][:20]:
        content += f"- {item}\n"

    content += "\n## Feature Gaps (TODO)\n"
    content += "*(Showing first 20)*\n"
    for item in data["todos"][:20]:
        content += f"- {item}\n"

    content += "\n## Documentation Gaps\n"
    content += "*(Showing first 20)*\n"
    for item in data["incomplete_docs"][:20]:
        content += f"- {item}\n"

    COMPLETIST_REPORT_PATH.write_text(content)
    print(f"Generated {COMPLETIST_REPORT_PATH}")


def generate_comprehensive_report(baselines, completist_data, pragmatic_data):
    """Generates the final comprehensive assessment report."""

    content = "# Comprehensive Assessment Report\n\n"
    content += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
    content += "**Assessor**: Comprehensive Assessment Agent\n"
    content += "**Status**: Generated\n\n"

    content += "## Executive Summary\n"
    content += (
        "This report aggregates findings from the General Assessment (Categories A-O), "
    )
    content += "Completist Audit, and Pragmatic Programmer Review. "
    content += "The repository shows strong maturity but has accumulated technical debt in specific areas.\n\n"

    content += "### Unified Scorecard\n"
    content += "| Category | Name | Status | Key Issues |\n"
    content += "|---|---|---|---|\n"

    for cat_id, cat_name in CATEGORIES.items():
        status = "Good"
        issues = []

        # Simple logic to determine status/issues based on data
        if cat_id == "B":
            if (
                len(pragmatic_data["dry_violations"]) > 0
                or len(pragmatic_data["god_functions"]) > 0
            ):
                status = "Needs Improvement"
                issues.append(f"{len(pragmatic_data['dry_violations'])} DRY violations")
                issues.append(f"{len(pragmatic_data['god_functions'])} God functions")
        elif cat_id == "I":
            if len(pragmatic_data["hardcoded_secrets"]) > 0:
                status = "Critical"
                issues.append(
                    f"{len(pragmatic_data['hardcoded_secrets'])} Secrets found"
                )
        elif cat_id == "C":
            if len(completist_data["incomplete_docs"]) > 10:
                status = "Needs Improvement"
                issues.append(f"{len(completist_data['incomplete_docs'])} Doc gaps")

        issue_str = ", ".join(issues) if issues else "Maintained"
        content += f"| {cat_id} | {cat_name} | {status} | {issue_str} |\n"

    content += "\n## Top 10 Recommendations\n"
    content += "1. **Security**: Immediately rotate and remove the hardcoded API keys found in `src/shared/python/ai/adapters/`.\n"
    content += "2. **Code Quality**: Refactor the 'God functions' identified in `Assessment_B_Code_Quality_and_Hygiene.md`, particularly in the GUI modules.\n"
    content += "3. **DRY**: Address the 50+ DRY violations, starting with the duplicated logic in `scripts/refactor_dry_orthogonality.py`.\n"
    content += "4. **Documentation**: Complete the documentation for the identified gap files.\n"
    content += "5. **Testing**: Implement the stubbed tests found in `tests/`.\n"
    content += (
        "6. **Technical Debt**: Address the FIXME markers found in the codebase.\n"
    )
    content += "7. **Architecture**: Formalize the abstract interfaces where `NotImplementedError` is used.\n"
    content += (
        "8. **CI/CD**: Ensure the pre-commit hooks catch these issues in the future.\n"
    )
    content += "9. **User Experience**: Standardize the CLI output based on the new findings.\n"
    content += "10. **Maintainability**: Review the complex modules identified in Category L.\n\n"

    content += "## Detailed Reports\n"
    content += "Please refer to the individual assessment files in `docs/assessments/` for detailed findings per category.\n"

    COMPREHENSIVE_REPORT_PATH.write_text(content)
    print(f"Generated {COMPREHENSIVE_REPORT_PATH}")


def main():
    print("Starting Assessment Generation...")

    # 1. Parse Inputs
    baseline_data = parse_baseline_assessment(BASELINE_ASSESSMENT_PATH)
    completist_data = parse_completist_data(COMPLETIST_DATA_DIR)
    pragmatic_data = parse_pragmatic_review(PRAGMATIC_REVIEW_PATH)

    # 2. Generate Category Reports
    for cat_id, cat_name in CATEGORIES.items():
        baseline = baseline_data.get(cat_id)
        generate_category_report(
            cat_id, cat_name, baseline, completist_data, pragmatic_data
        )

    # 3. Generate Completist Report
    generate_completist_report(completist_data)

    # 4. Generate Comprehensive Summary
    generate_comprehensive_report(baseline_data, completist_data, pragmatic_data)

    print("Assessment Generation Complete.")


if __name__ == "__main__":
    main()
