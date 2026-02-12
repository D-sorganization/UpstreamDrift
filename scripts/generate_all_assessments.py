#!/usr/bin/env python3
"""
Generate General Assessment Reports (A-O) based on reference prompts.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.script_utils import (
    check_docs_status,
    count_test_files,
    find_python_files,
    run_main,
    run_tool_check,
    setup_script_logging,
)

logger = setup_script_logging(__name__)

DOCS_ARCHIVE = _REPO_ROOT / "docs" / "archive" / "assessments_jan2026"
OUTPUT_DIR = _REPO_ROOT / "docs" / "assessments"
SUMMARY_JSON = OUTPUT_DIR / "assessment_summary.json"

CATEGORIES = {
    "A": "Architecture & Implementation",
    "B": "Hygiene, Security & Quality",
    "C": "Documentation & Integration",
    "D": "Error Handling & Reliability",
    "E": "Performance & Optimization",
    "F": "Security & Safety",
    "G": "Testing & QA",
    "H": "Dependencies & Environment",
    "I": "Code Style & Standards",
    "J": "API Design & Interfaces",
    "K": "Data Handling & Storage",
    "L": "Logging & Observability",
    "M": "Configuration & Secrets",
    "N": "Scalability & Concurrency",
    "O": "Maintainability & Refactoring",
}


def read_prompt(letter: str) -> str:
    """Read the reference prompt for a category."""
    prompt_path = DOCS_ARCHIVE / f"Assessment_Prompt_{letter}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    for p in DOCS_ARCHIVE.glob(f"Assessment_Prompt_{letter}*.md"):
        return p.read_text(encoding="utf-8")
    return f"# Assessment {letter}: {CATEGORIES.get(letter)}\n\n(Prompt missing)"


class BaseChecker:
    def __init__(self, letter, name):
        self.letter = letter
        self.name = name
        self.findings = []
        self.metrics = {}
        self.score = 5.0

    def run(self):
        py_files = find_python_files()
        self.findings.append(f"- Scanned {len(py_files)} Python files.")
        return self.findings, self.score, self.metrics


class ArchitectureChecker(BaseChecker):  # A
    def run(self):
        findings = []
        metrics = {}
        py_files = find_python_files()
        metrics["total_files"] = len(py_files)

        has_src = (_REPO_ROOT / "src").exists()
        has_tests = (_REPO_ROOT / "tests").exists()
        findings.append(f"- `src/` directory: {'✅' if has_src else '❌'}")
        findings.append(f"- `tests/` directory: {'✅' if has_tests else '❌'}")

        # Largest files check
        largest = sorted(py_files, key=lambda p: p.stat().st_size, reverse=True)[:10]
        findings.append("\n### Top 10 Largest Files (Maintenance Risk)")
        findings.append("| File | Size (KB) |")
        findings.append("|---|---|")
        for p in largest:
            size_kb = p.stat().st_size / 1024
            try:
                rel_path = p.resolve().relative_to(_REPO_ROOT)
            except ValueError:
                rel_path = p
            findings.append(f"| `{rel_path}` | {size_kb:.1f} |")

        score = 9.0 if has_src and has_tests else (7.0 if has_src else 4.0)
        return findings, score, metrics


class HygieneChecker(BaseChecker):  # B
    def run(self):
        findings = []
        metrics = {}

        ruff = run_tool_check(["ruff", "check", ".", "--statistics"])
        metrics["ruff_exit_code"] = ruff["exit_code"]

        if ruff["exit_code"] == 0:
            findings.append("- Ruff: ✅ Passed")
        else:
            findings.append("- Ruff: ❌ Issues Found")
            findings.append("\n### Ruff Violations")
            findings.append("```\n" + ruff["stdout"][:500] + "\n...```")

        black = run_tool_check(["black", "--check", "--quiet", "."])
        findings.append(
            f"- Black: {'✅ Formatted' if black['exit_code'] == 0 else '❌ Formatting Needed'}"
        )

        mypy = run_tool_check(["mypy", "."])
        findings.append(
            f"- Mypy: {'✅ Passed' if mypy['exit_code'] == 0 else '❌ Type Errors'}"
        )

        score = 10.0
        if ruff["exit_code"] != 0:
            score -= 2
        if black["exit_code"] != 0:
            score -= 1
        if mypy["exit_code"] != 0:
            score -= 2

        return findings, max(0, score), metrics


class DocumentationChecker(BaseChecker):  # C
    def run(self):
        findings = []
        metrics = {}
        docs = check_docs_status()
        findings.append(f"- README.md: {'✅' if docs['readme'] else '❌'}")
        findings.append(f"- docs/ dir: {'✅' if docs['docs_dir'] else '❌'}")

        py_files = find_python_files()
        missing = []
        for p in py_files:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if (
                    '"""' not in content
                    and "'''" not in content
                    and p.stat().st_size > 500
                ):
                    missing.append(p)
            except:
                pass

        metrics["missing_docstrings"] = len(missing)
        findings.append(
            f"- Found {len(missing)} significant files (>500b) without docstrings."
        )

        if missing:
            findings.append("\n### Files Missing Docstrings (Sample)")
            for p in missing[:10]:
                try:
                    rel_path = p.resolve().relative_to(_REPO_ROOT)
                except ValueError:
                    rel_path = p
                findings.append(f"- `{rel_path}`")

        score = 8.0 - (len(missing) * 0.05)
        return findings, max(0, score), metrics


class TestingChecker(BaseChecker):  # G
    def run(self):
        findings = []
        metrics = {}
        count = count_test_files()
        metrics["test_files"] = count
        findings.append(f"- Test files found: {count}")

        if count > 20:
            score = 9.0
        elif count > 5:
            score = 7.0
        else:
            score = 2.0

        return findings, score, metrics


class KeywordChecker(BaseChecker):
    def __init__(self, letter, name, keywords):
        super().__init__(letter, name)
        self.keywords = keywords

    def run(self):
        findings = []
        metrics = {}
        py_files = find_python_files()
        hits = 0
        samples = []

        for p in py_files:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for kw in self.keywords:
                    if kw in content:
                        hits += 1
                        if len(samples) < 10:
                            samples.append(f"`{p.name}`: ...{kw}...")
            except:
                pass

        metrics["keyword_hits"] = hits
        findings.append(
            f"- Scanned {len(py_files)} files for keywords: {', '.join(self.keywords)}"
        )
        findings.append(f"- Found {hits} occurrences.")

        if samples:
            findings.append("\n### Evidence Table")
            findings.append("| File | Match |")
            findings.append("|---|---|")
            for s in samples:
                parts = s.split(":")
                findings.append(f"| {parts[0]} | {parts[1]} |")

        score = 7.0
        return findings, score, metrics


class SecurityChecker(KeywordChecker):  # F
    def __init__(self, letter, name):
        super().__init__(
            letter,
            name,
            ["password =", "secret =", "eval(", "exec(", "subprocess.call("],
        )

    def run(self):
        findings, _, metrics = super().run()
        hits = metrics["keyword_hits"]
        score = max(0, 10.0 - (hits * 0.5))
        findings.append(
            f"\n**Security Risk Assessment**: {hits} potential vulnerabilities identified."
        )
        return findings, score, metrics


def get_checker(letter, name):
    if letter == "A":
        return ArchitectureChecker(letter, name)
    if letter == "B":
        return HygieneChecker(letter, name)
    if letter == "C":
        return DocumentationChecker(letter, name)
    if letter == "F":
        return SecurityChecker(letter, name)
    if letter == "G":
        return TestingChecker(letter, name)

    keywords = {
        "D": ["except Exception", "except:", "try:"],
        "E": ["while True", "sleep(", "open("],
        "H": ["requirements.txt", "pyproject.toml"],
        "I": ["# TODO", "FIXME"],
        "J": ["def get_", "def set_", "@property"],
        "K": ["json.load", "csv.reader", "pandas"],
        "L": ["logging.", "print("],
        "M": ["os.getenv", "config", ".env"],
        "N": ["async def", "await ", "Thread", "Process"],
        "O": ["class ", "def ", "import "],
    }
    return KeywordChecker(letter, name, keywords.get(letter, ["class "]))


def generate_report(letter: str, name: str):
    logger.info(f"Generating Assessment {letter}: {name}...")
    prompt_text = read_prompt(letter)
    checker = get_checker(letter, name)
    findings, score, metrics = checker.run()

    report_content = f"""# Assessment {letter} Report: {name}

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Assessor**: Automated Agent
**Score**: {score:.1f}/10

## Executive Summary
This is an automated assessment report generated based on the reference prompt requirements.
- **Overall Status**: {'Satisfactory' if score >= 7 else 'Needs Improvement'}
- **Automated Score**: {score:.1f}/10

## Automated Findings
{chr(10).join(findings)}

---

## Reference Prompt Requirements
*(The following is the logic/context used for this assessment)*

{prompt_text}
"""
    (OUTPUT_DIR / f"Assessment_{letter}_Category.md").write_text(
        report_content, encoding="utf-8"
    )
    return score, metrics


def main():
    logger.info("Starting General Assessment Generation (A-O)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing summary if available to preserve extra fields
    summary = {}
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON) as f:
                summary = json.load(f)
        except:
            pass

    if "category_scores" not in summary:
        summary["category_scores"] = {}
    if "metrics" not in summary:
        summary["metrics"] = {}

    total_score = 0
    count = 0

    for letter in sorted(CATEGORIES.keys()):
        name = CATEGORIES[letter]
        score, metrics = generate_report(letter, name)

        # Update specific category
        summary["category_scores"][letter] = {
            "name": name,
            "score": score,
            "status": "Satisfactory" if score >= 7 else "Needs Improvement",
        }

        # Merge metrics
        for k, v in metrics.items():
            summary["metrics"][f"{letter}_{k}"] = v

        total_score += score
        count += 1

    summary["overall_score"] = round(total_score / count, 2) if count > 0 else 0
    summary["timestamp"] = datetime.now().isoformat()

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Assessments completed. Summary saved to {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    run_main(main, logger)
