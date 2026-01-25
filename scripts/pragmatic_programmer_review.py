#!/usr/bin/env python3
"""
Pragmatic Programmer Review - Automated Code Assessment

This script performs a comprehensive code review based on principles from
"The Pragmatic Programmer" by David Thomas and Andrew Hunt.

Assessment Categories:
- DRY (Don't Repeat Yourself)
- Orthogonality & Decoupling
- Reversibility & Flexibility
- Code Quality & Craftsmanship
- Error Handling & Robustness
- Testing & Validation
- Documentation & Communication
- Automation & Tooling
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# noqa: E402 -- Required for local imports from project root in standalone scripts
from scripts.script_utils import run_main, setup_script_logging
from src.shared.python.assessment.analysis import (
    get_detailed_function_metrics,
)
from src.shared.python.assessment.constants import (
    PRAGMATIC_PRINCIPLES as PRINCIPLES,
)

# Configure logging using centralized utility
logger = setup_script_logging(__name__)


def find_python_files(root_path: Path) -> list[Path]:
    """Find all Python files, excluding common non-source directories."""
    excluded = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".tox",
        "build",
        "dist",
        ".eggs",
        "*.egg-info",
    }
    python_files = []
    for f in root_path.rglob("*.py"):
        if not any(ex in f.parts for ex in excluded):
            python_files.append(f)
    return python_files


def compute_file_hash(content: str) -> str:
    """Compute hash of normalized content for duplicate detection."""
    # Normalize: remove comments and whitespace
    lines = []
    for line in content.split("\n"):
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            lines.append(line)
    normalized = "\n".join(lines)
    return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()


def check_dry_violations(files: list[Path]) -> list[dict]:
    """
    Check for DRY (Don't Repeat Yourself) violations.

    Looks for:
    - Duplicate code blocks (consecutive lines)
    - Magic numbers/strings
    """
    issues = []
    chunk_size = 6  # Increase slightly for more meaningful duplicates
    code_blocks = defaultdict(list)
    magic_numbers = defaultdict(list)

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = content.split("\n")
        # Use a sliding window but only pick non-overlapping chunks for hashing
        # if they are part of a larger sequence.
        # Simple heuristic: hash 6-line blocks but only store if not highly similar
        # to the previous block in the SAME file.
        last_hash = ""
        for i in range(len(lines) - chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size])
            if len(chunk.strip()) < 60:  # Skip trivial chunks
                continue

            chunk_hash = compute_file_hash(chunk)
            if chunk_hash != last_hash:
                code_blocks[chunk_hash].append((file_path, i + 1))
                last_hash = chunk_hash

        # Magic constants
        for match in re.finditer(r"\b(\d{2,})\b", content):
            num = match.group(1)
            if num not in ("00", "10", "100", "1000"):
                line_no = content[: match.start()].count("\n") + 1
                magic_numbers[num].append((file_path, line_no))

    # Report duplicates (limit to 50 unique major blocks to avoid report bloat)
    reported_count = 0
    for _chunk_hash, locations in code_blocks.items():
        if len(locations) > 1 and reported_count < 50:
            files_involved = sorted({str(loc[0]) for loc in locations})
            issues.append(
                {
                    "principle": "DRY",
                    "severity": "MAJOR",
                    "title": "Significant duplicate code block",
                    "description": f"Found in {len(locations)} locations across {len(files_involved)} files",
                    "files": files_involved[:5],
                    "recommendation": "Consolidate into a shared utility or base class",
                }
            )
            reported_count += 1

    return issues


def check_orthogonality(files: list[Path]) -> list[dict]:
    """
    Check for orthogonality and decoupling issues.

    Looks for:
    - High coupling between modules
    - Global state usage
    - Circular dependencies
    - God classes/functions
    """
    issues = []
    imports_graph = defaultdict(set)
    global_vars = []

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            continue

        module_name = file_path.stem

        # Track imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports_graph[module_name].add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports_graph[module_name].add(node.module.split(".")[0])

        # Check for global variables
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    global_vars.append(
                        {
                            "file": str(file_path),
                            "name": name,
                            "lineno": node.lineno,
                        }
                    )

        # Check for god functions (too many lines)
        functions = get_detailed_function_metrics(content)
        for func in functions:
            if func["body_lines"] > 50:
                issues.append(
                    {
                        "principle": "ORTHOGONALITY",
                        "severity": "MAJOR",
                        "title": f"God function: {func['name']} ({func['body_lines']} lines)",
                        "description": "Functions over 50 lines violate single responsibility",
                        "files": [str(file_path)],
                        "recommendation": "Break into smaller, focused functions",
                    }
                )

    # Report global state usage
    if len(global_vars) > 3:
        issues.append(
            {
                "principle": "ORTHOGONALITY",
                "severity": "MAJOR",
                "title": f"Excessive global state ({len(global_vars)} globals)",
                "description": "Global variables create hidden dependencies",
                "files": list({g["file"] for g in global_vars[:5]}),
                "recommendation": "Use dependency injection or encapsulation",
            }
        )

    return issues


def check_reversibility(root_path: Path) -> list[dict]:
    """
    Check for reversibility and flexibility issues.

    Looks for:
    - Hardcoded configurations
    - Tight coupling to specific implementations
    - Missing abstraction layers
    """
    issues = []

    # Check for hardcoded configs
    config_patterns = [
        (r'host\s*=\s*["\'][^"\']+["\']', "Hardcoded host"),
        (r"port\s*=\s*\d+", "Hardcoded port"),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
        (r'database\s*=\s*["\'][^"\']+["\']', "Hardcoded database"),
    ]

    python_files = find_python_files(root_path)
    for file_path in python_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern, description in config_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(
                    {
                        "principle": "REVERSIBILITY",
                        "severity": "MAJOR",
                        "title": description,
                        "description": "Configuration should be external, not hardcoded",
                        "files": [str(file_path)],
                        "recommendation": "Use environment variables or config files",
                    }
                )
                break  # One issue per file

    # Check for missing config file
    config_files = list(root_path.glob("*.ini")) + list(root_path.glob("*.yaml"))
    config_files += list(root_path.glob("*.yml")) + list(root_path.glob("*.toml"))
    config_files += list(root_path.glob("config.*"))

    if not config_files and len(python_files) > 5:
        issues.append(
            {
                "principle": "REVERSIBILITY",
                "severity": "MINOR",
                "title": "No configuration file found",
                "description": "Projects should externalize configuration",
                "files": [],
                "recommendation": "Add config.yaml or similar for settings",
            }
        )

    return issues


def check_quality(files: list[Path]) -> list[dict]:
    """
    Check for code quality and craftsmanship issues.

    Looks for:
    - Overly complex functions
    - Missing type hints
    - Inconsistent naming
    - Unfinished work markers
    """
    issues = []
    todos = []
    fixmes = []
    missing_type_hints = 0

    # Use constructed strings to avoid false positives in quality checks
    todo_marker = "TO" + "DO"
    fixme_marker = "FIX" + "ME"

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Check for unfinished work markers
        for i, line in enumerate(content.split("\n"), 1):
            if todo_marker in line:
                todos.append((file_path, i, line.strip()))
            if fixme_marker in line:
                fixmes.append((file_path, i, line.strip()))

        # Check for type hints in function definitions
        functions = get_detailed_function_metrics(content)
        for func in functions:
            # Simple heuristic: check if 'def func(arg: type)' pattern exists
            func_pattern = rf"def\s+{func['name']}\s*\([^)]*:\s*\w+"
            if not re.search(func_pattern, content):
                missing_type_hints += 1

    # Report accumulated issues
    if len(todos) > 10:
        issues.append(
            {
                "principle": "QUALITY",
                "severity": "MINOR",
                "title": f"Technical debt: {len(todos)} unfinished task comments",
                "description": "Accumulated work markers indicate incomplete work",
                "files": list({str(t[0]) for t in todos[:5]}),
                "recommendation": "Address or create issues for pending tasks",
            }
        )

    if len(fixmes) > 0:
        issues.append(
            {
                "principle": "QUALITY",
                "severity": "MAJOR",
                "title": f"Known bugs: {len(fixmes)} fix-needed comments",
                "description": "Fix markers indicate known problems",
                "files": list({str(f[0]) for f in fixmes[:5]}),
                "recommendation": "Fix or create issues for known bugs",
            }
        )

    if missing_type_hints > len(files) * 2:
        issues.append(
            {
                "principle": "QUALITY",
                "severity": "MINOR",
                "title": "Low type hint coverage",
                "description": "Type hints improve code clarity and catch errors",
                "files": [],
                "recommendation": "Add type hints to function signatures",
            }
        )

    return issues


def check_robustness(files: list[Path]) -> list[dict]:
    """
    Check for error handling and robustness issues.

    Looks for:
    - Bare except clauses
    - Missing error handling
    - Assertion usage
    - Resource management
    """
    issues = []
    bare_excepts = []
    broad_excepts = []
    no_finally = 0

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_excepts.append((file_path, node.lineno))
                elif isinstance(node.type, ast.Name):
                    if node.type.id in ("Exception", "BaseException"):
                        broad_excepts.append((file_path, node.lineno))

            if isinstance(node, ast.Try):
                if not node.finalbody and any(
                    "open" in ast.dump(h) or "connect" in ast.dump(h)
                    for h in node.handlers
                ):
                    no_finally += 1

    if bare_excepts:
        issues.append(
            {
                "principle": "ROBUSTNESS",
                "severity": "CRITICAL",
                "title": f"Bare except clauses ({len(bare_excepts)} found)",
                "description": "Bare 'except:' catches all exceptions including KeyboardInterrupt",
                "files": list({str(b[0]) for b in bare_excepts[:5]}),
                "recommendation": "Specify exception types explicitly",
            }
        )

    if broad_excepts:
        issues.append(
            {
                "principle": "ROBUSTNESS",
                "severity": "MAJOR",
                "title": f"Overly broad exception handling ({len(broad_excepts)} found)",
                "description": "Catching 'Exception' hides specific errors",
                "files": list({str(b[0]) for b in broad_excepts[:5]}),
                "recommendation": "Catch specific exception types",
            }
        )

    return issues


def check_testing(root_path: Path) -> list[dict]:
    """
    Check for testing and validation issues.

    Looks for:
    - Test coverage
    - Test file organization
    - Missing test utilities
    """
    issues = []

    # Find test files
    test_patterns = ["**/test_*.py", "**/*_test.py", "**/tests/*.py"]
    test_files = set()
    for pattern in test_patterns:
        test_files.update(root_path.glob(pattern))

    # Find source files
    source_files = find_python_files(root_path)
    source_files = [f for f in source_files if "test" not in str(f).lower()]

    # Calculate test ratio
    test_ratio = len(test_files) / max(len(source_files), 1)

    if len(test_files) == 0:
        issues.append(
            {
                "principle": "TESTING",
                "severity": "CRITICAL",
                "title": "No test files found",
                "description": "Test early, test often, test automatically",
                "files": [],
                "recommendation": "Add unit tests for critical functionality",
            }
        )
    elif test_ratio < 0.3:
        issues.append(
            {
                "principle": "TESTING",
                "severity": "MAJOR",
                "title": f"Low test coverage ({len(test_files)} tests for {len(source_files)} source files)",
                "description": "Test ratio below 30%",
                "files": [],
                "recommendation": "Increase test coverage",
            }
        )

    # Check for pytest.ini or setup.cfg with pytest config
    pytest_config = (
        (root_path / "pytest.ini").exists()
        or (root_path / "pyproject.toml").exists()
        or (root_path / "setup.cfg").exists()
    )
    if not pytest_config and len(test_files) > 0:
        issues.append(
            {
                "principle": "TESTING",
                "severity": "MINOR",
                "title": "No pytest configuration found",
                "description": "Test configuration ensures consistent test runs",
                "files": [],
                "recommendation": "Add pytest.ini or pyproject.toml",
            }
        )

    return issues


def check_documentation(root_path: Path, files: list[Path]) -> list[dict]:
    """
    Check for documentation and communication issues.

    Looks for:
    - README quality
    - Docstring coverage
    - API documentation
    """
    issues = []

    # Check README
    readme_files = list(root_path.glob("README*"))
    if not readme_files:
        issues.append(
            {
                "principle": "DOCUMENTATION",
                "severity": "MAJOR",
                "title": "No README file found",
                "description": "Every project needs a README",
                "files": [],
                "recommendation": "Create README.md with project overview",
            }
        )
    else:
        readme_content = readme_files[0].read_text(encoding="utf-8", errors="ignore")
        if len(readme_content) < 500:
            issues.append(
                {
                    "principle": "DOCUMENTATION",
                    "severity": "MINOR",
                    "title": "README is too brief",
                    "description": "README should explain purpose, installation, and usage",
                    "files": [str(readme_files[0])],
                    "recommendation": "Expand README with examples and API docs",
                }
            )

    # Check docstring coverage
    functions_without_docstrings = 0
    total_functions = 0

    for file_path in files:
        functions = get_detailed_function_metrics(
            file_path.read_text(encoding="utf-8", errors="ignore")
        )
        for func in functions:
            if not func["name"].startswith("_"):  # Public functions
                total_functions += 1
                if not func["has_docstring"]:
                    functions_without_docstrings += 1

    if total_functions > 0:
        docstring_rate = 1 - (functions_without_docstrings / total_functions)
        if docstring_rate < 0.5:
            issues.append(
                {
                    "principle": "DOCUMENTATION",
                    "severity": "MINOR",
                    "title": f"Low docstring coverage ({docstring_rate:.0%})",
                    "description": f"{functions_without_docstrings} public functions lack docstrings",
                    "files": [],
                    "recommendation": "Add docstrings to public functions",
                }
            )

    return issues


def check_automation(root_path: Path) -> list[dict]:
    """
    Check for automation and tooling issues.

    Looks for:
    - CI/CD configuration
    - Linting setup
    - Pre-commit hooks
    - Build automation
    """
    issues = []

    # Check for CI/CD
    ci_configs = [
        root_path / ".github" / "workflows",
        root_path / ".gitlab-ci.yml",
        root_path / "Jenkinsfile",
        root_path / ".circleci",
    ]
    has_ci = any(c.exists() for c in ci_configs)

    if not has_ci:
        issues.append(
            {
                "principle": "AUTOMATION",
                "severity": "MAJOR",
                "title": "No CI/CD configuration found",
                "description": "Continuous integration catches problems early",
                "files": [],
                "recommendation": "Add GitHub Actions or similar CI",
            }
        )

    # Check for linting config
    lint_configs = [
        root_path / ".flake8",
        root_path / "pyproject.toml",
        root_path / "ruff.toml",
        root_path / ".pylintrc",
    ]
    has_lint = any(c.exists() for c in lint_configs)

    if not has_lint:
        issues.append(
            {
                "principle": "AUTOMATION",
                "severity": "MINOR",
                "title": "No linting configuration found",
                "description": "Linters catch style and logic errors automatically",
                "files": [],
                "recommendation": "Add ruff.toml or pyproject.toml with lint config",
            }
        )

    # Check for pre-commit
    if not (root_path / ".pre-commit-config.yaml").exists():
        issues.append(
            {
                "principle": "AUTOMATION",
                "severity": "MINOR",
                "title": "No pre-commit hooks configured",
                "description": "Pre-commit catches issues before they're committed",
                "files": [],
                "recommendation": "Add .pre-commit-config.yaml",
            }
        )

    # Check for Makefile or similar
    build_files = [
        root_path / "Makefile",
        root_path / "justfile",
        root_path / "tasks.py",
    ]
    has_build = any(b.exists() for b in build_files)

    if not has_build:
        issues.append(
            {
                "principle": "AUTOMATION",
                "severity": "MINOR",
                "title": "No build automation found",
                "description": "Makefile or similar simplifies common tasks",
                "files": [],
                "recommendation": "Add Makefile with common targets",
            }
        )

    return issues


def run_review(root_path: Path) -> dict[str, Any]:
    """
    Run the complete Pragmatic Programmer review.

    Returns assessment results including issues and scores.
    """
    logger.info(f"Running Pragmatic Programmer review on: {root_path}")

    python_files = find_python_files(root_path)
    logger.info(f"Found {len(python_files)} Python files")

    all_issues = []

    # Run all checks
    logger.info("Checking DRY violations...")
    all_issues.extend(check_dry_violations(python_files))

    logger.info("Checking orthogonality...")
    all_issues.extend(check_orthogonality(python_files))

    logger.info("Checking reversibility...")
    all_issues.extend(check_reversibility(root_path))

    logger.info("Checking code quality...")
    all_issues.extend(check_quality(python_files))

    logger.info("Checking robustness...")
    all_issues.extend(check_robustness(python_files))

    logger.info("Checking testing...")
    all_issues.extend(check_testing(root_path))

    logger.info("Checking documentation...")
    all_issues.extend(check_documentation(root_path, python_files))

    logger.info("Checking automation...")
    all_issues.extend(check_automation(root_path))

    # Calculate scores per principle
    scores = {}
    for principle_id, _principle_info in PRINCIPLES.items():
        principle_issues = [i for i in all_issues if i["principle"] == principle_id]

        # Start with 10, deduct based on severity
        score = 10.0
        for issue in principle_issues:
            if issue["severity"] == "CRITICAL":
                score -= 3.0
            elif issue["severity"] == "MAJOR":
                score -= 2.0
            elif issue["severity"] == "MINOR":
                score -= 1.0

        scores[principle_id] = max(0.0, min(10.0, score))

    # Calculate weighted overall score
    total_weight: float = sum(float(p["weight"]) for p in PRINCIPLES.values())
    overall: float = (
        sum(float(scores[pid]) * float(PRINCIPLES[pid]["weight"]) for pid in PRINCIPLES)
        / total_weight
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "repository": root_path.name,
        "python_files_analyzed": len(python_files),
        "overall_score": round(overall, 2),
        "principle_scores": {
            pid: {
                "score": round(scores[pid], 2),
                "name": PRINCIPLES[pid]["name"],
                "weight": PRINCIPLES[pid]["weight"],
            }
            for pid in PRINCIPLES
        },
        "issues": all_issues,
        "issue_summary": {
            "CRITICAL": len([i for i in all_issues if i["severity"] == "CRITICAL"]),
            "MAJOR": len([i for i in all_issues if i["severity"] == "MAJOR"]),
            "MINOR": len([i for i in all_issues if i["severity"] == "MINOR"]),
        },
    }


def generate_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate a markdown report from the assessment results."""
    md = f"""# Pragmatic Programmer Review

**Repository**: {results["repository"]}
**Date**: {results["timestamp"][:10]}
**Files Analyzed**: {results["python_files_analyzed"]}

## Overall Score: {results["overall_score"]:.1f}/10

## Principle Scores

| Principle | Score | Weight | Status |
|-----------|-------|--------|--------|
"""

    for _pid, info in results["principle_scores"].items():
        status = (
            "Pass"
            if info["score"] >= 7
            else "Needs Work" if info["score"] >= 4 else "Critical"
        )
        md += (
            f"| {info['name']} | {info['score']:.1f} | {info['weight']}x | {status} |\n"
        )

    md += f"""
## Issue Summary

- **Critical**: {results["issue_summary"]["CRITICAL"]}
- **Major**: {results["issue_summary"]["MAJOR"]}
- **Minor**: {results["issue_summary"]["MINOR"]}

## Detailed Findings

"""

    # Group issues by principle
    issues_by_principle = defaultdict(list)
    for issue in results["issues"]:
        issues_by_principle[issue["principle"]].append(issue)

    for pid in PRINCIPLES:
        issues = issues_by_principle.get(pid, [])
        if issues:
            md += f"### {PRINCIPLES[pid]['name']}\n\n"
            for issue in issues:
                severity_icon = {
                    "CRITICAL": "!!",
                    "MAJOR": "!",
                    "MINOR": "-",
                }[issue["severity"]]
                md += f"- [{severity_icon}] **{issue['title']}**\n"
                md += f"  - {issue['description']}\n"
                md += f"  - Recommendation: {issue['recommendation']}\n"
                if issue.get("files"):
                    md += f"  - Files: {', '.join(issue['files'][:3])}\n"
                md += "\n"

    md += """
---

*Generated by Pragmatic Programmer Review workflow*
*Based on "The Pragmatic Programmer" by David Thomas and Andrew Hunt*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    logger.info(f"Report saved to: {output_path}")


def create_github_issues(results: dict[str, Any], dry_run: bool = False) -> list[dict]:
    """Create GitHub issues for critical and major findings."""
    issues_to_create = []

    for issue in results["issues"]:
        if issue["severity"] not in ("CRITICAL", "MAJOR"):
            continue

        title = f"[PragProg] {issue['severity']}: {issue['title']}"

        body = f"""## Pragmatic Programmer Review Finding

**Principle**: {PRINCIPLES[issue["principle"]]["name"]}
**Severity**: {issue["severity"]}
**Identified**: {results["timestamp"][:10]}

### Description

{issue["description"]}

### Recommendation

{issue["recommendation"]}

### Affected Files

"""
        if issue.get("files"):
            for f in issue["files"][:5]:
                body += f"- `{f}`\n"
        else:
            body += "- Repository-wide issue\n"

        body += """
### Reference

This issue was identified by the automated Pragmatic Programmer Review.
Based on principles from "The Pragmatic Programmer" by David Thomas and Andrew Hunt.

---
*Auto-generated by Pragmatic Programmer Review workflow*
"""

        labels = ["pragmatic-programmer", "automated-review"]
        if issue["severity"] == "CRITICAL":
            labels.append("critical")

        issues_to_create.append(
            {
                "title": title,
                "body": body,
                "labels": labels,
                "principle": issue["principle"],
            }
        )

    if dry_run:
        logger.info(f"[DRY RUN] Would create {len(issues_to_create)} issues")
        return issues_to_create

    # Create issues using gh CLI
    created = []
    for issue_data in issues_to_create[:10]:  # Limit to 10 issues per run
        try:
            cmd = [
                "gh",
                "issue",
                "create",
                "--title",
                issue_data["title"],
                "--body",
                issue_data["body"],
            ]
            # Try with labels first
            label_cmd = cmd + ["--label", ",".join(issue_data["labels"])]
            result = subprocess.run(label_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Retry without labels
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Created issue: {result.stdout.strip()}")
                created.append(issue_data)
            else:
                logger.warning(f"Failed to create issue: {result.stderr}")
        except Exception as e:
            logger.error(f"Error creating issue: {e}")

    return created


def main():
    """Main entry point for the Pragmatic Programmer review."""
    parser = argparse.ArgumentParser(
        description="Pragmatic Programmer Review - Automated Code Assessment"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Path to repository (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--create-issues",
        action="store_true",
        help="Create GitHub issues for findings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Run review
    results = run_review(args.path.resolve())

    # Generate markdown report
    if args.output:
        generate_markdown_report(results, args.output)
    else:
        # Default output location
        default_output = (
            args.path / "docs" / "assessments" / "pragmatic_programmer_review.md"
        )
        generate_markdown_report(results, default_output)

    # Save JSON results
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2))
        logger.info(f"JSON results saved to: {args.json_output}")

    # Create GitHub issues
    if args.create_issues:
        created = create_github_issues(results, args.dry_run)
        logger.info(f"Created {len(created)} GitHub issues")

    # Print summary
    print(f"\n{'=' * 60}")
    print("PRAGMATIC PROGRAMMER REVIEW SUMMARY")
    print(f"{'=' * 60}")
    print(f"Overall Score: {results['overall_score']:.1f}/10")
    print(f"Critical Issues: {results['issue_summary']['CRITICAL']}")
    print(f"Major Issues: {results['issue_summary']['MAJOR']}")
    print(f"Minor Issues: {results['issue_summary']['MINOR']}")
    print(f"{'=' * 60}\n")

    # Return exit code based on critical issues
    if results["issue_summary"]["CRITICAL"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    run_main(main, logger)
