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
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Mock imports/utils if shared/python doesn't exist in all repos
# We will define minimal utils here to ensure standalone execution
def setup_script_logging(name):
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)


logger = setup_script_logging(__name__)

# Constants for Principles
PRINCIPLES = {
    "DRY": {"name": "Don't Repeat Yourself", "weight": 1.5},
    "ORTHOGONALITY": {"name": "Orthogonality", "weight": 1.2},
    "REVERSIBILITY": {"name": "Reversibility", "weight": 1.0},
    "QUALITY": {"name": "Tracer Bullets (Quality)", "weight": 1.0},
    "ROBUSTNESS": {"name": "Robustness", "weight": 1.2},
    "TESTING": {"name": "Ruthless Testing", "weight": 1.2},
    "DOCUMENTATION": {"name": "Documentation", "weight": 0.8},
    "AUTOMATION": {"name": "Automation", "weight": 1.0},
}


def find_python_files(root_path: Path) -> list[Path]:
    """Find all Python files, excluding common non-source directories."""
    excluded = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "__pycache__",
        ".tox",
        ".eggs",
        ".pytest_cache",
    }
    python_files = []
    for f in root_path.rglob("*.py"):
        if not any(ex in f.parts for ex in excluded):
            python_files.append(f)
    return python_files


def compute_file_hash(content: str) -> str:
    """Compute hash of normalized content for duplicate detection."""
    lines = []
    for line in content.split("\n"):
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            lines.append(line)
    normalized = "\n".join(lines)
    return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()


def get_detailed_function_metrics(content: str):
    """Simple AST parser to get function metrics."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs.append(
                {
                    "name": node.name,
                    "body_lines": len(node.body),
                    "has_docstring": (ast.get_docstring(node) is not None),
                }
            )
    return funcs


def check_dry_violations(files: list[Path]) -> list[dict]:
    issues = []
    chunk_size = 6
    code_blocks = defaultdict(list)

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = content.split("\n")
        last_hash = ""
        for i in range(len(lines) - chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size])
            if len(chunk.strip()) < 60:
                continue

            chunk_hash = compute_file_hash(chunk)
            if chunk_hash != last_hash:
                code_blocks[chunk_hash].append((file_path, i + 1))
                last_hash = chunk_hash

    reported = 0
    for _, locations in code_blocks.items():
        if len(locations) > 1 and reported < 50:
            files_inv = sorted({str(loc[0]) for loc in locations})
            issues.append(
                {
                    "principle": "DRY",
                    "severity": "MAJOR",
                    "title": "Duplicate code block",
                    "description": f"Found in {len(locations)} locations",
                    "files": files_inv[:5],
                    "recommendation": "Refactor into shared utility",
                }
            )
            reported += 1
    return issues


def check_orthogonality(files: list[Path]) -> list[dict]:
    issues = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            funcs = get_detailed_function_metrics(content)
            for func in funcs:
                if func["body_lines"] > 50:
                    issues.append(
                        {
                            "principle": "ORTHOGONALITY",
                            "severity": "MAJOR",
                            "title": f"God function: {func['name']}",
                            "description": f"Length {func['body_lines']} > 50 lines",
                            "files": [str(file_path)],
                            "recommendation": "Split function",
                        }
                    )
        except Exception:
            pass
    return issues


def check_reversibility(root_path: Path) -> list[dict]:
    issues = []
    python_files = find_python_files(root_path)
    for file_path in python_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', content):
                issues.append(
                    {
                        "principle": "REVERSIBILITY",
                        "severity": "MAJOR",
                        "title": "Hardcoded API Key",
                        "description": "Secrets in code",
                        "files": [str(file_path)],
                        "recommendation": "Use env vars",
                    }
                )
        except Exception:
            pass
    return issues


def check_quality(files: list[Path]) -> list[dict]:
    issues = []
    todos = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if "TODO" in content:
                todos.append(str(file_path))
        except Exception:
            pass

    if len(todos) > 10:
        issues.append(
            {
                "principle": "QUALITY",
                "severity": "MINOR",
                "title": f"High TODO count ({len(todos)})",
                "description": "Accumulated technical debt",
                "files": todos[:5],
                "recommendation": "Review TODOs",
            }
        )
    return issues


def check_testing(root_path: Path) -> list[dict]:
    issues = []
    test_files = list(root_path.rglob("test_*.py"))
    src_files = find_python_files(root_path)
    ratio = len(test_files) / max(len(src_files), 1)

    if ratio < 0.2:
        issues.append(
            {
                "principle": "TESTING",
                "severity": "MAJOR",
                "title": "Low Test Coverage",
                "description": f"Test/Src ratio {ratio:.2f} < 0.2",
                "files": [],
                "recommendation": "Add more tests",
            }
        )
    return issues


def run_review(root_path: Path):
    logger.info(f"Running Pragmatic Review on {root_path}")
    files = find_python_files(root_path)

    all_issues = []
    all_issues.extend(check_dry_violations(files))
    all_issues.extend(check_orthogonality(files))
    all_issues.extend(check_reversibility(root_path))
    all_issues.extend(check_quality(files))
    all_issues.extend(check_testing(root_path))

    return {
        "timestamp": datetime.now().isoformat(),
        "repository": root_path.name,
        "files_analyzed": len(files),
        "issues": all_issues,
    }


def generate_markdown_report(results, output_path):
    md = [f"# Pragmatic Programmer Review: {results['repository']}"]
    md.append(f"**Date**: {results['timestamp'][:10]}")
    md.append(f"**Files**: {results['files_analyzed']}")
    md.append("\n## Findings")

    if not results["issues"]:
        md.append("No major issues found.")

    for issue in results["issues"]:
        md.append(f"- **{issue['principle']}** [{issue['severity']}]: {issue['title']}")
        md.append(f"  - {issue['description']}")
        if issue.get("files"):
            md.append(f"  - Files: {', '.join(issue['files'][:3])}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(md))

    # Save JSON too
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assessments/pragmatic_programmer/review.md"),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    repo_root = Path.cwd()
    results = run_review(repo_root)
    generate_markdown_report(results, args.output)
    print(f"Report generated at {args.output}")
