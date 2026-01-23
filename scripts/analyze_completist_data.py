import os
import re
from datetime import datetime
from collections import defaultdict
from typing import Any, TypedDict, cast
from collections.abc import Mapping

DATA_DIR = ".jules/completist_data"
REPORT_DIR = "docs/assessments/completist"

# Input files
MARKERS_FILE = os.path.join(DATA_DIR, "todo_markers.txt")
NOT_IMPL_FILE = os.path.join(DATA_DIR, "not_implemented.txt")
STUBS_FILE = os.path.join(DATA_DIR, "stub_functions.txt")
DOCS_FILE = os.path.join(DATA_DIR, "incomplete_docs.txt")
ABSTRACT_FILE = os.path.join(DATA_DIR, "abstract_methods.txt")

EXCLUDED_PATHS = [
    "scripts/analyze_completist_data.py",
    "docs/",
    ".github/",
    ".jules/",
    "CRITICAL_PROJECT_REVIEW.md",
    "WORKFLOW_AND_AGENTS_REPORT.md",
    "pyproject.toml"
]

class Finding(TypedDict):
    file: str
    line: str
    text: str
    name: str | None # Optional
    type: str | None # Optional

def is_excluded(filepath: str) -> bool:
    """Check if filepath should be excluded from analysis."""
    if not filepath:
        return True
    # Normalize path
    fp = filepath.replace("\\", "/")
    if fp.startswith("./"):
        fp = fp[2:]

    for excl in EXCLUDED_PATHS:
        if fp.startswith(excl) or excl in fp:
            return True
    return False

def parse_grep_line(line: str) -> tuple[str | None, str | None, str | None]:
    """Parse a grep output line."""
    parts = line.split(":", 2)
    if len(parts) < 3:
        return None, None, None
    filepath = parts[0]
    lineno = parts[1]
    content = parts[2].strip()
    return filepath, lineno, content

def get_module_from_path(filepath: str) -> str:
    """Extract a module name from a filepath."""
    filepath = filepath.replace("\\", "/")
    parts = filepath.split("/")

    if parts[0] == ".":
        parts = parts[1:]

    if len(parts) > 2:
        return "/".join(parts[:3])
    elif len(parts) > 0:
        return parts[0]
    return "root"

def analyze_todos() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze TO-DO and FIX-ME markers."""
    todos: list[dict[str, Any]] = []
    fixmes: list[dict[str, Any]] = []
    todo_str = "TO" + "DO"
    fixme_markers = ["FIX" + "ME", "XXX", "HACK", "TEMP"]

    if os.path.exists(MARKERS_FILE):
        with open(MARKERS_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                filepath, lineno, content = parse_grep_line(line)
                if not filepath or not lineno or content is None:
                    continue

                if is_excluded(filepath):
                    continue

                if todo_str in content:
                    todos.append({"file": filepath, "line": lineno, "text": content, "type": "TODO"})
                elif any(x in content for x in fixme_markers):
                    # Identify specific marker
                    marker = "FIXME"
                    for m in fixme_markers:
                        if m in content:
                            marker = m
                            break
                    fixmes.append({"file": filepath, "line": lineno, "text": content, "type": marker})
    return todos, fixmes

def analyze_stubs() -> list[dict[str, Any]]:
    """Analyze stub functions."""
    stubs: list[dict[str, Any]] = []
    if os.path.exists(STUBS_FILE):
        with open(STUBS_FILE, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit(" ", 1)
                if len(parts) < 2:
                    continue
                loc = parts[0]
                name = parts[1]
                if ":" not in loc:
                    continue
                filepath, lineno = loc.rsplit(":", 1)

                if is_excluded(filepath):
                    continue

                stubs.append({"file": filepath, "line": lineno, "name": name, "type": "Stub"})
    return stubs

def analyze_docs() -> list[dict[str, Any]]:
    """Analyze missing documentation."""
    missing_docs: list[dict[str, Any]] = []
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit(" ", 1)
                if len(parts) < 2:
                    continue
                loc = parts[0]
                name = parts[1]
                if ":" not in loc:
                    continue
                filepath, lineno = loc.rsplit(":", 1)

                if is_excluded(filepath):
                    continue

                missing_docs.append({"file": filepath, "line": lineno, "name": name, "type": "DocGap"})
    return missing_docs

def analyze_not_implemented() -> list[dict[str, Any]]:
    """Analyze Not Implemented Error occurrences."""
    errors: list[dict[str, Any]] = []
    not_impl_str = "NotImplemented" + "Error"

    if os.path.exists(NOT_IMPL_FILE):
        with open(NOT_IMPL_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                filepath, lineno, content = parse_grep_line(line)
                if not filepath or not lineno or content is None:
                    continue

                if is_excluded(filepath):
                    continue

                if not_impl_str in content:
                    errors.append({"file": filepath, "line": lineno, "text": content, "type": not_impl_str})
    return errors

def analyze_abstract_methods() -> list[dict[str, Any]]:
    """Analyze Abstract Methods."""
    abstracts: list[dict[str, Any]] = []
    if os.path.exists(ABSTRACT_FILE):
        with open(ABSTRACT_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                filepath, lineno, content = parse_grep_line(line)
                if not filepath or not lineno or content is None:
                    continue

                if is_excluded(filepath):
                    continue

                if "@abstractmethod" in content:
                    abstracts.append({"file": filepath, "line": lineno, "text": content, "type": "Abstract"})
    return abstracts

def calculate_metrics(item: Mapping[str, Any]) -> tuple[int, int, int]:
    """Calculate User Impact, Test Coverage, Complexity."""
    filepath = cast(str, item["file"])
    item_type = cast(str, item.get("type", ""))

    # Impact Heuristic (1-5)
    impact = 1
    if "shared/python" in filepath or "engines/" in filepath or "api/" in filepath:
        impact = 5
    elif "tools/" in filepath:
        impact = 3
    elif "tests/" in filepath:
        impact = 1

    # Test Coverage Heuristic (1-5)
    coverage = 1
    if "tests/" in filepath:
        coverage = 5
    elif "shared/python" in filepath:
        coverage = 3
    else:
        coverage = 2

    # Complexity Heuristic (1-5)
    complexity = 3
    if "Stub" in item_type or "NotImplemented" in item_type:
        complexity = 4 # Harder to fix, missing logic
    elif "FIXME" in item_type:
        complexity = 2 # Usually a correction/refactor
    elif "TODO" in item_type:
        complexity = 3 # New feature or logic
    elif "DocGap" in item_type:
        complexity = 1 # Just documentation
    elif "Abstract" in item_type:
        complexity = 5 # Needs implementation in subclasses

    return impact, coverage, complexity

def generate_report() -> None:
    """Generate the completist report."""
    todos, fixmes = analyze_todos()
    stubs = analyze_stubs()
    missing_docs = analyze_docs()
    not_impl_errors = analyze_not_implemented()
    abstract_methods = analyze_abstract_methods()

    # Filter criticals (Stubs and NotImplemented in core code)
    critical_candidates = []
    not_impl_str = "NotImplemented" + "Error"

    for s in stubs:
        if "tests" not in s["file"] and "test_" not in s["file"]:
            critical_candidates.append(s)

    for e in not_impl_errors:
        if "tests" not in e["file"] and "test_" not in e["file"]:
            critical_candidates.append(e)

    # Sort criticals by Impact desc, then Complexity asc (fix easiest high impact first?) or desc?
    # Usually priority = Impact * Urgency. Here just Impact.
    critical_candidates.sort(key=lambda x: calculate_metrics(x)[0], reverse=True)

    # Group TODOs
    module_todos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in todos:
        mod = get_module_from_path(cast(str, t["file"]))
        module_todos[mod].append(t)

    # Generate Markdown
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_content = f"# Completist Report: {date_str}\n\n"

    report_content += "## Executive Summary\n"
    report_content += f"- **Critical Incomplete Items**: {len(critical_candidates)}\n"
    report_content += f"- **Feature Gaps ({'TO' + 'DO'}s)**: {len(todos)}\n"
    report_content += f"- **Technical Debt Items**: {len(fixmes)}\n"
    report_content += f"- **Documentation Gaps**: {len(missing_docs)}\n"
    report_content += f"- **Abstract Methods**: {len(abstract_methods)}\n\n"

    report_content += "## Critical Incomplete (Priority List)\n"
    report_content += "| File | Line | Type | User Impact | Test Coverage | Complexity |\n"
    report_content += "|---|---|---|---|---|---|\n"
    for item in critical_candidates[:50]:
        impact, coverage, complexity = calculate_metrics(item)
        name = cast(str, item.get("name", item.get("text", "")))[:40].replace("|", "\\|")
        file_p = item["file"]
        line_p = item["line"]
        type_p = item.get("type", "")
        report_content += f"| `{file_p}` | {line_p} | {type_p} | {impact} | {coverage} | {complexity} |\n"
    if len(critical_candidates) > 50:
        report_content += f"\n*(...and {len(critical_candidates) - 50} more)*\n"

    report_content += f"\n## Feature Gap Matrix (Module -> Missing Features)\n"

    sorted_modules = sorted(module_todos.items(), key=lambda x: len(x[1]), reverse=True)

    for mod, items in sorted_modules[:10]:
        report_content += f"\n### {mod} ({len(items)} items)\n"
        report_content += "| Line | Content |\n"
        report_content += "|---|---|\n"
        for item in items[:5]:
            text = cast(str, item["text"])[:80].replace("|", "\\|")
            report_content += f"| {item['line']} | {text} |\n"
        if len(items) > 5:
             report_content += f"| ... | *({len(items)-5} more)* |\n"

    report_content += "\n## Technical Debt Register (Top 20)\n"
    report_content += "| File | Line | Content | Type |\n"
    report_content += "|---|---|---|---|\n"
    for item in fixmes[:20]:
        text = cast(str, item["text"])[:100].replace("|", "\\|")
        type_p = item.get("type", "FIXME")
        report_content += f"| `{item['file']}` | {item['line']} | {text} | {type_p} |\n"

    report_content += "\n## Documentation Gaps (Top 20)\n"
    report_content += "| File | Line | Symbol |\n"
    report_content += "|---|---|---|\n"
    for item in missing_docs[:20]:
        report_content += f"| `{item['file']}` | {item['line']} | {item['name']} |\n"

    report_content += "\n## Recommended Implementation Order\n"
    report_content += (
        "1. Address Critical Incomplete items in `shared/python` and `engines/`.\n"
    )
    report_content += (
        f"2. Fill in missing features marked with {'TO' + 'DO'} in core logic.\n"
    )
    report_content += (
        f"3. Resolve Technical Debt ({'FIX' + 'ME'}) to ensure stability.\n"
    )
    report_content += "4. Add docstrings to public interfaces.\n"

    report_content += "\n## Issues to be Created\n"
    report_content += (
        "The following critical items block core functionality and require issues with label 'incomplete-implementation,critical':\n\n"
    )

    # Filter criticals for Issue creation (Impact >= 4)
    issues_to_create = []
    for item in critical_candidates:
        impact, _, _ = calculate_metrics(item)
        if impact >= 4:
            issues_to_create.append(item)

    if not issues_to_create:
        report_content += "No critical issues found needing immediate creation.\n"
    else:
        report_content += "Run the following commands to create issues (if configured):\n\n"
        report_content += "```bash\n"
        for item in issues_to_create[:10]:
            item_type = item.get("type", "Issue")
            file_p = item["file"]
            line_p = item["line"]
            name = cast(str, item.get("name", item.get("text", "")))[:50].strip()
            title = f"[Incomplete] {item_type} in {file_p}:{line_p}"
            body = f"Found {item_type} in {file_p} at line {line_p}.\\nContext: {name}"
            # Escaping for bash
            title_esc = title.replace('"', '\\"')
            body_esc = body.replace('"', '\\"')
            report_content += f'gh issue create --title "{title_esc}" --body "{body_esc}" --label "incomplete-implementation,critical"\n'
        report_content += "```\n"
        if len(issues_to_create) > 10:
             report_content += f"\n*(...and {len(issues_to_create) - 10} more)*\n"

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_filename = f"Completist_Report_{date_str}.md"
    report_path = os.path.join(REPORT_DIR, report_filename)
    latest_path = os.path.join(REPORT_DIR, "COMPLETIST_LATEST.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Reports generated: {report_path}, {latest_path}")

if __name__ == "__main__":
    generate_report()
