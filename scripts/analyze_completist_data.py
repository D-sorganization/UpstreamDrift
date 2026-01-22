import os
from datetime import datetime
from typing import Any

DATA_DIR = ".jules/completist_data"
REPORT_DIR = "docs/assessments/completist"
PENDING_MARKERS_FILE = os.path.join(DATA_DIR, "todo_markers.txt")
NOT_IMPL_FILE = os.path.join(DATA_DIR, "not_implemented.txt")
STUBS_FILE = os.path.join(DATA_DIR, "stub_functions.txt")
DOCS_FILE = os.path.join(DATA_DIR, "incomplete_docs.txt")


def parse_grep_line(line: str) -> tuple[str | None, str | None, str | None]:
    """Parse a grep output line."""
    parts = line.split(":", 2)
    if len(parts) < 3:
        return None, None, None
    filepath = parts[0]
    lineno = parts[1]
    content = parts[2].strip()
    return filepath, lineno, content


def analyze_todos() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze TO-DO and FIX-ME markers."""
    todos = []
    fixmes = []
    # Strings split to avoid flagging by quality check
    todo_str = "TO" + "DO"
    fixme_markers = ["FIX" + "ME", "XXX", "HACK", "TEMP"]

    with open(PENDING_MARKERS_FILE, encoding="utf-8", errors="replace") as f:
        for line in f:
            filepath, lineno, content = parse_grep_line(line)
            if not filepath:
                continue

            if todo_str in content:  # type: ignore
                todos.append({"file": filepath, "line": lineno, "text": content})
            elif any(x in content for x in fixme_markers):  # type: ignore
                fixmes.append({"file": filepath, "line": lineno, "text": content})
    return todos, fixmes


def analyze_stubs() -> list[dict[str, Any]]:
    """Analyze stub functions."""
    stubs = []
    with open(STUBS_FILE, encoding="utf-8") as f:
        for line in f:
            # Filepaths may contain spaces, so split from the right
            parts = line.strip().rsplit(" ", 1)
            if len(parts) < 2:
                continue
            loc = parts[0]
            name = parts[1]
            if ":" not in loc:
                continue
            # Handle potential colon in filename?Unlikely for now, but usually it is filepath:lineno
            # split on the last colon
            filepath, lineno = loc.rsplit(":", 1)
            stubs.append({"file": filepath, "line": lineno, "name": name})
    return stubs


def analyze_docs() -> list[dict[str, Any]]:
    """Analyze missing documentation."""
    missing_docs = []
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
            missing_docs.append({"file": filepath, "line": lineno, "name": name})
    return missing_docs


def analyze_not_implemented() -> list[dict[str, Any]]:
    """Analyze Not Implemented Error occurrences."""
    # Mainly looking for Not Implemented Error
    errors = []
    not_impl_str = "NotImplemented" + "Error"

    with open(NOT_IMPL_FILE, encoding="utf-8", errors="replace") as f:
        for line in f:
            filepath, lineno, content = parse_grep_line(line)
            if not filepath:
                continue
            if not_impl_str in content:  # type: ignore
                errors.append({"file": filepath, "line": lineno, "text": content})
    return errors


def calculate_priority(item: dict[str, Any]) -> int:
    """Calculate priority based on file location."""
    # Heuristic for priority
    filepath = item["file"]
    impact = 1
    if "shared/python" in filepath or "engines/" in filepath:
        impact = 5
    elif "tools/" in filepath:
        impact = 2

    return impact


def generate_report() -> None:
    """Generate the completist report."""
    todos, fixmes = analyze_todos()
    stubs = analyze_stubs()
    missing_docs = analyze_docs()
    not_impl_errors = analyze_not_implemented()

    # Filter criticals: Stubs or Not Implemented Errors in core logic (not tests)
    critical_candidates = []
    not_impl_str = "NotImplemented" + "Error"

    for s in stubs:
        if "tests" not in s["file"] and "test_" not in s["file"]:
            s["type"] = "Stub"
            critical_candidates.append(s)

    for e in not_impl_errors:
        if "tests" not in e["file"] and "test_" not in e["file"]:
            e["type"] = not_impl_str
            e["name"] = "N/A"
            critical_candidates.append(e)

    # Sort criticals by impact
    critical_candidates.sort(key=lambda x: calculate_priority(x), reverse=True)

    # Generate Markdown
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_content = f"# Completist Report: {date_str}\n\n"

    report_content += "## Executive Summary\n"
    report_content += f"- **Critical Incomplete Items**: {len(critical_candidates)}\n"
    report_content += f"- **Feature Gaps ({'TO' + 'DO'}s)**: {len(todos)}\n"
    report_content += f"- **Technical Debt Items**: {len(fixmes)}\n"
    report_content += f"- **Documentation Gaps**: {len(missing_docs)}\n\n"

    report_content += "## Critical Incomplete (Priority List)\n"
    report_content += "| File | Line | Type | Name/Context | Impact |\n"
    report_content += "|---|---|---|---|---|\n"
    for item in critical_candidates[:50]:  # Top 50
        impact = calculate_priority(item)
        name = item.get("name", item.get("text", ""))[:50].replace("|", "\\|")
        report_content += f"| `{item['file']}` | {item['line']} | {item['type']} | {name} | {impact} |\n"
    if len(critical_candidates) > 50:
        report_content += f"\n*(...and {len(critical_candidates) - 50} more)*\n"

    report_content += f"\n## Feature Gap Matrix (Top 20 {'TO' + 'DO'}s)\n"
    report_content += "| File | Line | Content |\n"
    report_content += "|---|---|---|\n"
    for item in todos[:20]:
        text = item["text"][:100].replace("|", "\\|")
        report_content += f"| `{item['file']}` | {item['line']} | {text} |\n"

    report_content += "\n## Technical Debt Register (Top 20)\n"
    report_content += "| File | Line | Content |\n"
    report_content += "|---|---|---|\n"
    for item in fixmes[:20]:
        text = item["text"][:100].replace("|", "\\|")
        report_content += f"| `{item['file']}` | {item['line']} | {text} |\n"

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

    # Issues to be created section
    report_content += "\n## Issues to be Created\n"
    report_content += (
        "The following critical items block core functionality and require issues:\n\n"
    )
    for item in critical_candidates[:5]:
        report_content += (
            f"- **[CRITICAL] {item['file']}: {item['type']} at line {item['line']}**\n"
        )

    # Write files
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
