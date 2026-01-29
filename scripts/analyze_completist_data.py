import datetime
import json
import os
import re
from typing import Any

COMPLETIST_DIR = ".jules/completist_data"
REPORT_DIR = "docs/assessments/completist"
ISSUES_DIR = "docs/assessments/issues"

def parse_grep_output(filename: str) -> list[dict[str, Any]]:
    filepath = os.path.join(COMPLETIST_DIR, filename)
    results = []
    if not os.path.exists(filepath):
        return results

    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(':', 2)
            if len(parts) >= 3:
                file_path = parts[0]
                try:
                    lineno = int(parts[1])
                except ValueError:
                    continue # specific grep error or weird formatting
                content = parts[2]
                results.append({
                    "file": file_path,
                    "line": lineno,
                    "content": content.strip()
                })
    return results

def parse_stubs(filename: str) -> list[dict[str, Any]]:
    filepath = os.path.join(COMPLETIST_DIR, filename)
    results = []
    if not os.path.exists(filepath):
        return results

    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                loc = parts[0]
                func_name = parts[1]
                loc_parts = loc.split(':')
                if len(loc_parts) == 2:
                    results.append({
                        "file": loc_parts[0],
                        "line": int(loc_parts[1]),
                        "function": func_name
                    })
    return results

def is_abstract_method(filepath: str, error_line: int) -> bool:
    """Check if the NotImplementedError is inside an abstract method."""
    try:
        with open(filepath, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Scan backwards from error_line (0-based index)
        idx = error_line - 1
        if idx >= len(lines):
            return False

        # Look back up to 100 lines for 'def ' or '@abstractmethod'
        for i in range(idx, max(-1, idx - 100), -1):
            line = lines[i].strip()
            if "@abstractmethod" in line:
                return True
            if line.startswith("def "):
                # If we hit the function def and haven't seen abstractmethod yet,
                # check if the previous line has it (decorators)
                if i > 0 and "@abstractmethod" in lines[i-1]:
                    return True
                return False
    except Exception:
        pass
    return False

def determine_category(item: dict[str, Any], source: str) -> str:
    content = item.get("content", "").upper()
    file_path = item["file"]

    is_test = "test" in file_path.lower()

    if source == "not_implemented":
        if not file_path.endswith(".py"):
            return "Feature Gap" # Or ignore? NotImplementedError in non-python is likely text.

        # Check if abstract
        if "@abstractmethod" in content or "Abstract" in content:
            return "Abstract" # Not incomplete per se

        # Check context for abstract method
        if is_abstract_method(file_path, item["line"]):
            return "Abstract"

        # Check for false positives (regex, except blocks)
        if "RE.COMPILE" in content or "EXCEPT " in content:
            return "Technical Debt" # Handling code, not missing code

        return "Critical" if not is_test else "Technical Debt"

    if source == "todo":
        if "reports/" in file_path or ".github/" in file_path:
            return "Technical Debt" # Low priority
        if "scripts/analyze_completist_data.py" in file_path or "scripts/create_completist_issues.py" in file_path:
            return "Technical Debt" # Ignore self

        if "FIXME" in content:
            return "Technical Debt"

        # Use regex for HACK/TEMP to avoid "attempt" matching "TEMP"
        if re.search(r'\bHACK\b', content) or re.search(r'\bTEMP\b', content):
            return "Technical Debt"

        return "Feature Gap"

    if source == "stubs":
        if is_test:
            return "Technical Debt" # Empty test

        # Check if abstract
        if is_abstract_method(file_path, item["line"]):
             return "Abstract"

        # Per audit requirements: pass statements in non-test code are Critical Incomplete
        return "Critical"

    if source == "docs":
        return "Documentation Gap"

    return "Unknown"

def calculate_priority(item: dict[str, Any]) -> int:
    file_path = item["file"]

    # Impact
    impact = 1
    if "src/" in file_path or "engines/" in file_path:
        impact = 5
    elif "api/" in file_path:
        impact = 4
    elif "shared/" in file_path:
        impact = 4
    elif "tools/" in file_path:
        impact = 3
    elif "tests/" in file_path:
        impact = 2

    # Complexity (heuristic) - currently unused
    # complexity = 3

    return impact # For now, just impact based

def main():
    # Load Data
    todos = parse_grep_output("todo_markers.txt")
    not_impl = parse_grep_output("not_implemented.txt")
    abstracts = parse_grep_output("abstract_methods.txt")
    stubs = parse_stubs("stub_functions.txt")
    docs = parse_stubs("incomplete_docs.txt") # Same format

    # Analyze
    findings = []

    # Critical: NotImplementedError
    abstract_lines = {(a['file'], a['line']) for a in abstracts}

    for item in not_impl:
        if (item['file'], item['line']) in abstract_lines:
            continue
        # Double check content for abstractmethod decorator which might be on same line or nearby
        # But grep matched the line.
        if "@abstractmethod" in item["content"]:
            continue

        category = determine_category(item, "not_implemented")
        item["category"] = category
        item["priority"] = calculate_priority(item)
        item["source"] = "NotImplementedError"
        findings.append(item)

    # Feature Gaps / Tech Debt: TODOs
    for item in todos:
        category = determine_category(item, "todo")
        item["category"] = category
        item["priority"] = calculate_priority(item)
        item["source"] = "Comment"
        findings.append(item)

    # Stubs
    # Check content of stubs to see if they are 'pass' or '...'
    for item in stubs:
        # Simple heuristic: if it's in not_impl, skip (already handled as critical)
        # But we don't know the exact line of the NotImpl error vs the function def.
        # Usually function def is line X, error is X+1.
        # Let's just include them as Feature Gaps for now unless we upgrade them.
        # Check if it's a pass
        try:
            with open(item['file']) as f:
                lines = f.readlines()
                # Simple check of body
                if item['line'] < len(lines):
                    # The function def is at item['line'] (1-based)
                    # The body is following.
                    # This is getting complicated to parse without AST.
                    # Let's rely on find_stubs.py intent.
                    pass
        except Exception:
            pass

        category = determine_category(item, "stubs")
        item["category"] = category
        item["priority"] = calculate_priority(item)
        item["source"] = "Stub"
        findings.append(item)

    # Docs
    for item in docs:
        item["category"] = "Documentation Gap"
        item["priority"] = calculate_priority(item)
        item["source"] = "Missing Docstring"
        findings.append(item)

    # Categorize lists
    critical = [f for f in findings if f["category"] == "Critical"]
    feature_gaps = [f for f in findings if f["category"] == "Feature Gap"]
    tech_debt = [f for f in findings if f["category"] == "Technical Debt"]
    doc_gaps = [f for f in findings if f["category"] == "Documentation Gap"]

    # Sort Critical
    critical.sort(key=lambda x: x["priority"], reverse=True)

    # Generate Report
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    report_path = os.path.join(REPORT_DIR, f"Completist_Report_{today}.md")
    latest_path = os.path.join(REPORT_DIR, "COMPLETIST_LATEST.md")

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Completist Audit Report - {today}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("| Category | Count |\n")
        f.write("|----------|-------|\n")
        f.write(f"| Critical Incomplete | {len(critical)} |\n")
        f.write(f"| Feature Gaps | {len(feature_gaps)} |\n")
        f.write(f"| Technical Debt | {len(tech_debt)} |\n")
        f.write(f"| Documentation Gaps | {len(doc_gaps)} |\n\n")

        # Critical
        f.write("## Critical Incomplete (Blocking Features)\n\n")
        if critical:
            f.write("| Priority | File | Line | Issue |\n")
            f.write("|----------|------|------|-------|\n")
            for item in critical:
                link = f"[{os.path.basename(item['file'])}]({item['file']})"
                content = item.get("content", item.get("function", "Incomplete"))
                f.write(f"| {item['priority']} | {link} | {item['line']} | `{content}` |\n")
        else:
            f.write("No critical blocking issues found.\n")
        f.write("\n")

        # Feature Gaps (Grouped)
        f.write("## Feature Gaps\n\n")
        # Group by directory
        grouped_gaps = {}
        for item in feature_gaps:
            folder = os.path.dirname(item['file'])
            if folder not in grouped_gaps:
                grouped_gaps[folder] = []
            grouped_gaps[folder].append(item)

        for folder in sorted(grouped_gaps.keys()):
            f.write(f"### {folder}\n")
            for item in grouped_gaps[folder]:
                content = item.get("content", item.get("function", ""))
                f.write(f"- {os.path.basename(item['file'])}:{item['line']} - {content}\n")
            f.write("\n")

        # Technical Debt
        f.write("## Technical Debt Register\n\n")
        for item in tech_debt:
             content = item.get("content", item.get("function", ""))
             f.write(f"- {item['file']}:{item['line']} - `{content}`\n")
        f.write("\n")

        # Recommendations
        f.write("## Recommended Implementation Order\n\n")
        f.write("1. Address Critical Incomplete items in `src/`.\n")
        f.write("2. Review Feature Gaps in core engines.\n")
        f.write("3. Address Technical Debt in shared utilities.\n")

    # Update Latest
    with open(latest_path, 'w', encoding='utf-8') as f:
        with open(report_path, encoding='utf-8') as src:
            f.write(src.read())

    # Output Critical Items JSON for tools
    critical_json_path = os.path.join(COMPLETIST_DIR, "critical_items.json")
    with open(critical_json_path, 'w', encoding='utf-8') as f:
        json.dump(critical, f, indent=2)

    print(f"Report generated at {report_path}")
    print(f"Critical items saved to {critical_json_path}")

if __name__ == "__main__":
    main()
