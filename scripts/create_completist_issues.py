import datetime
import json
import os

CRITICAL_ITEMS_PATH = ".jules/completist_data/critical_items.json"
ISSUES_DIR = "docs/assessments/issues"


def main():
    if not os.path.exists(CRITICAL_ITEMS_PATH):
        print(f"No critical items file found at {CRITICAL_ITEMS_PATH}")
        return

    with open(CRITICAL_ITEMS_PATH) as f:
        items = json.load(f)

    if not items:
        print("No critical items to process.")
        return

    # Group by file
    grouped = {}
    for item in items:
        filepath = item["file"]
        if filepath not in grouped:
            grouped[filepath] = []
        grouped[filepath].append(item)

    os.makedirs(ISSUES_DIR, exist_ok=True)

    count = 0
    for filepath, file_items in grouped.items():
        filename = os.path.basename(filepath)
        safe_filename = filename.replace(".", "_")
        issue_filename = f"ISSUE_COMPLETIST_Critical_{safe_filename}.md"
        issue_path = os.path.join(ISSUES_DIR, issue_filename)

        # Determine module/component
        module = "Unknown"
        if "src/engines" in filepath:
            module = "Physics Engine"
        elif "src/shared" in filepath:
            module = "Shared Utility"
        elif "src/api" in filepath:
            module = "API"
        elif "src/tools" in filepath:
            module = "Tools"

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        content = f"""# Critical Incomplete Implementation in {filename}

**Status**: Open
**Priority**: Critical
**Created**: {today}
**Module**: {module}
**File**: `{filepath}`

## Description
The following critical incomplete implementations (stubs, placeholders, or NotImplementedError) were identified in `{filename}`. These items block core functionality or represent significant technical debt in non-test code.

## Findings

| Line | Function/Context | Issue Type |
|------|------------------|------------|
"""
        for item in file_items:
            ctx = item.get("function", item.get("content", "Unknown"))
            source = item.get("source", "Unknown")
            content += f"| {item['line']} | `{ctx}` | {source} |\n"

        content += """
## Impact
- **Blocking**: Features relying on these functions will fail or return incomplete data.
- **Stability**: Potential for runtime errors (NotImplementedError) or silent failures (pass).

## Recommended Action
1. Review the listed stubs.
2. Implement the missing logic or remove the function if unused.
3. If the function is an interface definition, ensure it is marked as abstract or uses `Protocol`.

## References
- Completist Audit Report ({today})
"""

        with open(issue_path, "w") as f:
            f.write(content)

        print(f"Created issue: {issue_path}")
        count += 1

    print(f"Total issues created: {count}")


if __name__ == "__main__":
    main()
