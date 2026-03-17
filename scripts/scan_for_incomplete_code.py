import os
import re
from os.path import join


def scan_for_incomplete_code(root_dir):
    patterns = {
        "TODO": re.compile(r"TODO"),
        "FIXME": re.compile(r"FIXME"),
        "NotImplementedError": re.compile(r"raise NotImplementedError"),
        "pass_block": re.compile(r"^\s*pass\s*$"),
    }

    results: dict[str, list[tuple[int, str, str]]] = {}

    for dirpath, _, filenames in os.walk(root_dir):
        if any(
            x in dirpath
            for x in [
                ".git",
                "__pycache__",
                "node_modules",
                "vendor",
                "docs",
                "archive",
            ]
        ):
            continue

        for filename in filenames:
            if not filename.endswith((".py", ".ts", ".js", ".cpp", ".h", ".m")):
                continue

            filepath = join(dirpath, filename)

            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    for key, pattern in patterns.items():
                        if pattern.search(line):
                            if filepath not in results:
                                results[filepath] = []
                            results[filepath].append((i + 1, key, line.strip()))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    return results


if __name__ == "__main__":
    findings = scan_for_incomplete_code(".")

    print(
        f"Found {sum(len(v) for v in findings.values())} potential issues in {len(findings)} files."
    )

    for filepath, items in sorted(findings.items()):
        print(f"\n{filepath}:")
        for line_num, key, content in items:
            print(f"  Line {line_num} [{key}]: {content}")
