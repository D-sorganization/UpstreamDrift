"""
Auto-resolve git conflict markers by choosing HEAD (ours) side.
For each conflict block: keep everything between <<<<<<< and =======, discard the rest.
"""

import re
import subprocess
from pathlib import Path

CONFLICT_START = re.compile(r"^<<<<<<< .*$", re.MULTILINE)
CONFLICT_SEP = re.compile(r"^=======\s*$", re.MULTILINE)
CONFLICT_END = re.compile(r"^>>>>>>> .*$", re.MULTILINE)


def resolve_ours(content: str) -> tuple[str, int]:
    """Remove conflict markers, keeping the HEAD (ours) side."""
    resolved = []
    pos = 0
    count = 0
    while pos < len(content):
        start_match = CONFLICT_START.search(content, pos)
        if not start_match:
            resolved.append(content[pos:])
            break

        resolved.append(content[pos : start_match.start()])

        sep_match = CONFLICT_SEP.search(content, start_match.end())
        end_match = CONFLICT_END.search(content, start_match.end())

        if not sep_match or not end_match:
            resolved.append(content[start_match.start() :])
            pos = len(content)
            break

        ours_content = content[start_match.end() : sep_match.start()]
        resolved.append(ours_content)
        pos = end_match.end()
        if pos < len(content) and content[pos] == "\n":
            pos += 1
        count += 1

    return "".join(resolved), count


result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)

extensions = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".sh",
    ".js",
    ".ts",
    ".json",
}
fixed = []
errors = []

for fname in result.stdout.splitlines():
    path = Path(fname)
    if not path.exists() or path.suffix not in extensions:
        continue
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
        if "<<<<<<< " not in original:
            continue
        resolved, count = resolve_ours(original)
        if count > 0:
            path.write_text(resolved, encoding="utf-8")
            fixed.append((count, fname))
            print(f"Fixed {count} conflict(s): {fname}")
    except Exception as e:
        errors.append((fname, str(e)))
        print(f"ERROR {fname}: {e}")

print(f"\nTotal fixed: {len(fixed)} files")
if errors:
    print(f"Errors: {len(errors)}")
    for fname, err in errors:
        print(f"  {fname}: {err}")
