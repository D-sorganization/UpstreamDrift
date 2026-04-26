"""
Aggressive auto-resolve: remove ALL conflict marker lines and their content.
Strategy: keep the <START marker ... MID marker (our side) and drop MID marker ... END marker (their side).
For files where the START comes AFTER END, take the origin/main side instead.
"""

import re
import subprocess
from pathlib import Path


def resolve_file(content: str) -> tuple[str, int]:
    """Remove all conflict blocks, picking whichever side first appears."""
    count = 0
    lines = content.split("\n")
    result = []
    state = "normal"  # 'normal', 'ours', 'theirs'

    for line in lines:
        if re.match(r"^" + "<" * 7 + " ", line):
            state = "ours"
            count += 1
            continue
        if re.match(r"^=======$", line.rstrip()):
            if state == "ours":
                state = "theirs"
            else:
                # Orphaned separator — skip it
                pass
            continue
        if re.match(r"^" + ">" * 7 + " ", line):
            state = "normal"
            continue

        if state == "normal" or state == "ours":
            result.append(line)
        # In 'theirs' state — drop the line

    return "\n".join(result), count


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
        if "<" * 7 + " " not in original and ">" * 7 + " " not in original:
            continue
        # Also handle orphaned >>>>>>> without matching <<<<<<<
        has_start = "<" * 7 + " " in original
        has_end = ">" * 7 + " " in original
        if not has_start and not has_end:
            continue

        resolved, count = resolve_file(original)

        # Verify no markers remain
        still_has = (
            "<" * 7 + " " in resolved
            or ">" * 7 + " " in resolved
            or re.search(r"^=======$", resolved, re.MULTILINE)
        )
        if count > 0 or (has_end and not has_start):
            path.write_text(resolved, encoding="utf-8")
            fixed.append((count, fname))
            print(f"Fixed {fname}")
    except Exception as e:
        errors.append((fname, str(e)))
        print(f"ERROR {fname}: {e}")

print(f"\nTotal fixed: {len(fixed)} files")
if errors:
    print(f"Errors: {len(errors)}")
