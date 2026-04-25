"""Find all files with git conflict markers."""

import subprocess
from pathlib import Path

result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)

conflicted = []
for fname in result.stdout.splitlines():
    path = Path(fname)
    if not path.exists() or path.suffix not in (
        ".py",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
        ".sh",
        ".js",
        ".ts",
    ):
        continue
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if "<<<<<<< HEAD" in content or "<<<<<<< " in content:
            count = content.count("<<<<<<< ")
            conflicted.append((count, fname))
    except Exception:
        pass

conflicted.sort(reverse=True)
print(f"Total files with conflict markers: {len(conflicted)}")
for count, fname in conflicted[:30]:
    print(f"  {count:3d} conflicts: {fname}")
