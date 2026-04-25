"""
Fix syntax-broken files by checking them out from origin/main.
These files were mangled by our conflict resolution.
"""

import ast
import subprocess
from pathlib import Path

result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)

bad_files = []
for fname in result.stdout.splitlines():
    path = Path(fname)
    if not path.exists() or path.suffix != ".py":
        continue
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        ast.parse(content)
    except SyntaxError:
        bad_files.append(fname)

print(f"Files with syntax errors: {len(bad_files)}")
for f in bad_files:
    # Restore from origin/main
    r = subprocess.run(
        ["git", "checkout", "origin/main", "--", f], capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"  Restored: {f}")
    else:
        print(f"  FAILED to restore {f}: {r.stderr.strip()}")
