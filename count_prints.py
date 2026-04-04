from pathlib import Path

# Identify where print is used
src_dir = Path("src")

matches = []
for p in src_dir.rglob("*.py"):
    try:
        content = p.read_text(encoding="utf-8")
        if "print(" in content:
            matches.append(p)
    except Exception:
        pass

for _m in matches:
    pass
