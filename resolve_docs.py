"""Final cleanup: resolve 8-chevron conflicts in docs, skip CI workflow false-positives."""

import re
from pathlib import Path

# 8-chevron conflict pattern
START8 = re.compile(r"^<{8} .*$", re.MULTILINE)
SEP = re.compile(r"^={8}\s*$", re.MULTILINE)
END8 = re.compile(r"^>{8} .*$", re.MULTILINE)

doc_files = [
    "docs/assessments/issues/Issue_2299_Incomplete_Stub_in_models_py_20.md",
    "docs/assessments/issues/Issue_2282_Incomplete_Stub_in_flexible_shaft_py_342.md",
    "docs/assessments/issues/Issue_2159_Incomplete_Stub_in_impact_model_py_135.md",
    "docs/assessments/issues/Issue_2153_Incomplete_Stub_in_flexible_shaft_py_365.md",
]

for fname in doc_files:
    p = Path(fname)
    if not p.exists():
        print(f"Skip (not found): {fname}")
        continue
    content = p.read_text(encoding="utf-8", errors="replace")

    lines = content.split("\n")
    result = []
    state = "normal"
    count = 0
    for line in lines:
        if re.match(r"^<{8} ", line):
            state = "ours"
            count += 1
            continue
        elif re.match(r"^={8}\s*$", line.rstrip()):
            if state == "ours":
                state = "theirs"
            continue
        elif re.match(r"^>{8} ", line):
            state = "normal"
            continue
        if state != "theirs":
            result.append(line)

    p.write_text("\n".join(result), encoding="utf-8")
    print(f"Fixed {count} conflict(s): {fname}")

print("Done!")
