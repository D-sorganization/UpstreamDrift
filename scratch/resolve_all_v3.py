import os
import subprocess


def resolve_file(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return

    new_lines = []
    in_head = False
    in_origin = False

    for line in lines:
        if line.startswith(("<<<<<<<", "<<<<<<<<")):
            in_head = True
            in_origin = False
            continue
        if line.startswith("======="):
            in_head = False
            in_origin = True
            continue
        if line.startswith(">>>>>>>"):
            in_head = False
            in_origin = False
            continue

        if in_origin or not in_head:
            new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# Find all files with markers
try:
    output = subprocess.check_output(
        ["git", "grep", "-l", "-E", "<{7,}"], encoding="utf-8"
    )
    files = output.strip().split("\n")
except subprocess.CalledProcessError:
    files = []

for f in files:
    if not os.path.exists(f):
        continue
    if f.endswith((".pdf", ".woff2", ".png", ".jpg")):
        continue
    if "ci-standard.yml" in f or "ci-optional-stack.yml" in f:
        continue
    resolve_file(f)
