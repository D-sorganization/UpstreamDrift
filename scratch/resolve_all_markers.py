import os
import re


def resolve_markers(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Choose origin/main side: everything between ======= and >>>>>>> origin/main
    # And discard everything between <<<<<<< HEAD and =======

    pattern = r"<<<<<<< HEAD.*?=======([\s\S]*?)>>>>>>> origin/main"

    new_content = re.sub(pattern, r"\1", content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


# Find all files with markers
files_with_markers = []
import subprocess

try:
    output = subprocess.check_output(["git", "grep", "-l", "<<<<<<<"], encoding="utf-8")
    files_with_markers = output.strip().split("\n")
except subprocess.CalledProcessError:
    pass

for path in files_with_markers:
    if not os.path.exists(path):
        continue
    if resolve_markers(path):
        pass
