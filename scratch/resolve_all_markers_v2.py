import os
import re


def resolve_markers(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Choose origin/main side: everything between ======= and >>>>>>> origin/main
    # Handle both <<<<<<< (7) and <<<<<<<< (8) markers

    pattern = r"<{7,}[^=\n]*?[\s\S]*?={7,}.*?\n([\s\S]*?)>{7,}.*?\n"

    new_content = re.sub(pattern, r"\1", content)

    if new_content != content:
        # Double check for nested markers after one pass
        while re.search(r"<{7,}", new_content) and re.search(r"={7,}", new_content):
            new_content = re.sub(pattern, r"\1", new_content)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


# Find all files with markers
files_with_markers = []
import subprocess

try:
    # Search for at least 7 < characters
    output = subprocess.check_output(
        ["git", "grep", "-l", "-E", "<{7,}"], encoding="utf-8"
    )
    files_with_markers = output.strip().split("\n")
except subprocess.CalledProcessError:
    pass

for path in files_with_markers:
    if not os.path.exists(path):
        continue
    # Skip binary files
    if path.endswith((".pdf", ".woff2", ".png", ".jpg")):
        continue
    # Skip workflow files that contain the string as part of a check
    if "ci-standard.yml" in path or "ci-optional-stack.yml" in path:
        continue

    if resolve_markers(path):
        pass
