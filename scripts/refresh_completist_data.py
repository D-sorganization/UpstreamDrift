import os
import subprocess
import sys
from os.path import exists, join

DATA_DIR = ".jules/completist_data"
os.makedirs(DATA_DIR, exist_ok=True)

EXCLUDE_DIRS = [
    ".git",
    ".jules",
    ".Jules",
    ".claude",
    ".agent",
    "node_modules",
    "build",
    "dist",
    "docs",
    "output",
]


def run_grep(pattern, output_file, extended_regex=False) -> None:
    """Run grep with the given pattern and write results to output_file."""
    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string")
    if not isinstance(output_file, str):
        raise ValueError("output_file must be a string")
    if not isinstance(extended_regex, bool):
        raise ValueError("extended_regex must be a bool")

    cmd = ["grep", "-rn"]
    if extended_regex:
        cmd.append("-E")
    cmd.append(pattern)
    cmd.append(".")

    # Exclude directories
    for d in EXCLUDE_DIRS:
        cmd.extend(["--exclude-dir", d])

    try:
        with open(output_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.SubprocessError):
        pass


def main() -> None:
    """Refresh completist audit data by running grep scans and stub finders."""

    # 1. Run find_stubs.py
    if exists("scripts/find_stubs.py"):
        subprocess.run([sys.executable, "scripts/find_stubs.py"])
    else:
        pass

    # 2. Grep for TODOs
    run_grep(
        "TRACKED_TASK|TRACKED_DEFECT|XXX|HACK|TEMP",
        join(DATA_DIR, "todo_markers.txt"),
        extended_regex=True,
    )

    # 3. Grep for NotImplementedError
    run_grep("NotImplementedError", join(DATA_DIR, "not_implemented.txt"))

    # 4. Grep for abstractmethod
    run_grep("@abstractmethod", join(DATA_DIR, "abstract_methods.txt"))


if __name__ == "__main__":
    main()
