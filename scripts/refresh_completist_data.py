import os
import subprocess
import sys

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


def run_grep(pattern, output_file, extended_regex=False):
    cmd = ["grep", "-rn"]
    if extended_regex:
        cmd.append("-E")
    cmd.append(pattern)
    cmd.append(".")

    # Exclude directories
    for d in EXCLUDE_DIRS:
        cmd.extend(["--exclude-dir", d])

    print(f"Running: {' '.join(cmd)} > {output_file}")

    try:
        with open(output_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"Error running grep: {e}")


def main():
    print("Refreshing completist data...")

    # 1. Run find_stubs.py
    if os.path.exists("scripts/find_stubs.py"):
        print("Running scripts/find_stubs.py...")
        subprocess.run([sys.executable, "scripts/find_stubs.py"])
    else:
        print("scripts/find_stubs.py not found!")

    # 2. Grep for TODOs
    run_grep(
        "TODO|FIXME|XXX|HACK|TEMP",
        os.path.join(DATA_DIR, "todo_markers.txt"),
        extended_regex=True,
    )

    # 3. Grep for NotImplementedError
    run_grep("NotImplementedError", os.path.join(DATA_DIR, "not_implemented.txt"))

    # 4. Grep for abstractmethod
    run_grep("@abstractmethod", os.path.join(DATA_DIR, "abstract_methods.txt"))

    print("Done.")


if __name__ == "__main__":
    main()
