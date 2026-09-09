"""Wrong-cwd artifact-dir hygiene contract (issue #9415).

`.scratch/`, top-level `motion_matching/` and `motion_matching_training/`
are wrong-cwd run artifacts (UpstreamDrift #8837/#8841, tracked by #9415),
and `output/` is a gitignored smoke-debris dir allowed to keep only its
`.gitkeep`. The #9427 fix removed the committed files; this contract pins
that they never come back, and that the `.gitignore` patterns stay in
place so regeneration does not silently re-track them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dirs that must contain zero *tracked* files (they may exist on disk as
# unignored-but-empty or ignored regeneration targets).
ZERO_TRACKED_DIRS = (
    "motion_matching",
    "motion_matching_training",
    ".scratch",
)

# Allowed tracked entries in output/ (the smoke-debris dir is gitignored
# except for its keepfile).
OUTPUT_ALLOWED = frozenset({".gitkeep"})

# .gitignore lines (path prefix) that must stay present so regenerated
# artifacts are never re-tracked accidentally.
REQUIRED_GITIGNORE_PATTERNS = (
    "/output/*",
    "!/output/.gitkeep",
    "/.scratch/",
    ".scratch/",
    "/motion_matching_training/",
    "/motion_matching/",
    "/reports/*.json",
)


def _tracked_files(path_prefix: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", path_prefix],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"git ls-files failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_artifact_dirs_have_no_tracked_files() -> None:
    offenders: dict[str, list[str]] = {}
    for directory in ZERO_TRACKED_DIRS:
        tracked = _tracked_files(directory)
        if tracked:
            offenders[directory] = tracked
    assert offenders == {}, (
        f"Wrong-cwd artifact dirs must stay untracked (#9415): {offenders!r}"
    )


def test_output_dir_tracks_only_the_keepfile() -> None:
    names = sorted(
        Path(entry).relative_to("output").as_posix()
        for entry in _tracked_files("output")
    )
    allowed = sorted(OUTPUT_ALLOWED)
    assert names == allowed, (
        f"output/ may only track the keepfile (#9415): tracked {names!r}"
    )


def test_reports_dir_tracks_no_scanner_dumps() -> None:
    tracked = _tracked_files("reports")
    json_dumps = [f for f in tracked if f.endswith(".json")]
    assert json_dumps == [], (
        f"reports/ must not track scanner dumps (#8836, #9415): {json_dumps!r}"
    )


def test_gitignore_pins_artifact_dir_patterns() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [
        pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in gitignore
    ]
    assert missing == [], f".gitignore lost artifact-dir patterns (#9415): {missing!r}"


if __name__ == "__main__":
    sys.exit(0)
