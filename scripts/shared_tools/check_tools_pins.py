"""Assert every Tools pin in this repository points at one revision.

UpstreamDrift consumes D-sorganization/Tools through several pins that must
never disagree (UD #9406, Phase 1 of RM #1505):

* the ``vendor/ud-tools`` submodule gitlink (authoritative — the Python tree),
* ``Cargo.toml`` ``[workspace.dependencies] tools-core = { git = ..., rev = ... }``
  (the Rust kernel),
* any ``ud-tools @ git+https://github.com/D-sorganization/Tools.git@<sha>``
  pin inside ``pyproject.toml`` (the pip consumption path).

A ``ud_tools @ https://github.com/D-sorganization/Tools/releases/download/...whl``
release-wheel pin is *reported* with its version but not compared: releases are
cut from a tag commit (v1.15.0 = e87b04105) that need not equal the gitlink.
The gitlink wins for the source tree (tests, Docker, launcher fallback); the
wheel is what ``pip install "upstream-drift[tools]"`` users receive.

Exit 0 when every pin found equals the gitlink SHA, 1 otherwise. ``--json``
prints the resolved pins for other tooling (e.g. vendor-freshness).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess  # nosec B404 - fixed git invocation, no shell
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_PATH = "vendor/ud-tools"
TOOLS_REPO_URL = "https://github.com/D-sorganization/Tools.git"

_CARGO_RE = re.compile(
    r'^\s*tools-core\s*=\s*\{[^}]*git\s*=\s*"([^"]+)"[^}]*rev\s*=\s*"([0-9a-fA-F]{7,40})"',
    re.MULTILINE,
)
_WHEEL_RE = re.compile(
    r"ud[-_]tools\s*@\s*https://github\.com/D-sorganization/Tools/releases/download/"
    r"v(?P<tag>[0-9][^/]*)/ud_tools-(?P<version>[^-]+)-py3-none-any\.whl"
)
_PIP_RE = re.compile(
    r"ud-tools\s*@\s*git\+" + re.escape(TOOLS_REPO_URL) + r"@([0-9a-fA-F]{7,40})"
)


@dataclass(frozen=True)
class Pin:
    surface: str
    path: str
    sha: str | None


def read_gitlink(repo_root: Path) -> str | None:
    """Return the submodule SHA recorded in the index (works without checkout)."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        completed = subprocess.run(  # nosec B603
            [git, "-C", str(repo_root), "ls-files", "--stage", SUBMODULE_PATH],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    fields = completed.stdout.split()
    if len(fields) >= 2 and fields[0] == "160000":
        return fields[1]
    return None


def read_cargo_pin(cargo_toml: Path) -> tuple[str | None, str | None]:
    """Return (git url, rev) of the ``tools-core`` workspace dependency."""
    if not cargo_toml.is_file():
        return None, None
    match = _CARGO_RE.search(cargo_toml.read_text(encoding="utf-8"))
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def read_pyproject_pins(pyproject: Path) -> list[str]:
    if not pyproject.is_file():
        return []
    return _PIP_RE.findall(pyproject.read_text(encoding="utf-8"))


def read_pyproject_wheel_pins(pyproject: Path) -> list[tuple[str, str]]:
    """Return ``(tag, version)`` for every Tools release-wheel pin."""
    if not pyproject.is_file():
        return []
    return [
        (m.group("tag"), m.group("version"))
        for m in _WHEEL_RE.finditer(pyproject.read_text(encoding="utf-8"))
    ]


def collect_pins(repo_root: Path) -> list[Pin]:
    pins = [Pin("submodule gitlink", SUBMODULE_PATH, read_gitlink(repo_root))]
    url, rev = read_cargo_pin(repo_root / "Cargo.toml")
    pins.append(Pin("Cargo.toml tools-core", "Cargo.toml", rev))
    if url is not None and url.rstrip("/") != TOOLS_REPO_URL:
        pins.append(Pin("Cargo.toml tools-core url", "Cargo.toml", f"UNEXPECTED:{url}"))
    for index, sha in enumerate(read_pyproject_pins(repo_root / "pyproject.toml")):
        pins.append(Pin(f"pyproject ud-tools pin #{index + 1}", "pyproject.toml", sha))
    for tag, version in read_pyproject_wheel_pins(repo_root / "pyproject.toml"):
        # A release wheel is versioned, not sha-pinned. The gitlink still wins
        # for the source tree; the wheel is reported, never compared (UD #9406).
        pins.append(
            Pin(
                f"pyproject ud_tools wheel v{tag}", "pyproject.toml", f"WHEEL:{version}"
            )
        )
    return pins


def check_pins(pins: list[Pin]) -> list[str]:
    """Return human-readable mismatch messages (empty when consistent)."""
    errors: list[str] = []
    gitlink = pins[0].sha
    if gitlink is None:
        return [f"{SUBMODULE_PATH} gitlink not found in the index"]
    for pin in pins[1:]:
        if pin.sha is None:
            errors.append(f"{pin.surface} ({pin.path}): no Tools pin found")
        elif pin.sha.startswith("WHEEL:"):
            continue  # informational: release wheel version, not a commit
        elif pin.sha.startswith("UNEXPECTED:"):
            errors.append(
                f"{pin.surface}: {pin.sha[len('UNEXPECTED:') :]} is not {TOOLS_REPO_URL}"
            )
        elif not gitlink.startswith(pin.sha.lower()) and not pin.sha.lower().startswith(
            gitlink
        ):
            errors.append(
                f"{pin.surface} ({pin.path}) = {pin.sha} but {SUBMODULE_PATH} "
                f"gitlink = {gitlink}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="Print pins as JSON")
    args = parser.parse_args(argv)
    pins = collect_pins(args.repo_root.resolve())
    errors = check_pins(pins)
    if args.json:
        print(
            json.dumps({"pins": [asdict(p) for p in pins], "errors": errors}, indent=2)
        )
    else:
        for pin in pins:
            print(f"{pin.surface:32s} {pin.sha}")
    if errors:
        print("Tools pin consistency FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Tools pins are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
