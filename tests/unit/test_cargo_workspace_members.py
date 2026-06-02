"""Guard against duplicate Rust workspace members in ``Cargo.toml`` (#7065).

The root ``[workspace] members`` list once carried ``upstream-mocap-io`` twice.
A duplicate member makes ``cargo metadata`` emit the crate more than once and is
pure noise that hides real list edits in review. This test pins each declared
member to a single occurrence using only the standard-library TOML reader, so it
runs without a ``cargo`` toolchain installed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_TOML = REPO_ROOT / "Cargo.toml"


def _workspace_members() -> list[str]:
    data = tomllib.loads(CARGO_TOML.read_text(encoding="utf-8"))
    workspace = data.get("workspace", {})
    members = workspace.get("members", [])
    assert isinstance(members, list), "[workspace] members must be a list"
    return [str(member) for member in members]


def test_workspace_members_have_no_duplicates() -> None:
    """Post: every declared workspace member appears exactly once."""
    counts = Counter(_workspace_members())
    duplicates = {member: n for member, n in counts.items() if n > 1}
    assert not duplicates, f"Duplicate workspace members in Cargo.toml: {duplicates}"


def test_upstream_mocap_io_listed_once() -> None:
    """Regression pin for #7065: the de-duplicated member stays singular."""
    members = _workspace_members()
    assert members.count("rust_core/upstream-mocap-io") == 1
