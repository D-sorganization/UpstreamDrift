"""install.sh is a checksum-verified tagged release, never curl|sh (UD #15762)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts.ci.verify_release_artifacts import parse_sha256sums

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
USER_GUIDE = REPO_ROOT / "docs" / "user_guide" / "installation.md"
README = REPO_ROOT / "README.md"

# Same line-oriented pipe-to-shell detector as the Dockerfile check (#15762).
PIPE_TO_SHELL = re.compile(r"\b(curl|wget)\b[^\n]*\|\s*(sudo\s+)?(ba|z)?sh\b")

RECOMMENDATION_DOCS = (
    REPO_ROOT / "docs" / "development" / "GEPPETTO_VIEWER_AND_STANDALONE_PACKAGING.md",
    REPO_ROOT
    / "docs"
    / "development"
    / "architecture_implementation_plan"
    / "part-02-phase-3-unified-distribution.md",
)


def _read_text(path: Path) -> str:
    """Return the UTF-8 text of a required repository file.

    Preconditions: ``path`` exists and is a file.
    Postcondition: the return value is the file's full text.
    """
    if not path.is_file():
        raise ValueError(f"required file missing: {path}")
    return path.read_text(encoding="utf-8")


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_pipe_detector_flags_unpinned_install_pipe() -> None:
    sample = (
        "curl -fsSL https://raw.githubusercontent.com/D-sorganization/"
        "UpstreamDrift/main/install.sh | bash"
    )
    assert PIPE_TO_SHELL.search(sample)
    assert not PIPE_TO_SHELL.search(
        "curl -fsSL -o install.sh https://example.invalid/install.sh"
    )


def test_install_sh_has_no_pipe_to_shell() -> None:
    text = _read_text(INSTALL_SH)
    match = PIPE_TO_SHELL.search(text)
    assert match is None, (
        f"install.sh pipes a download into a shell: {match.group(0)!r}"
    )
    assert "/main/install.sh" not in text


def test_install_sh_runs_only_after_full_parse() -> None:
    """A truncated download must not execute the installer body."""
    text = _read_text(INSTALL_SH)
    lines = text.splitlines()
    set_flags = next(
        i for i, line in enumerate(lines) if line.strip() == "set -euo pipefail"
    )
    main_def = next(i for i, line in enumerate(lines) if line.strip() == "main() {")
    assert set_flags < main_def
    assert _nonempty_lines(text)[-1] == 'main "$@"'


def test_install_sh_documents_checksum_verified_tag() -> None:
    text = _read_text(INSTALL_SH)
    assert "releases/download/" in text
    assert "sha256sum -c" in text
    assert "install.sh.sha256" in text


def test_release_publishes_install_script_and_checksum() -> None:
    text = _read_text(RELEASE_WORKFLOW)
    assert "release-assets/install.sh" in text
    assert "release-assets/install.sh.sha256" in text
    assert "sha256sum -c install.sh.sha256" in text


def test_release_checksum_manifest_keeps_bare_filenames(tmp_path: Path) -> None:
    """SHA256SUMS names must match dist file names.

    ``sha256sum ./*`` records ``./<name>``. ``verify_release_artifacts`` then
    reports every artifact as untracked. ``sha256sum -- *`` keeps the basename
    and still satisfies shellcheck SC2035.
    """
    text = _read_text(RELEASE_WORKFLOW)
    assert "sha256sum ./*" not in text
    assert "sha256sum -- *" in text

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "upstream.txt").write_text("payload\n", encoding="utf-8")
    manifest = dist / "SHA256SUMS.txt"
    subprocess.run(
        ["bash", "-c", "sha256sum -- * > SHA256SUMS.txt"],
        cwd=dist,
        check=True,
    )
    assert set(parse_sha256sums(manifest)) == {"upstream.txt"}


def test_user_guide_documents_verified_tagged_install() -> None:
    text = _read_text(USER_GUIDE)
    assert PIPE_TO_SHELL.search(text) is None
    assert "## Checksum-Verified Tagged Release" in text
    assert "releases/download/" in text
    assert "sha256sum -c" in text


def test_readme_points_at_verified_tagged_install() -> None:
    text = _read_text(README)
    assert PIPE_TO_SHELL.search(text) is None
    assert "checksum-verified-tagged-release" in text


@pytest.mark.parametrize("path", RECOMMENDATION_DOCS, ids=lambda p: p.name)
def test_install_recommendations_have_no_pipe_to_shell(path: Path) -> None:
    text = _read_text(path)
    match = PIPE_TO_SHELL.search(text)
    assert match is None, (
        f"{path.name} pipes a download into a shell: {match.group(0)!r}"
    )
    assert "sha256sum -c" in text
