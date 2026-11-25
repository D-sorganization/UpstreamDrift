"""Dockerfiles never execute unverified downloaded scripts (UD #15762)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = sorted(REPO_ROOT.glob("Dockerfile*"))
PIPE_TO_SHELL = re.compile(r"\b(curl|wget)\b[^\n]*\|\s*(sudo\s+)?(ba|z)?sh\b")


def _joined_instructions(path: Path) -> str:
    """Return the Dockerfile with backslash continuations folded and comments dropped."""
    text = path.read_text(encoding="utf-8").replace("\\n", " ")
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_dockerfiles_are_discovered() -> None:
    assert REPO_ROOT / "Dockerfile" in DOCKERFILES


def test_pipe_detector_flags_curl_pipe_sh() -> None:
    assert PIPE_TO_SHELL.search("RUN curl -fsSL https://x.example/i.sh | sh")
    assert PIPE_TO_SHELL.search("RUN wget -qO- https://x.example/i.sh | bash")
    assert not PIPE_TO_SHELL.search(
        "RUN curl -fsSL -o /tmp/f https://x && sha256sum -c"
    )


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_no_pipe_to_shell(path: Path) -> None:
    assert PIPE_TO_SHELL.findall(_joined_instructions(path)) == []


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_no_unverified_rustup_bootstrap_script(path: Path) -> None:
    assert "sh.rustup.rs" not in _joined_instructions(path)


def test_rustup_init_is_checksum_verified() -> None:
    body = _joined_instructions(REPO_ROOT / "Dockerfile")
    assert "rustup-init" in body
    assert re.search(r"sha256sum\s+(-c|--check)", body)
    assert re.search(r"RUSTUP_VERSION=\d+\.\d+\.\d+", body)
