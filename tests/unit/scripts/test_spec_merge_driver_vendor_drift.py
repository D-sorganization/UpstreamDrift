"""The vendored spec-merge-driver files must not drift from upstream.

Repository_Management owns both ``scripts/install_spec_merge_driver.py`` and
``shared_scripts/spec_changelog.py``; this repository vendors byte copies of
them. Issue #9476 records how a silently diverged copy shipped a false claim
into three repositories at once: the vendor step ran once and nothing compared
afterwards, so a corrected upstream never propagated. These tests pin the
SHA-256 digests of the corrected upstream files (Repository_Management main,
as fixed by Repository_Management#1521) so any future divergence fails loudly
instead of re-creating that failure mode.

The installer carries exactly one deliberate, pinned divergence: the module
docstring's wiring paragraph names this repository's own hook-setup entry
point (issue #9476, second comment). The drift test normalises that single
pinned line back to the upstream wording before hashing, so the pinned
upstream digest stays stable across that one reviewed edit -- and any other
byte that changes, in either file, fails the test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]

INSTALLER = REPO_ROOT / "scripts" / "install_spec_merge_driver.py"
SPEC_CHANGELOG = REPO_ROOT / "shared_scripts" / "spec_changelog.py"

#: sha256 of ``scripts/install_spec_merge_driver.py`` on Repository_Management
#: main as corrected by Repository_Management#1521 (commit ``36b42ec``);
#: verified 2026-09-08 against Repository_Management ``49c53dd``.
PINNED_INSTALLER_SHA256 = (
    "7008edb515ebdf3d659ed24c7dec1145977d66b4932c473c4f4541a449ebe6ef"
)
#: sha256 of ``shared_scripts/spec_changelog.py`` on Repository_Management
#: main (Repository_Management#1521, commit ``273160d``); verified 2026-09-08
#: against Repository_Management ``49c53dd``.
PINNED_SPEC_CHANGELOG_SHA256 = (
    "2f0ab9696ef87c8d85c28919bca078f5ebff384691307368e4cdb1e7dff239c4"
)

#: Upstream wording of the docstring span this repository is allowed to
#: rewrite -- the wiring paragraph naming the per-repo hook-setup entry
#: point -- and this repository's pinned replacement for it (issue #9476).
UPSTREAM_ENTRY_POINT_BLOCK = (
    "``scripts/install_workspace_hooks.py``; in Tools it is\n"
    "``scripts/setup_hooks.py``, which already registers that repo's other merge\n"
    "driver the same way. A vendored copy of this file that nothing invokes leaves\n"
)
VENDORED_ENTRY_POINT_BLOCK = (
    "``scripts/install_workspace_hooks.py``; in Tools and UpstreamDrift it is\n"
    "``scripts/setup_hooks.py``, which registers the ``spec-rows`` driver the same\n"
    "way. A vendored copy of this file that nothing invokes leaves\n"
)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of ``path``'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installer_normalized_to_upstream() -> str:
    """Return the installer text with the pinned vendored block reverted.

    The substitution is capped at one occurrence: if the pinned divergence
    block ever appears twice, or drifts in any other way, the resulting
    digest no longer matches the pinned upstream digest and the test fails.
    """
    text = INSTALLER.read_text(encoding="utf-8")
    return text.replace(VENDORED_ENTRY_POINT_BLOCK, UPSTREAM_ENTRY_POINT_BLOCK, 1)


def test_spec_changelog_is_byte_identical_to_pinned_upstream() -> None:
    """The vendored spec_changelog.py must equal the pinned upstream bytes."""
    assert _sha256(SPEC_CHANGELOG) == PINNED_SPEC_CHANGELOG_SHA256


def test_installer_matches_pinned_upstream_apart_from_pinned_entry_point() -> None:
    """Everything but the pinned entry-point line must equal upstream bytes."""
    normalized = _installer_normalized_to_upstream()
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        PINNED_INSTALLER_SHA256
    )
