"""The child-copy counterpart guard must cover every headered file.

The two halves of the child-copy contract used to disagree:

* ``test_current_branch_does_not_edit_tools_child_copies`` treats *any* file
  carrying the ``DO NOT EDIT`` header as Tools-owned and refuses edits to it.
* ``test_tools_child_copy_headers_have_tools_counterparts`` only validated an
  allowlist of four paths plus ``sidekick/agent``.

So a file could carry the header -- and be frozen against editing in
UpstreamDrift -- while having no counterpart in Tools to edit instead. That is
exactly what happened to ``ai/tools/sidekick_analytics.py``: headered, frozen,
and absent from Tools, leaving its defect unfixable in either repo. The guard
was trusted precisely where it was blind.

These tests pin the *scope* of the guard, independent of today's waiver
contents, so the hole cannot silently reopen.

Part of D-sorganization/UpstreamDrift#9474.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.repo_hygiene.test_tools_child_copy_contract import (
    _HEADER,
    _headered_paths_missing_counterparts,
    _load_waived_missing_counterparts,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _write(root: Path, relative: str, *, headered: bool) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"{_HEADER}\n" if headered else ""
    path.write_text(body + '"""Module."""\n', encoding="utf-8")


def test_guard_flags_a_headered_file_anywhere_in_the_tree(tmp_path: Path) -> None:
    """A headered file outside any allowlist must still be caught.

    This is the regression: the path below is in no allowlist, and the old
    implementation returned nothing for it.
    """
    _write(tmp_path, "ai/tools/some_new_tool.py", headered=True)

    missing = _headered_paths_missing_counterparts(tmp_path, tools_paths=set())

    assert "ai/tools/some_new_tool.py" in missing


def test_guard_ignores_files_without_the_header(tmp_path: Path) -> None:
    """UD-only code with no header is not claimed by Tools."""
    _write(tmp_path, "ai/cli_providers/ud_only.py", headered=False)

    missing = _headered_paths_missing_counterparts(tmp_path, tools_paths=set())

    assert missing == []


def test_guard_accepts_a_headered_file_that_has_a_counterpart(
    tmp_path: Path,
) -> None:
    """A header backed by a real Tools file is exactly what should pass."""
    _write(tmp_path, "ai/tools/codemap_tools.py", headered=True)

    missing = _headered_paths_missing_counterparts(
        tmp_path, tools_paths={"ai/tools/codemap_tools.py"}
    )

    assert missing == []


def test_waiver_file_is_sorted_and_deduplicated() -> None:
    """The waiver is a reviewable ledger, so it must stay canonical."""
    waived = _load_waived_missing_counterparts()

    assert len(waived) == len(set(waived))
    assert waived == sorted(waived)


def test_sidekick_analytics_is_not_waived() -> None:
    """The file that exposed the hole must be fixed, never waived.

    It is upstreamed to Tools by D-sorganization/Tools#4959; once the
    ``vendor/ud-tools`` pin includes it, it has a counterpart and needs no
    waiver. If this fails, the fix was reverted rather than landed.
    """
    assert "ai/tools/sidekick_analytics.py" not in _load_waived_missing_counterparts()
