"""Guard: every pending ``changes/`` fragment must be collatable.

``collate-changes.yml`` folds fragments into SPEC.md and the development log
one at a time and aborts on the first failure, so a single uncollatable
fragment blocks every other fragment on ``main``. This test applies each
fragment, in file order, to in-memory copies of those files using the same
public functions the collate CLI uses, and reports every failure at once.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from shared_scripts import changes_fragment as cf

REPO_ROOT = Path(__file__).resolve().parents[3]
_SHA = "0" * 40
_PR = 1
_TODAY = date(2026, 1, 1)

pytestmark = pytest.mark.unit


def _collation_failures(repo_root: Path) -> list[str]:
    spec_path = repo_root / "SPEC.md"
    devlog_path = cf.schema.development_log.resolve_canonical_devlog_path(repo_root)
    spec = spec_path.read_text(encoding="utf-8")
    devlog = devlog_path.read_text(encoding="utf-8")
    failures: list[str] = []
    for path in cf.find_fragments(repo_root):
        try:
            fragment = cf.load_fragment(path)
            new_spec = cf.apply_spec_row(
                spec, pr=_PR, summary=fragment.summary, today=_TODAY
            )
            new_devlog = cf.apply_devlog_entry(
                devlog,
                fragment,
                pr=_PR,
                today=_TODAY,
                sha=_SHA,
                source=f"{cf.CHANGES_DIR}/{path.name}",
            )
        except (cf.FragmentError, ValueError) as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        spec, devlog = new_spec, new_devlog
    return failures


def test_all_pending_fragments_are_collatable() -> None:
    failures = _collation_failures(REPO_ROOT)
    assert not failures, (
        f"{len(failures)} fragment(s) cannot be collated and would abort "
        "collate-changes.yml:\n" + "\n".join(failures)
    )
