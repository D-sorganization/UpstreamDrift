#!/usr/bin/env python3
"""Change-fragment collation into SPEC.md and the development log (RM-5).

Split out of ``changes_fragment.py``, which stays the public facade and CLI.
``collate`` computes every new file content first and writes nothing, and
deletes no fragment, unless every update succeeds.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

try:
    from shared_scripts import changes_fragment_schema as schema
except ImportError:  # pragma: no cover - non-package fleet copy
    import importlib.util

    _path = Path(__file__).with_name("changes_fragment_schema.py")
    _cached = sys.modules.get("_fleet_changes_fragment_schema")
    if _cached is None:
        _spec = importlib.util.spec_from_file_location(
            "_fleet_changes_fragment_schema", _path
        )
        if _spec is None or _spec.loader is None:
            raise
        _cached = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _cached
        _spec.loader.exec_module(_cached)
    schema = _cached

log = logging.getLogger("changes_fragment")
development_log = schema.development_log
spec_changelog = schema.sibling("spec_changelog")
Fragment = schema.Fragment
FragmentError = schema.FragmentError
CHANGES_DIR = schema.CHANGES_DIR

SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")
PR_REF = r"#{pr}(?!\d)"
EMPTY_PR_VALUES = frozenset({"", "-", "n/a", "none", "not applicable", "not created"})


def apply_spec_row(text: str, *, pr: int, summary: str, today: date) -> str:
    """Return SPEC.md ``text`` with the ``#<pr>`` change-log row applied.

    Inserts ``| today | #pr | summary |`` at the top of the table (it is
    newest-first, as ``spec_changelog.union_rows`` assumes) when no row for the
    pull request exists. When one does, ``summary`` is appended to it unless it already
    holds it, so a re-run is a no-op and one pull request keeps one row.
    """
    changelog = spec_changelog.parse_changelog(text)
    key = f"#{pr}"
    rows = list(changelog.rows)
    for index, row in enumerate(rows):
        if row.key == key:
            if summary in row.summary:
                return text
            rows[index] = spec_changelog.Row(
                row.date, row.key, f"{row.summary}; {summary}"
            )
            break
    else:
        rows.insert(0, spec_changelog.Row(today.isoformat(), key, summary))
    result: str = spec_changelog.replace_rows(text, changelog, rows)
    return result


# ---------------------------------------------------------------------------
# collate: DEVELOPMENT_LOG.md
# ---------------------------------------------------------------------------


def _entry_span(lines: list[str], entry_id: str) -> tuple[int, int] | None:
    """Line range ``[start, end)`` of ``entry_id``'s block, or ``None``.

    Also finds a legacy ``DL-#N - Title`` heading, so it is never duplicated.
    """
    for start, raw in enumerate(lines):
        heading = development_log.ENTRY_LOOKUP_HEADING.match(raw.rstrip("\r\n"))
        if heading and heading.group(1) == entry_id:
            end = start + 1
            while end < len(lines) and not lines[end].startswith("#"):
                end += 1
            return start, end
    return None


def _canonical_heading(raw: str) -> str:
    """``raw`` unchanged when valid, else its legacy hyphen as a middle dot."""
    if development_log.ENTRY_HEADING.match(raw.rstrip("\r\n")):
        return raw
    heading = development_log.ENTRY_LOOKUP_HEADING.match(raw.rstrip("\r\n"))
    assert heading is not None, "caller passes a heading found by _entry_span"
    return f"### {heading.group(1)} · {heading.group(2)}\n"


def _section_insert_index(lines: list[str], heading_prefix: str) -> int | None:
    """Index just below the ``heading_prefix`` heading and its blank line."""
    for index, raw in enumerate(lines):
        if raw.startswith(heading_prefix):
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            return index
    return None


def _insert_block(lines: list[str], block: str, *, terminal: bool) -> list[str]:
    """Insert ``block`` at the top of Shipped (terminal) or Active."""
    candidates = ("## Shipped", "## Active") if terminal else ("## Active",)
    for prefix in candidates:
        index = _section_insert_index(lines, prefix)
        if index is not None:
            return lines[:index] + block.splitlines(keepends=True) + lines[index:]
    tail = [] if not lines or lines[-1].endswith("\n") else ["\n"]
    return lines + tail + ["\n"] + block.splitlines(keepends=True)


def _merge_pr(value: str, pr: int) -> str:
    """Add ``#pr`` to an entry's PR field unless it is already named."""
    if re.search(PR_REF.format(pr=pr), value):
        return value
    if value.strip().strip("`").lower() in EMPTY_PR_VALUES:
        return f"#{pr}"
    return f"{value}, #{pr}"


def _set_fields(block: list[str], updates: dict[str, str]) -> list[str]:
    """Replace field lines in ``block``; append fields it does not have yet."""
    result: list[str] = []
    pending = dict(updates)
    last_field = 0
    for raw in block:
        match = development_log.FIELD_LINE.match(raw.rstrip("\r\n"))
        if match:
            key = match.group("key").strip()
            if key in pending:
                raw = f"- **{key}:** {pending.pop(key)}\n"
            result.append(raw)
            last_field = len(result)
        else:
            result.append(raw)
    extra = [f"- **{key}:** {value}\n" for key, value in pending.items()]
    return result[:last_field] + extra + result[last_field:]


def _normalise_block(block: list[str]) -> str:
    return "".join(block).rstrip("\n") + "\n\n"


def _render_entry(
    fragment: Fragment, *, state: str, pr: int, today: date, verified: str
) -> str:
    next_step = fragment.next_step or f"Shipped in PR #{pr}."
    fields = (
        ("State", state),
        ("Owner", fragment.owner or "unassigned"),
        ("Issue", f"#{fragment.issue}"),
        ("Branch", fragment.branch or f"merged via #{pr}"),
        ("PR", f"#{pr}"),
        ("Paths", fragment.paths or f"see #{pr}"),
        ("Started", today.isoformat()),
        ("Last verified", verified),
        ("Summary", fragment.summary),
        ("Next step", next_step),
    )
    title = schema.effective_title(fragment.title, fragment.summary)
    body = "".join(f"- **{key}:** {value}\n" for key, value in fields)
    return f"### DL-#{fragment.issue} · {title}\n\n{body}\n"


def apply_devlog_entry(
    text: str,
    fragment: Fragment,
    *,
    pr: int,
    today: date,
    sha: str,
    source: str,
) -> str:
    """Return development-log ``text`` with ``fragment`` applied in place.

    An existing ``DL-#<issue>`` entry has its State (when given), PR, Last
    verified and Next step updated; an entry that becomes terminal moves to
    the top of the Shipped section. A missing entry is created (default state
    ``shipped``) at the top of Active or Shipped. Idempotent for equal input.

    Raises :class:`FragmentError` when the fragment would move a terminal
    (shipped or abandoned) entry to another state: that work needs a new
    issue-keyed entry.
    """
    assert SHA.match(sha), "sha must be a hex commit id"
    lines = text.splitlines(keepends=True)
    verified = f"{today.isoformat()} (`{sha[:8]}`; collated from {source})"
    entry_id = f"DL-#{fragment.issue}"
    span = _entry_span(lines, entry_id)
    terminal = development_log.TERMINAL_STATES

    if span is None:
        state = fragment.dl_state or "shipped"
        block = _render_entry(
            fragment, state=state, pr=pr, today=today, verified=verified
        )
        return "".join(_insert_block(lines, block, terminal=state in terminal))

    start, end = span
    lines[start] = _canonical_heading(lines[start])
    current = development_log.parse_entries("".join(lines[start:end]))[0]
    old_state = current.state
    new_state = fragment.dl_state or old_state
    if old_state in terminal and new_state != old_state:
        raise FragmentError(
            f"{entry_id} is {old_state}; it never moves to {new_state}. Open a "
            "new issue and write the fragment against that issue's new entry."
        )
    updates = {
        "Last verified": verified,
        "PR": _merge_pr(current.fields.get("PR", ""), pr),
    }
    if fragment.dl_state:
        updates["State"] = fragment.dl_state
    if fragment.next_step:
        updates["Next step"] = fragment.next_step
    elif new_state == "shipped" and old_state != "shipped":
        updates["Next step"] = f"Shipped in PR #{pr} (`{sha[:8]}`)."
    if (
        new_state in development_log.BRANCH_STATES
        and not current.fields.get("Branch", "").strip()
    ):
        if not fragment.branch:
            raise FragmentError(f"{entry_id} becomes {new_state}; set 'branch:'")
        updates["Branch"] = fragment.branch
    block = _normalise_block(_set_fields(lines[start:end], updates))

    if new_state in terminal and old_state not in terminal:
        remaining = lines[:start] + lines[end:]
        return "".join(_insert_block(remaining, block, terminal=True))
    return "".join(lines[:start]) + block + "".join(lines[end:])


# ---------------------------------------------------------------------------
# collate: orchestration
# ---------------------------------------------------------------------------


def collate(
    repo_root: Path,
    fragment_paths: Sequence[Path],
    *,
    pr: int,
    today: date,
    sha: str,
) -> list[Path]:
    """Fold ``fragment_paths`` into SPEC.md and the development log, then delete.

    Preconditions: ``pr >= 1``; ``sha`` is a 7-40 character hex commit id;
    every fragment validates. Raises :class:`FragmentError` (and touches no
    file, deleting no fragment) when a fragment is invalid, SPEC.md has no
    parsable change log, or a development-log update is refused. A missing
    SPEC.md or development log is skipped. Returns the fragments deleted.
    """
    if pr < 1:
        raise ValueError(f"pr must be a positive pull request number, got {pr}")
    if not SHA.match(sha):
        raise ValueError(f"sha must be a 7-40 character hex commit id, got {sha!r}")
    loaded: list[tuple[Path, Fragment]] = []
    errors: list[str] = []
    for path in fragment_paths:
        try:
            loaded.append((path, schema.load_fragment(path)))
        except FragmentError as exc:
            errors.append(str(exc))
    if errors:
        raise FragmentError("\n".join(errors))
    if not loaded:
        return []

    # Compute every new file content first; write and delete only when all
    # updates succeeded, so a failure never loses a fragment.
    writes: list[tuple[Path, str]] = []
    spec_path = repo_root / "SPEC.md"
    if spec_path.is_file():
        summaries = list(dict.fromkeys(fragment.summary for _, fragment in loaded))
        try:
            updated = apply_spec_row(
                spec_path.read_text(encoding="utf-8"),
                pr=pr,
                summary="; ".join(summaries),
                today=today,
            )
        except spec_changelog.SpecChangelogError as exc:
            raise FragmentError(f"SPEC.md change log not updated: {exc}") from exc
        writes.append((spec_path, updated))

    devlog_path = development_log.resolve_canonical_devlog_path(repo_root)
    if devlog_path.is_file():
        text = devlog_path.read_text(encoding="utf-8")
        for path, fragment in loaded:
            text = apply_devlog_entry(
                text,
                fragment,
                pr=pr,
                today=today,
                sha=sha,
                source=f"{CHANGES_DIR}/{path.name}",
            )
        writes.append((devlog_path, text))

    for target, content in writes:
        target.write_text(content, encoding="utf-8", newline="\n")
    for path, _fragment in loaded:
        path.unlink()
    log.debug("collated %d fragment(s) for #%d", len(loaded), pr)
    return [path for path, _ in loaded]
