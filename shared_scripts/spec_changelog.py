#!/usr/bin/env python3
"""Portable SPEC.md change-log parser, validator, migrator and row-union merge.

Why this module exists
----------------------
Until Repository_Management#1520 every pull request in the fleet had to add a
``SPEC.md`` change-log row carrying *the next serial spec version* and bump the
``Spec Version`` field in the Identity table to match. Both of those are
**global counters**, so two concurrent pull requests necessarily pick the same
next value and necessarily edit the same two lines. The result is a textual
conflict on every second merge that carries no information: the fix is always
"renumber my row above theirs". On 2026-09-03 twelve re-merges were performed
across UpstreamDrift, Tools, Repository_Management and AffineDrift purely to
renumber rows.

The fix is to key a row by something that is unique *by construction* rather
than by coordination: the pull request (or issue) number. Two pull requests can
never pick the same key, so two rows can never disagree about what belongs on a
line. Row order becomes merge order, which is the only ordering that was ever
meaningful.

Row format
----------
::

    | Date       | PR    | Changes            |
    | ---------- | ----- | ------------------ |
    | 2026-09-03 | #1520 | one-line summary   |

``PR`` is ``#<number>`` — the pull request that adds the row, or the governing
issue when the number is known before the pull request exists. Historical rows
migrated by :func:`migrate_text` carry ``n/a`` when no reference could be
recovered from their summary; their original serial is preserved inline as
``(spec X.Y.Z)`` so nothing is lost.

The ``Spec Version`` field in the Identity table is **release-derived** from
this change onward: it is bumped when a release is cut (see
``scripts/bump_spec_version.py``), never by an individual pull request.

This module is deliberately dependency-free and importable by file path so the
fleet sync can drop it into any repository next to ``fleet_hooks.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Rows dated before this cutover were serial-versioned. Several genuinely share
# a governing issue (one issue, several pull requests), so their recovered keys
# are not unique and uniqueness is not enforced over them. This is the migration
# boundary for Repository_Management#1520, not a policy knob.
PR_KEYED_SINCE = "2026-09-03"

#: Marker used when a migrated historical row carries no recoverable reference.
NO_KEY = "n/a"

CANONICAL_HEADER = ("Date", "PR", "Changes")

_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,4})\s+(?:\d+(?:\.\d+)*\.?\s+)?change\s?log\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_KEY_RE = re.compile(r"^#\d+$")
_ISSUE_REF_RE = re.compile(r"#(\d+)")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")


class SpecChangelogError(RuntimeError):
    """Raised when SPEC.md has no parsable change-log table."""


@dataclass(frozen=True)
class Row:
    """One change-log row."""

    date: str
    key: str
    summary: str

    def render(self) -> str:
        return f"| {self.date} | {self.key} | {self.summary} |"

    @property
    def identity(self) -> tuple[str, str, str]:
        """Value used for de-duplication during a row union."""
        return (self.date, self.key, self.summary.strip())


@dataclass
class Changelog:
    """A parsed change-log table and where it sits in the document."""

    header: tuple[str, ...]
    rows: list[Row]
    #: Character offsets of the table (header line through last row) in the text.
    start: int
    end: int
    #: Rendered header + separator lines, preserved verbatim on rewrite.
    header_block: str

    @property
    def is_pr_keyed(self) -> bool:
        return tuple(self.header[:3]) == CANONICAL_HEADER


def _split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    inner = stripped.strip("|")
    return [cell.strip() for cell in inner.split("|")]


def parse_changelog(text: str) -> Changelog:
    """Parse the first change-log table in ``text``.

    Raises :class:`SpecChangelogError` when no ``Change Log`` heading is present
    or the heading is not followed by a table whose first column is ``Date``.
    """
    heading = _HEADING_RE.search(text)
    if heading is None:
        raise SpecChangelogError("no 'Change Log' heading found")

    lines = text.splitlines(keepends=True)
    # Offset of the first character after the heading line.
    offset = 0
    start_line = 0
    for index, line in enumerate(lines):
        if offset > heading.start():
            break
        if offset == heading.start():
            start_line = index + 1
            offset += len(line)
            break
        offset += len(line)
    else:  # pragma: no cover - heading always lies on a line boundary
        raise SpecChangelogError("could not locate the heading line")

    cursor = offset
    table_start: int | None = None
    header: tuple[str, ...] | None = None
    header_block = ""
    rows: list[Row] = []
    table_end = cursor

    for line in lines[start_line:]:
        cells = _split_cells(line)
        if header is None:
            if cells and cells[0].lower() == "date":
                header = tuple(cells)
                table_start = cursor
                header_block = line
                table_end = cursor + len(line)
            elif _HEADING_RE.match(line):
                break
            cursor += len(line)
            continue

        if _SEPARATOR_RE.match(line.strip()):
            header_block += line
            table_end = cursor + len(line)
            cursor += len(line)
            continue

        if not cells or len(cells) < 3:
            # First non-row line ends the table.
            break

        rows.append(Row(date=cells[0], key=cells[1], summary=" | ".join(cells[2:])))
        table_end = cursor + len(line)
        cursor += len(line)

    if header is None or table_start is None:
        raise SpecChangelogError(
            "the 'Change Log' heading is not followed by a table whose first "
            "column is 'Date'"
        )

    return Changelog(
        header=header,
        rows=rows,
        start=table_start,
        end=table_end,
        header_block=header_block,
    )


def render_changelog(changelog: Changelog, rows: list[Row]) -> str:
    body = "".join(row.render() + "\n" for row in rows)
    return changelog.header_block + body


def replace_rows(text: str, changelog: Changelog, rows: list[Row]) -> str:
    """Return ``text`` with the parsed table replaced by ``rows``."""
    rendered = render_changelog(changelog, rows)
    if not rendered.endswith("\n"):
        rendered += "\n"
    tail = text[changelog.end :]
    if tail.startswith("\n"):
        rendered = rendered.rstrip("\n")
    return text[: changelog.start] + rendered + tail


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(changelog: Changelog) -> list[str]:
    """Return human-readable problems with a parsed change log.

    The checks are deliberately the *same set* the serial-version gate had,
    with "unique serial version" replaced by "unique PR key":

    * the table header is the canonical ``Date | PR | Changes``;
    * every row has an ISO date, a ``#<number>`` key (or the ``n/a`` legacy
      marker), and a non-empty summary;
    * no ``#<number>`` key appears twice among rows dated on or after the
      :data:`PR_KEYED_SINCE` cutover;
    * no row still carries a bare semantic version where the key belongs,
      which is how a pre-migration row or a stale agent habit shows up.
    """
    failures: list[str] = []

    if not changelog.is_pr_keyed:
        failures.append(
            "change-log table header is "
            f"{' | '.join(changelog.header)!r}; expected "
            f"{' | '.join(CANONICAL_HEADER)!r} (Repository_Management#1520)"
        )

    seen: dict[str, str] = {}
    for row in changelog.rows:
        label = f"row {row.date} {row.key}"
        if not _DATE_RE.match(row.date):
            failures.append(f"{label}: first column is not an ISO date (YYYY-MM-DD)")
        if _SEMVER_RE.match(row.key):
            failures.append(
                f"{label}: the second column is a serial spec version. Rows are "
                "keyed by pull request now: use '#<pr or issue number>' "
                "(Repository_Management#1520)"
            )
        elif row.key != NO_KEY and not _KEY_RE.match(row.key):
            failures.append(
                f"{label}: second column must be '#<number>' "
                f"(or {NO_KEY!r} for a migrated historical row)"
            )
        if not row.summary.strip():
            failures.append(f"{label}: empty summary")
        elif (
            "|" in row.summary
            and _DATE_RE.match(row.date)
            and row.date >= PR_KEYED_SINCE
        ):
            # A pipe splits the row into extra cells, so the rendered markdown
            # table and the parsed row disagree and the row stops round-tripping
            # through the merge driver.
            failures.append(
                f"{label}: the summary contains '|', which markdown reads as a "
                "column break. Escape it as a backslash-pipe, or reword."
            )

        if (
            _KEY_RE.match(row.key)
            and _DATE_RE.match(row.date)
            and row.date >= PR_KEYED_SINCE
        ):
            if row.key in seen:
                failures.append(
                    f"duplicate change-log key {row.key} "
                    f"(already used by the {seen[row.key]} row). One row per "
                    "pull request; edit your own row instead of adding a second."
                )
            else:
                seen[row.key] = row.date

    return failures


def rows_added(before: str, after: str) -> list[Row]:
    """Return rows present in ``after`` but not in ``before``."""
    try:
        old = {row.identity for row in parse_changelog(before).rows}
    except SpecChangelogError:
        old = set()
    try:
        new = parse_changelog(after).rows
    except SpecChangelogError:
        return []
    return [row for row in new if row.identity not in old]


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _recover_key(summary: str) -> str:
    """Recover a reference from a historical summary, else :data:`NO_KEY`."""
    match = _ISSUE_REF_RE.search(summary)
    return f"#{match.group(1)}" if match else NO_KEY


def migrate_rows(rows: list[Row]) -> tuple[list[Row], int]:
    """Rewrite serial-versioned rows to PR-keyed rows, losing no content.

    A row whose key column holds a semantic version is rewritten so that the
    key becomes the first ``#<number>`` reference found in its summary (or
    :data:`NO_KEY`), and the original serial is appended to the summary as
    ``(spec X.Y.Z)`` so the old numbering stays traceable from the row itself.
    Rows already in the new form are returned untouched.
    """
    migrated: list[Row] = []
    changed = 0
    for row in rows:
        if not _SEMVER_RE.match(row.key):
            migrated.append(row)
            continue
        summary = row.summary.rstrip()
        note = f"(spec {row.key})"
        if note not in summary:
            summary = f"{summary} {note}" if summary else note
        migrated.append(
            Row(date=row.date, key=_recover_key(row.summary), summary=summary)
        )
        changed += 1
    return migrated, changed


#: Prose entries some repositories keep above the table, e.g.
#: ``Current 1.2.73 entry: ...``. They carry the same global serial the table
#: rows did, are inserted at the same offset by every pull request, and so are
#: the second conflict site. The migration freezes them: the text is preserved
#: verbatim, the word "Current" and the coordination it implies are not.
_PROSE_ENTRY_RE = re.compile(
    r"^Current (?P<version>\d+\.\d+\.\d+) entry:", re.MULTILINE
)

POLICY_NOTE = (
    "Rows are keyed by pull request, not by a serial spec version: "
    "`| YYYY-MM-DD | #<pr> | summary |`. Add exactly one row for your own "
    "pull request and do not renumber anybody else's; the `Spec Version` "
    "field in section 1 is bumped at release time by "
    "`scripts/bump_spec_version.py`, never by an individual pull request. "
    "See [Repository_Management#1520](https://github.com/D-sorganization/Repository_Management/issues/1520).\n"
)

FROZEN_PROSE_NOTE = (
    "The `Archived entry (spec X.Y.Z)` paragraphs below are frozen: they are "
    "the pre-#1520 serial-versioned narrative entries, kept verbatim for "
    "traceability. Do not add new ones — new detail goes in the row summary "
    "or the pull request.\n"
)


def migrate_prose_entries(text: str) -> tuple[str, int]:
    """Freeze ``Current X.Y.Z entry:`` prose paragraphs. Content is preserved."""
    matches = _PROSE_ENTRY_RE.findall(text)
    if not matches:
        return text, 0
    rewritten = _PROSE_ENTRY_RE.sub(
        lambda match: f"Archived entry (spec {match.group('version')}):", text
    )
    return rewritten, len(matches)


def ensure_policy_note(text: str, *, include_frozen_note: bool) -> str:
    """Insert the PR-keyed policy note directly under the Change Log heading."""
    heading = _HEADING_RE.search(text)
    if heading is None:
        raise SpecChangelogError("no 'Change Log' heading found")
    if POLICY_NOTE.strip() in text:
        return text
    note = "\n" + POLICY_NOTE
    if include_frozen_note:
        note += "\n" + FROZEN_PROSE_NOTE
    insert_at = heading.end()
    return text[:insert_at] + "\n" + note.lstrip("\n") + text[insert_at:]


#: Prettier pads markdown table cells to the width of the widest row, so with
#: the change-log table in its scope a single new row re-pads every other row --
#: turning a one-line addition into a whole-table diff that conflicts with every
#: concurrent pull request, which is the exact failure #1520 removes. The fence
#: is narrower than putting all of SPEC.md in `.prettierignore`: Prettier keeps
#: formatting the rest of the specification. (Tools already carried this fence;
#: the migration now gives it to every repository.)
PRETTIER_FENCE_START = "<!-- prettier-ignore-start -->"
PRETTIER_FENCE_END = "<!-- prettier-ignore-end -->"


def ensure_prettier_fence(text: str) -> str:
    """Wrap the change-log table in Prettier ignore markers. Idempotent."""
    changelog = parse_changelog(text)
    before = text[: changelog.start]
    if before.rstrip().endswith(PRETTIER_FENCE_START):
        return text
    table = text[changelog.start : changelog.end]
    after = text[changelog.end :]
    if not before.endswith("\n"):
        before += "\n"
    fenced = f"{PRETTIER_FENCE_START}\n\n{table}"
    if not fenced.endswith("\n"):
        fenced += "\n"
    fenced += f"\n{PRETTIER_FENCE_END}\n"
    return before + fenced + after.removeprefix("\n")


def migrate_text(text: str) -> tuple[str, int]:
    """Migrate the change log in ``text``. Returns ``(text, n_rows_rewritten)``.

    Three edits, all content-preserving and idempotent:

    1. ``Current X.Y.Z entry:`` prose paragraphs become
       ``Archived entry (spec X.Y.Z):`` — frozen, not deleted.
    2. A policy note naming the new row format is inserted under the heading,
       and the table is wrapped in Prettier ignore markers so a new row can
       never re-pad the whole table.
    3. Every serial-versioned table row is rewritten to a PR key, with the old
       serial appended to its summary as ``(spec X.Y.Z)``.
    """
    text, prose_count = migrate_prose_entries(text)
    text = ensure_policy_note(text, include_frozen_note=prose_count > 0)
    text = ensure_prettier_fence(text)
    changelog = parse_changelog(text)
    rows, changed = migrate_rows(changelog.rows)
    header = list(changelog.header)
    header_block = changelog.header_block
    if tuple(header[:3]) != CANONICAL_HEADER:
        header_lines = header_block.splitlines(keepends=True)
        widths = [
            max(len(CANONICAL_HEADER[index]), 10)
            for index in range(len(CANONICAL_HEADER))
        ]
        new_header = (
            "| "
            + " | ".join(
                CANONICAL_HEADER[index].ljust(widths[index]) for index in range(3)
            )
            + " |\n"
        )
        new_separator = (
            "| " + " | ".join("-" * widths[index] for index in range(3)) + " |\n"
        )
        header_block = new_header + new_separator
        if len(header_lines) > 2:  # pragma: no cover - defensive
            header_block += "".join(header_lines[2:])
        changelog.header_block = header_block
    return replace_rows(text, changelog, rows), changed


# ---------------------------------------------------------------------------
# Row union (merge driver support)
# ---------------------------------------------------------------------------


def _additions(side: list[Row], allowance: Counter[tuple[str, str, str]]) -> list[Row]:
    """Rows in ``side`` beyond the occurrences ``allowance`` already accounts for.

    ``allowance`` is consumed in place, so calling this for ``ours`` and then
    ``theirs`` counts a row both sides added exactly once.
    """
    extra: list[Row] = []
    seen: Counter[tuple[str, str, str]] = Counter()
    for row in side:
        seen[row.identity] += 1
        if seen[row.identity] > allowance[row.identity]:
            extra.append(row)
            allowance[row.identity] += 1
    return extra


def union_rows(base: list[Row], ours: list[Row], theirs: list[Row]) -> list[Row]:
    """Three-way union of change-log rows.

    Rows are independent facts about independent pull requests, so a merge
    never has to choose between two of them — it keeps both. Order is: rows
    either side added (ours first), then the surviving base rows in base order.
    Additions go on top because the table is newest-first and both sides
    prepend; putting them last would silently reorder the log.

    **Multiplicity is preserved exactly.** Several fleet change logs contain
    byte-identical duplicate rows — the same change logged twice under two
    serials, which is the very defect the serial scheme produced. Collapsing
    them here would make a merge silently delete history, so a row that appears
    twice in the base still appears twice in the result. Only a row *both*
    sides deleted is dropped, so a rebase can never drop somebody else's row.
    """
    base_counts: Counter[tuple[str, str, str]] = Counter(row.identity for row in base)
    our_counts: Counter[tuple[str, str, str]] = Counter(row.identity for row in ours)
    their_counts: Counter[tuple[str, str, str]] = Counter(
        row.identity for row in theirs
    )

    # Keep each base occurrence unless BOTH sides removed it.
    kept: list[Row] = []
    seen: Counter[tuple[str, str, str]] = Counter()
    for row in base:
        seen[row.identity] += 1
        survives = max(our_counts[row.identity], their_counts[row.identity])
        if seen[row.identity] <= survives:
            kept.append(row)

    allowance = base_counts.copy()
    added = _additions(ours, allowance)
    added.extend(_additions(theirs, allowance))
    return [*added, *kept]


def union_text(base: str, ours: str, theirs: str) -> str:
    """Return ``ours`` with its change-log table replaced by the row union.

    Only the table is merged here; everything outside it is the caller's
    problem (``scripts/spec_rows_merge_driver.py`` delegates the remainder to
    ``git merge-file``).
    """
    our_log = parse_changelog(ours)
    try:
        base_rows = parse_changelog(base).rows
    except SpecChangelogError:
        base_rows = []
    try:
        their_rows = parse_changelog(theirs).rows
    except SpecChangelogError:
        their_rows = []
    return replace_rows(ours, our_log, union_rows(base_rows, our_log.rows, their_rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    # SPEC.md is UTF-8 and full of arrows and box characters; on Windows the
    # default cp1252 stdout would raise UnicodeEncodeError mid-report and make a
    # passing check look like a crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["validate", "migrate", "rows"], help="what to do"
    )
    parser.add_argument("--spec", default="SPEC.md", help="path to SPEC.md")
    parser.add_argument(
        "--write",
        action="store_true",
        help="migrate: write the result instead of printing a dry-run summary",
    )
    args = parser.parse_args(argv)

    spec = Path(args.spec)
    if not spec.is_file():
        print(f"{spec} not found; nothing to do.")
        return 0

    text = _read(spec)

    if args.command == "migrate":
        try:
            migrated, changed = migrate_text(text)
        except SpecChangelogError as exc:
            print(f"ERROR: {spec}: {exc}")
            return 1
        if args.write:
            _write(spec, migrated)
            print(f"{spec}: rewrote {changed} serial-versioned row(s) to PR keys.")
        else:
            print(
                f"{spec}: would rewrite {changed} serial-versioned row(s) "
                "to PR keys (dry run; pass --write)."
            )
        return 0

    try:
        changelog = parse_changelog(text)
    except SpecChangelogError as exc:
        print(f"ERROR: {spec}: {exc}")
        return 1

    if args.command == "rows":
        for row in changelog.rows:
            print(row.render())
        return 0

    failures = validate(changelog)
    if failures:
        print("ERROR: SPEC.md change-log boundary")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"SPEC.md change log OK ({len(changelog.rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
