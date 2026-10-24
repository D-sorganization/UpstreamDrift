#!/usr/bin/env python3
"""Classify a CI Standard run into a tier and a docs-only flag (issue #1915).

Used by the ``changes`` job of ``.github/workflows/ci-standard.yml``. It reads
the changed paths (one per line) on stdin and writes ``GITHUB_OUTPUT`` lines:

* ``tier`` -- ``fast`` for ``pull_request`` and ``push`` (lint, type check and
  the fast tests), ``full`` for everything else (``merge_group``,
  ``workflow_dispatch``): the full suite plus the security scan, performance
  checks and container build. Unknown events fail toward ``full``.
* ``code_changed`` -- ``false`` only on a ``pull_request`` whose every changed
  path matches :data:`DOCS_ONLY_PATTERNS`. It is never ``false`` on
  ``merge_group`` (a queue group can mix PRs) and never ``false`` for an empty
  change set (an unreadable diff must not skip the suite).

Only the standard library is used, so the job needs no third-party action.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from functools import cache
from typing import TextIO

#: Same list, same order, as the ``push.paths-ignore`` filter of
#: ci-standard.yml (a test keeps the two identical). GitHub glob semantics:
#: ``**`` matches across directories, ``*`` within one path segment.
DOCS_ONLY_PATTERNS: tuple[str, ...] = ("docs/**", "**.md", "LICENSE", ".gitignore")

FAST_TIER_EVENTS = frozenset({"pull_request", "push"})
DOCS_FAST_PATH_EVENTS = frozenset({"pull_request"})


@cache
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a GitHub workflow path glob into an anchored regex."""
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts) + r"\Z")


def is_docs_only(paths: Iterable[str]) -> bool:
    """Return True when every non-blank path matches a docs-only pattern.

    Postcondition: an empty (or all-blank) change set returns False.
    """
    cleaned = [path.strip() for path in paths if path.strip()]
    if not cleaned:
        return False
    return all(
        any(_glob_regex(pattern).match(path) for pattern in DOCS_ONLY_PATTERNS)
        for path in cleaned
    )


def tier_for_event(event_name: str) -> str:
    """Return ``fast`` or ``full`` for a GitHub event name."""
    return "fast" if event_name in FAST_TIER_EVENTS else "full"


def classify(event_name: str, paths: Iterable[str]) -> dict[str, str]:
    """Return the ``changes`` job outputs for an event and its changed paths."""
    if not event_name:
        raise ValueError("event_name must be a non-empty GitHub event name")
    docs_only = event_name in DOCS_FAST_PATH_EVENTS and is_docs_only(paths)
    outputs = {
        "tier": tier_for_event(event_name),
        "code_changed": "false" if docs_only else "true",
    }
    assert outputs["code_changed"] == "true" or outputs["tier"] == "fast"
    return outputs


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True, help="github.event_name")
    args = parser.parse_args(argv)
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    outputs = classify(args.event, source.read().splitlines())
    sink.write("".join(f"{key}={value}\n" for key, value in outputs.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
